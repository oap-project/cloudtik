import hashlib
import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
try:  # py3
    from shlex import quote
except ImportError:  # py2
    from pipes import quote


from cloudtik.core._private.utils import validate_workspace_config, prepare_workspace_config
from cloudtik.core._private.providers import _get_workspace_provider_cls, _get_workspace_provider, \
    _WORKSPACE_PROVIDERS, _PROVIDER_PRETTY_NAMES

from cloudtik.core._private.cli_logger import cli_logger, cf

from cloudtik.core._private.debug import log_once


logger = logging.getLogger(__name__)

RUN_ENV_TYPES = ["auto", "host", "docker"]

POLL_INTERVAL = 5

Port_forward = Union[Tuple[int, int], List[Tuple[int, int]]]


def try_logging_config(config: Dict[str, Any]) -> None:
    if config["provider"]["type"] == "aws":
        from cloudtik.providers._private.aws.config import log_to_cli
        log_to_cli(config)


def try_get_log_state(provider_config: Dict[str, Any]) -> Optional[dict]:
    if provider_config["type"] == "aws":
        from cloudtik.providers._private.aws.config import get_log_state
        return get_log_state()
    return None


def try_reload_log_state(provider_config: Dict[str, Any],
                         log_state: dict) -> None:
    if not log_state:
        return
    if provider_config["type"] == "aws":
        from cloudtik.providers._private.aws.config import reload_log_state
        return reload_log_state(log_state)


def delete_workspace(
        config_file: str, yes: bool,
        override_workspace_name: Optional[str] = None) -> Dict[str, Any]:
    """Destroys the workspace and associated Cloud resources."""
    config = yaml.safe_load(open(config_file).read())
    if override_workspace_name is not None:
        config["workspace_name"] = override_workspace_name

    config = _bootstrap_workspace_config(config)

    cli_logger.confirm(yes, "Destroying workspace.", _abort=True)

    provider = _get_workspace_provider(config["provider"], config["workspace_name"])

    provider.delete_workspace(config)


def create_or_update_workspace(
        config_file: str,
        override_workspace_name: Optional[str] = None,
        no_workspace_config_cache: bool = False) -> Dict[str, Any]:
    """Creates or updates an scaling cluster from a config json."""

    def handle_yaml_error(e):
        cli_logger.error("Workspace config invalid")
        cli_logger.newline()
        cli_logger.error("Failed to load YAML file " + cf.bold("{}"),
                         config_file)
        cli_logger.newline()
        with cli_logger.verbatim_error_ctx("PyYAML error:"):
            cli_logger.error(e)
        cli_logger.abort()

    try:
        config = yaml.safe_load(open(config_file).read())
    except FileNotFoundError:
        cli_logger.abort(
            "Provided workspace configuration file ({}) does not exist",
            cf.bold(config_file))
    except yaml.parser.ParserError as e:
        handle_yaml_error(e)
        raise
    except yaml.scanner.ScannerError as e:
        handle_yaml_error(e)
        raise

    importer = _WORKSPACE_PROVIDERS.get(config["provider"]["type"])
    if not importer:
        cli_logger.abort(
            "Unknown provider type " + cf.bold("{}") + "\n"
            "Available providers are: {}", config["provider"]["type"],
            cli_logger.render_list([
                k for k in _WORKSPACE_PROVIDERS.keys()
                if _WORKSPACE_PROVIDERS[k] is not None
            ]))

    printed_overrides = False

    def handle_cli_override(key, override):
        if override is not None:
            if key in config:
                nonlocal printed_overrides
                printed_overrides = True
                cli_logger.warning(
                    "`{}` override provided on the command line.\n"
                    "  Using " + cf.bold("{}") + cf.dimmed(
                        " [configuration file has " + cf.bold("{}") + "]"),
                    key, override, config[key])
            config[key] = override

    handle_cli_override("workspace_name", override_workspace_name)

    if printed_overrides:
        cli_logger.newline()

    cli_logger.labeled_value("Workspace", config["workspace_name"])

    cli_logger.newline()
    config = _bootstrap_workspace_config(config,
        no_workspace_config_cache=no_workspace_config_cache)

    provider = _get_workspace_provider(config["provider"], config["workspace_name"])

    if provider.check_workspace_resource(config):
        cli_logger.print("workspace resource has existed, no need to recreate workspace")
    else:
        provider.create_workspace(config)


CONFIG_CACHE_VERSION = 1


def _bootstrap_workspace_config(config: Dict[str, Any],
                                no_workspace_config_cache: bool = False) -> Dict[str, Any]:
    config = prepare_workspace_config(config)
    # Note: delete workspace only need to contain workspace_name

    hasher = hashlib.sha1()
    hasher.update(json.dumps([config], sort_keys=True).encode("utf-8"))
    cache_key = os.path.join(tempfile.gettempdir(),
                             "cloudtik-workspace-config-{}".format(hasher.hexdigest()))

    provider_cls = _get_workspace_provider_cls(config["provider"])

    if os.path.exists(cache_key) and not no_workspace_config_cache:
        config_cache = json.loads(open(cache_key).read())
        if config_cache.get("_version", -1) == CONFIG_CACHE_VERSION :
            # todo: is it fine to re-resolve? afaik it should be.
            # we can have migrations otherwise or something
            # but this seems overcomplicated given that resolving is
            # relatively cheap
            try_reload_log_state(config_cache["config"]["provider"],
                                 config_cache.get("provider_log_info"))
            if log_once("_printed_cached_config_warning"):
                cli_logger.verbose_warning(
                    "Loaded cached provider configuration "
                    "from " + cf.bold("{}"), cache_key)
                if cli_logger.verbosity == 0:
                    cli_logger.warning("Loaded cached provider configuration")
                cli_logger.warning(
                    "If you experience issues with "
                    "the cloud provider, try re-running "
                    "the command with {}.", cf.bold("--no-workspace-config-cache"))
            return config_cache["config"]
        else:
            cli_logger.warning(
                "Found cached workspace config "
                "but the version " + cf.bold("{}") + " "
                "(expected " + cf.bold("{}") + ") does not match.\n"
                "This is normal if cluster launcher was updated.\n"
                "Config will be re-resolved.",
                config_cache.get("_version", "none"), CONFIG_CACHE_VERSION)

    cli_logger.print("Checking {} environment settings",
                     _PROVIDER_PRETTY_NAMES.get(config["provider"]["type"]))

    try:
        # NOTE: if `resources` field is missing, validate_config for providers
        # other than AWS and Kubernetes will fail (the schema error will ask
        # the user to manually fill the resources) as we currently support
        # autofilling resources for AWS and Kubernetes only.
        validate_workspace_config(config)
    except (ModuleNotFoundError, ImportError):
        cli_logger.abort(
            "Not all dependencies were found. Please "
            "update your install command.")

    resolved_config = provider_cls.bootstrap_workspace_config(config)

    if not no_workspace_config_cache:
        with open(cache_key, "w") as f:
            config_cache = {
                "_version": CONFIG_CACHE_VERSION,
                "provider_log_info": try_get_log_state(
                    resolved_config["provider"]),
                "config": resolved_config
            }
            f.write(json.dumps(config_cache))
    return resolved_config
