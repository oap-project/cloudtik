import hashlib
import json
import os
import tempfile
from typing import Any, Dict, Optional
from functools import partial
import yaml

from cloudtik.core._private.debug import log_once
from cloudtik.core._private.utils import prepare_config, decrypt_config, runtime_prepare_config, validate_config, \
    verify_config, encrypt_config, RUNTIME_CONFIG_KEY
from cloudtik.core._private.providers import _NODE_PROVIDERS, _PROVIDER_PRETTY_NAMES
from cloudtik.core._private.cli_logger import cli_logger, cf

CONFIG_CACHE_VERSION = 1


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


def _bootstrap_config(config: Dict[str, Any],
                      no_config_cache: bool = False,
                      init_config_cache: bool = False) -> Dict[str, Any]:
    # Check if bootstrapped, return if it is the case
    if config.get("bootstrapped", False):
        return config

    config = prepare_config(config)
    # NOTE: multi-node-type cluster scaler is guaranteed to be in use after this.

    hasher = hashlib.sha1()
    hasher.update(json.dumps([config], sort_keys=True).encode("utf-8"))
    cache_key = os.path.join(tempfile.gettempdir(),
                             "cloudtik-config-{}".format(hasher.hexdigest()))

    if os.path.exists(cache_key) and not no_config_cache:
        config_cache = json.loads(open(cache_key).read())
        if config_cache.get("_version", -1) == CONFIG_CACHE_VERSION:
            # todo: is it fine to re-resolve? afaik it should be.
            # we can have migrations otherwise or something
            # but this seems overcomplicated given that resolving is
            # relatively cheap
            cached_config = decrypt_config(config_cache["config"])
            try_reload_log_state(cached_config["provider"],
                                 config_cache.get("provider_log_info"))

            if log_once("_printed_cached_config_warning"):
                cli_logger.verbose_warning(
                    "Loaded cached provider configuration "
                    "from " + cf.bold("{}"), cache_key)
                cli_logger.verbose_warning(
                    "If you experience issues with "
                    "the cloud provider, try re-running "
                    "the command with {}.", cf.bold("--no-config-cache"))

            return cached_config
        else:
            cli_logger.warning(
                "Found cached cluster config "
                "but the version " + cf.bold("{}") + " "
                "(expected " + cf.bold("{}") + ") does not match.\n"
                "This is normal if cluster launcher was updated.\n"
                "Config will be re-resolved.",
                config_cache.get("_version", "none"), CONFIG_CACHE_VERSION)

    importer = _NODE_PROVIDERS.get(config["provider"]["type"])
    if not importer:
        raise NotImplementedError("Unsupported provider {}".format(
            config["provider"]))

    provider_cls = importer(config["provider"])

    cli_logger.print("Checking {} environment settings",
                     _PROVIDER_PRETTY_NAMES.get(config["provider"]["type"]))

    config = provider_cls.post_prepare(config)
    config = runtime_prepare_config(config.get(RUNTIME_CONFIG_KEY), config)

    try:
        validate_config(config)
    except (ModuleNotFoundError, ImportError):
        cli_logger.abort(
            "Not all dependencies were found. Please "
            "update your install command.")

    resolved_config = provider_cls.bootstrap_config(config)

    # add a verify step
    verify_config(resolved_config)

    if not no_config_cache or init_config_cache:
        with open(cache_key, "w", opener=partial(os.open, mode=0o600)) as f:
            encrypted_config = encrypt_config(resolved_config)
            config_cache = {
                "_version": CONFIG_CACHE_VERSION,
                "provider_log_info": try_get_log_state(
                    resolved_config["provider"]),
                "config": encrypted_config
            }
            f.write(json.dumps(config_cache))
    return resolved_config


def _load_cluster_config(config_file: str,
                         override_cluster_name: Optional[str] = None,
                         should_bootstrap: bool = True,
                         no_config_cache: bool = False) -> Dict[str, Any]:
    config = yaml.safe_load(open(config_file).read())
    if override_cluster_name is not None:
        config["cluster_name"] = override_cluster_name
    if should_bootstrap:
        config = _bootstrap_config(config, no_config_cache=no_config_cache)
    return config
