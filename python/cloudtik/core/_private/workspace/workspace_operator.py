import copy
import hashlib
import json
import logging
import os
import tempfile
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import prettytable as pt

import yaml

from cloudtik.core._private.cluster.cluster_operator import _get_cluster_info
from cloudtik.core.workspace_provider import Existence, CLOUDTIK_MANAGED_CLOUD_STORAGE, \
    CLOUDTIK_MANAGED_CLOUD_STORAGE_URI

try:  # py3
    from shlex import quote
except ImportError:  # py2
    from pipes import quote


from cloudtik.core._private.utils import validate_workspace_config, prepare_workspace_config, is_managed_cloud_storage, \
    print_dict_info, decrypt_config, encrypt_config
from cloudtik.core._private.providers import _get_workspace_provider_cls, _get_workspace_provider, \
    _WORKSPACE_PROVIDERS, _PROVIDER_PRETTY_NAMES, _get_node_provider_cls

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


def update_workspace_firewalls(
        config_file: str, yes: bool,
        override_workspace_name: Optional[str] = None):
    """Update the firewall rules of the workspace."""
    config = _load_workspace_config(config_file, override_workspace_name)

    cli_logger.confirm(yes, "Are you sure that you want to update the firewalls of  workspace {}?",
                       config["workspace_name"], _abort=True)

    _update_workspace_firewalls(config)


def _update_workspace_firewalls(config: Dict[str, Any]):
    provider = _get_workspace_provider(config["provider"], config["workspace_name"])

    if provider.check_workspace_integrity(config):
        provider.update_workspace_firewalls(config)
    else:
        raise RuntimeError(
            "Workspace with the name '{}' doesn't exist! ".format(config["workspace_name"]))


def delete_workspace(
        config_file: str, yes: bool,
        override_workspace_name: Optional[str] = None,
        delete_managed_storage: bool = False):
    """Destroys the workspace and associated Cloud resources."""
    config = _load_workspace_config(config_file, override_workspace_name)
    _delete_workspace(config, yes, delete_managed_storage)


def _delete_workspace(config: Dict[str, Any],
                      yes: bool = False,
                      delete_managed_storage: bool = False):
    workspace_name = config["workspace_name"]
    provider = _get_workspace_provider(config["provider"], workspace_name)
    existence = provider.check_workspace_existence(config)
    if existence == Existence.NOT_EXIST:
        raise RuntimeError(f"Workspace with the name {workspace_name} doesn't exist!")
    else:
        if existence == Existence.COMPLETED:
            # Only check the running cluster when the workspace is at completed status
            running_clusters = provider.list_clusters(config)
            if running_clusters is not None and len(running_clusters) > 0:
                cluster_names = ",".join(list(running_clusters.keys()))
                raise RuntimeError(
                    "Workspace {} has clusters ({}) in running. Please stop the clusters first.".format(
                        workspace_name, cluster_names
                    ))

        managed_cloud_storage = is_managed_cloud_storage(config)
        if managed_cloud_storage:
            if delete_managed_storage:
                cli_logger.warning("WARNING: The managed cloud storage associated with this workspace "
                                   "and the data in it will all be deleted!")
            else:
                cli_logger.print("The managed cloud storage associated with this workspace will not be deleted.")

        cli_logger.confirm(yes, "Are you sure that you want to delete workspace {}?",
                           config["workspace_name"], _abort=True)
        provider.delete_workspace(config, delete_managed_storage)


def create_workspace(
        config_file: str, yes: bool,
        override_workspace_name: Optional[str] = None,
        no_config_cache: bool = False):
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
                                         no_config_cache=no_config_cache)
    _create_workspace(config, yes)


def _create_workspace(config: Dict[str, Any], yes: bool = False):
    workspace_name = config["workspace_name"]
    provider = _get_workspace_provider(config["provider"], workspace_name)
    existence = provider.check_workspace_existence(config)
    if existence == Existence.COMPLETED:
        raise RuntimeError(f"A completed workspace with the name {workspace_name} already exists!")
    elif existence == Existence.IN_COMPLETED:
        raise RuntimeError(f"A workspace with the name {workspace_name} already exists but not completed!")
    else:
        cli_logger.confirm(yes, "Are you sure that you want to create workspace {}?",
                           config["workspace_name"], _abort=True)
        provider.create_workspace(config)


def list_workspace_clusters(
        config_file: str,
        override_workspace_name: Optional[str] = None):
    """List clusters of the workspace name."""
    config = _load_workspace_config(config_file, override_workspace_name)
    clusters = _list_workspace_clusters(config)
    if clusters is None:
        cli_logger.print("Workspace {} is not correctly configured.", config["workspace_name"])
    elif len(clusters) == 0:
        cli_logger.print("Workspace {} has no cluster in running.", config["workspace_name"])
    else:
        # Get cluster info by the cluster name
        clusters_info = _get_clusters_info(config, clusters)
        _show_clusters(clusters_info)


def _get_clusters_info(config: Dict[str, Any], clusters):
    clusters_info = []
    for cluster_name in clusters:
        cluster_info = {"cluster_name": cluster_name,
                        "head_node": clusters[cluster_name]}

        # Retrieve other information through cluster operator
        cluster_config = copy.deepcopy(config)
        cluster_config["cluster_name"] = cluster_name

        # Needs to do a provider bootstrap of the config for fill the missing configurations
        provider_cls = _get_node_provider_cls(cluster_config["provider"])
        cluster_config = provider_cls.bootstrap_config_for_api(cluster_config)

        info = _get_cluster_info(cluster_config, simple_config=True)
        cluster_info["total-workers"] = info.get("total-workers", 0)
        cluster_info["total-workers-ready"] = info.get("total-workers-ready", 0)
        cluster_info["total-workers-failed"] = info.get("total-workers-failed", 0)

        clusters_info.append(cluster_info)

    # sort cluster info based cluster name
    def cluster_info_sort(cluster_info):
        return cluster_info["cluster_name"]

    clusters_info.sort(key=cluster_info_sort)
    return clusters_info


def _show_clusters(clusters_info):
    tb = pt.PrettyTable()
    tb.field_names = ["cluster-name", "head-node-ip", "head-status", "head-public-ip",
                      "total-workers", "workers-ready", "workers-failed"]
    for cluster_info in clusters_info:
        tb.add_row([cluster_info["cluster_name"], cluster_info["head_node"]["private_ip"],
                    cluster_info["head_node"]["cloudtik-node-status"], cluster_info["head_node"]["public_ip"],
                    cluster_info["total-workers"], cluster_info["total-workers-ready"],
                    cluster_info["total-workers-failed"]
                    ])

    cli_logger.print("{} cluster(s) are running.", len(clusters_info))
    cli_logger.print(tb)


def _list_workspace_clusters(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    provider = _get_workspace_provider(config["provider"], config["workspace_name"])
    if not provider.check_workspace_integrity(config):
        return None

    return provider.list_clusters(config)


def show_status(
        config_file: str,
        override_workspace_name: Optional[str] = None):
    """Show workspace status."""
    config = _load_workspace_config(config_file, override_workspace_name)
    workspace_name = config["workspace_name"]
    existence = _get_workspace_status(config)
    if existence == Existence.NOT_EXIST:
        cli_logger.print(f"Workspace {workspace_name}: NOT EXIST.")
    elif existence == Existence.STORAGE_ONLY:
        cli_logger.print(f"Workspace {workspace_name}: STORAGE ONLY.")
    elif existence == Existence.IN_COMPLETED:
        cli_logger.print(f"Workspace {workspace_name}: NOT COMPLETED.")
    else:
        cli_logger.print(f"Workspace {workspace_name}: COMPLETED.")


def _get_workspace_status(config):
    workspace_name = config["workspace_name"]
    provider = _get_workspace_provider(config["provider"], workspace_name)
    return provider.check_workspace_existence(config)


def get_workspace_info(
        config_file: str,
        override_workspace_name: Optional[str] = None):
    config = _load_workspace_config(config_file, override_workspace_name)
    return _get_workspace_info(config)


def _get_workspace_info(
        config: Dict[str, Any]):
    provider = _get_workspace_provider(config["provider"], config["workspace_name"])
    return provider.get_workspace_info(config)


def show_workspace_info(
        config_file: str,
        override_workspace_name: Optional[str] = None):
    show_status(config_file, override_workspace_name)
    workspace_info = get_workspace_info(config_file, override_workspace_name)
    print_dict_info(workspace_info)


def show_managed_cloud_storage(
        config_file: str,
        override_workspace_name: Optional[str] = None):
    workspace_info = get_workspace_info(config_file, override_workspace_name)
    managed_cloud_storage = workspace_info.get(CLOUDTIK_MANAGED_CLOUD_STORAGE)
    if managed_cloud_storage is not None:
        print_dict_info(managed_cloud_storage)


def show_managed_cloud_storage_uri(
        config_file: str,
        override_workspace_name: Optional[str] = None):
    workspace_info = get_workspace_info(config_file, override_workspace_name)
    managed_cloud_storage = workspace_info.get(CLOUDTIK_MANAGED_CLOUD_STORAGE)
    if managed_cloud_storage is not None:
        cli_logger.print(managed_cloud_storage[CLOUDTIK_MANAGED_CLOUD_STORAGE_URI])


CONFIG_CACHE_VERSION = 1


def _bootstrap_workspace_config(config: Dict[str, Any],
                                no_config_cache: bool = False) -> Dict[str, Any]:
    config = prepare_workspace_config(config)
    # Note: delete workspace only need to contain workspace_name

    hasher = hashlib.sha1()
    hasher.update(json.dumps([config], sort_keys=True).encode("utf-8"))
    cache_key = os.path.join(tempfile.gettempdir(),
                             "cloudtik-workspace-config-{}".format(hasher.hexdigest()))

    provider_cls = _get_workspace_provider_cls(config["provider"])

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

    if not no_config_cache:
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


def _load_workspace_config(config_file: str,
                           override_workspace_name: Optional[str] = None,
                           should_bootstrap: bool = True,
                           no_config_cache: bool = False) -> Dict[str, Any]:
    config = yaml.safe_load(open(config_file).read())
    if override_workspace_name is not None:
        config["workspace_name"] = override_workspace_name
    if should_bootstrap:
        config = _bootstrap_workspace_config(config, no_config_cache=no_config_cache)
    return config
