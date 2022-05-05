import os
from typing import Any, Dict

from cloudtik.core._private.providers import _get_workspace_provider
from cloudtik.core._private.utils import merge_rooted_config_hierarchy, _get_runtime_config_object, is_runtime_enabled

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["com.facebook.presto.server.PrestoServer", False, "PrestoServer", "node"],
]

RUNTIME_ROOT_PATH = os.path.abspath(os.path.dirname(__file__))


def _config_runtime_resources(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    return cluster_config


def _config_depended_services(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    runtime_config = cluster_config.get("runtime")
    if "presto" not in runtime_config:
        runtime_config["presto"] = {}
    presto_config = runtime_config["presto"]

    workspace_name = cluster_config.get("workspace_name", "")
    workspace_provider = _get_workspace_provider(cluster_config["provider"], workspace_name)
    global_variables = workspace_provider.subscribe_global_variables(cluster_config)

    # Check metastore
    if not is_runtime_enabled(runtime_config, "metastore"):
        if presto_config.get("hive_metastore_uri") is None:
            hive_metastore_uri = global_variables.get("hive-metastore-uri")
            if hive_metastore_uri is not None:
                presto_config["hive_metastore_uri"] = hive_metastore_uri

    return cluster_config


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _is_runtime_scripts(script_file):
    if script_file.endswith(".presto.sql"):
        return True

    return False


def _get_runnable_command(target):
    command_parts = ["presto", "-f", target]
    return command_parts


def _with_runtime_environment_variables(runtime_config, provider):
    runtime_envs = {"PRESTO_ENABLED": True}
    presto_config = runtime_config.get("presto", {})

    # 1) Try to use local metastore if there is one started;
    # 2) Try to use defined metastore_uri;
    if is_runtime_enabled(runtime_config, "metastore"):
        runtime_envs["METASTORE_ENABLED"] = True
    elif presto_config.get("hive_metastore_uri") is not None:
        runtime_envs["HIVE_METASTORE_URI"] = presto_config.get("hive_metastore_uri")
    return runtime_envs


def _get_runtime_logs():
    logs_dir = os.path.join(os.getenv("PRESTO_HOME"), "logs")
    all_logs = {"presto": logs_dir}
    return all_logs


def _validate_config(config: Dict[str, Any], provider):
    pass


def _verify_config(config: Dict[str, Any], provider):
    pass


def _get_config_object(cluster_config: Dict[str, Any], object_name: str) -> Dict[str, Any]:
    config_root = os.path.join(RUNTIME_ROOT_PATH, "config")
    runtime_commands = _get_runtime_config_object(config_root, cluster_config["provider"], object_name)
    return merge_rooted_config_hierarchy(config_root, runtime_commands, object_name)


def _get_runtime_commands(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    return _get_config_object(cluster_config, "commands")


def _get_defaults_config(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    return _get_config_object(cluster_config, "defaults")


def _get_useful_urls(cluster_head_ip):
    urls = [
        {"name": "Presto Web UI", "url": "http://{}:8080".format(cluster_head_ip)},
    ]
    return urls
