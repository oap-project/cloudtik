import os
from typing import Any, Dict

from cloudtik.core._private.utils import merge_rooted_config_hierarchy, _get_runtime_config_object

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["gmetad", False, "GangliaMeta", "head"],
    ["gmond", False, "GangliaMonitor", "node"],
]

RUNTIME_ROOT_PATH = os.path.abspath(os.path.dirname(__file__))


def _config_runtime_resources(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    return cluster_config


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _is_runtime_scripts(script_file):
    return False


def _get_runnable_command(target):
    return None


def _with_runtime_environment_variables(runtime_config, provider):
    return {}


def _get_runtime_logs():
    return []


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
    urls =[
        {"name": "Yarn Web UI", "url": "http://{}:8088".format(cluster_head_ip)},
    ]
    return urls
