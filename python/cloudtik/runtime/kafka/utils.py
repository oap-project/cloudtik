import os
from typing import Any, Dict, List

from cloudtik.core._private.providers import _get_workspace_provider
from cloudtik.core._private.utils import merge_rooted_config_hierarchy, _get_runtime_config_object, \
    subscribe_cluster_variable, is_runtime_enabled, RUNTIME_CONFIG_KEY, subscribe_runtime_config, load_properties_file, \
    save_properties_file

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["kafka.Kafka", False, "KafkaBroker", "node"],
]

RUNTIME_ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
CONFIG_KEY_RUNTIME = "kafka"


def _config_runtime_resources(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    return cluster_config


def _config_depended_services(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    workspace_name = cluster_config.get("workspace_name")
    if workspace_name is None:
        return cluster_config

    runtime_config = cluster_config.get(RUNTIME_CONFIG_KEY)
    if CONFIG_KEY_RUNTIME not in runtime_config:
        runtime_config[CONFIG_KEY_RUNTIME] = {}
    kafka_config = runtime_config[CONFIG_KEY_RUNTIME]

    workspace_provider = _get_workspace_provider(cluster_config["provider"], workspace_name)
    global_variables = workspace_provider.subscribe_global_variables(cluster_config)

    # Check zookeeper
    if not is_runtime_enabled(runtime_config, "zookeeper"):
        if kafka_config.get("zookeeper_connect") is None:
            if kafka_config.get("auto_detect_zookeeper", True):
                zookeeper_uri = global_variables.get("zookeeper-uri")
                if zookeeper_uri is not None:
                    kafka_config["zookeeper_connect"] = zookeeper_uri

    return cluster_config


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _is_runtime_scripts(script_file):
    return False


def _get_runnable_command(target):
    return None


def _with_runtime_environment_variables(runtime_config, config, provider, node_id: str):
    runtime_envs = {"KAFKA_ENABLED": True}
    return runtime_envs


def _get_runtime_logs():
    kafka_logs_dir = os.path.join(os.getenv("KAFKA_HOME"), "logs")
    all_logs = {"kafka": kafka_logs_dir}
    return all_logs


def _validate_config(config: Dict[str, Any], provider):
    if not is_runtime_enabled(config.get(RUNTIME_CONFIG_KEY), "zookeeper"):
        # Check zookeeper connect configured
        runtime_config = config.get(RUNTIME_CONFIG_KEY)
        if (runtime_config is None) or (
                CONFIG_KEY_RUNTIME not in runtime_config) or (
                "zookeeper_connect" not in runtime_config[CONFIG_KEY_RUNTIME]):
            raise ValueError("Zookeeper connect must be configured!")


def _verify_config(config: Dict[str, Any], provider):
    pass


def _get_config_object(cluster_config: Dict[str, Any], object_name: str) -> Dict[str, Any]:
    config_root = os.path.join(RUNTIME_ROOT_PATH, "config")
    runtime_commands = _get_runtime_config_object(config_root, cluster_config["provider"], object_name)
    return merge_rooted_config_hierarchy(config_root, runtime_commands, object_name)


def _get_runtime_commands(runtime_config: Dict[str, Any],
                          cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    return _get_config_object(cluster_config, "commands")


def _get_defaults_config(runtime_config: Dict[str, Any],
                         cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    return _get_config_object(cluster_config, "defaults")


def _get_useful_urls(cluster_head_ip):
    # TODO: how to get the Kafka service address which established after head node
    return None


def _get_zookeeper_connect(runtime_config):
    if runtime_config is None:
        return None

    kafka_config = runtime_config.get(CONFIG_KEY_RUNTIME)
    if kafka_config is None:
        return None

    # check config
    zookeeper_connect = kafka_config.get("zookeeper_connect")
    if zookeeper_connect is not None:
        return zookeeper_connect

    # check redis endpoint publish
    zookeeper_connect = subscribe_cluster_variable("zookeeper-uri")
    return zookeeper_connect


def _get_server_config(runtime_config: Dict[str, Any]):
    kafka_config = runtime_config.get("kafka")
    if not kafka_config:
        return None

    return kafka_config.get("config")


def update_configurations():
    # Merge user specified configuration and default configuration
    runtime_config = subscribe_runtime_config()
    server_config = _get_server_config(runtime_config)
    if not server_config:
        return

    server_properties_file = os.path.join(os.getenv("KAFKA_HOME"), "config/server.properties")

    # Read in the existing configurations
    server_properties, comments = load_properties_file(server_properties_file)

    # Merge with the user configurations
    server_properties.update(server_config)

    # Write back the configuration file
    save_properties_file(server_properties_file, server_properties, comments=comments)
