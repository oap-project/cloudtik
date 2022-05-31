import os
from typing import Any, Dict

from cloudtik.core._private.providers import _get_workspace_provider
from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_METASTORE
from cloudtik.core._private.utils import merge_rooted_config_hierarchy, _get_runtime_config_object, is_runtime_enabled, \
    get_node_type, get_resource_of_node_type, round_memory_size_to_gb, RUNTIME_CONFIG_KEY, get_node_type_config

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["io.trino.server.TrinoServer", False, "TrinoServer", "node"],
]

RUNTIME_ROOT_PATH = os.path.abspath(os.path.dirname(__file__))

JVM_MAX_MEMORY_RATIO = 0.8
QUERY_MAX_MEMORY_PER_NODE_RATIO = 0.5
MEMORY_HEAP_HEADROOM_PER_NODE_RATIO = 0.25


def get_jvm_max_memory(total_memory):
    return int(total_memory * JVM_MAX_MEMORY_RATIO)


def get_query_max_memory_per_node(jvm_max_memory):
    return int(jvm_max_memory * QUERY_MAX_MEMORY_PER_NODE_RATIO)


def get_memory_heap_headroom_per_node(jvm_max_memory):
    return int(jvm_max_memory * MEMORY_HEAP_HEADROOM_PER_NODE_RATIO)


def _config_runtime_resources(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    return cluster_config


def _config_depended_services(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    runtime_config = cluster_config.get(RUNTIME_CONFIG_KEY)
    if "trino" not in runtime_config:
        runtime_config["trino"] = {}
    trino_config = runtime_config["trino"]

    workspace_name = cluster_config.get("workspace_name", "")
    workspace_provider = _get_workspace_provider(cluster_config["provider"], workspace_name)
    global_variables = workspace_provider.subscribe_global_variables(cluster_config)

    # Check metastore
    if not is_runtime_enabled(runtime_config, "metastore"):
        if trino_config.get("hive_metastore_uri") is None:
            if trino_config.get("auto_detect_metastore", True):
                hive_metastore_uri = global_variables.get("hive-metastore-uri")
                if hive_metastore_uri is not None:
                    trino_config["hive_metastore_uri"] = hive_metastore_uri

    return cluster_config


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _is_runtime_scripts(script_file):
    if script_file.endswith(".trino.sql"):
        return True

    return False


def _get_runnable_command(target):
    command_parts = ["trino", "-f", target]
    return command_parts


def _with_runtime_environment_variables(runtime_config, config, provider, node_id: str):
    runtime_envs = {"TRINO_ENABLED": True}
    trino_config = runtime_config.get("trino", {})
    cluster_runtime_config = config.get(RUNTIME_CONFIG_KEY)

    # 1) Try to use local metastore if there is one started;
    # 2) Try to use defined metastore_uri;
    if is_runtime_enabled(cluster_runtime_config, BUILT_IN_RUNTIME_METASTORE):
        runtime_envs["METASTORE_ENABLED"] = True
    elif trino_config.get("hive_metastore_uri") is not None:
        runtime_envs["HIVE_METASTORE_URI"] = trino_config.get("hive_metastore_uri")

    _with_memory_configurations(
        runtime_envs, trino_config=trino_config,
        config=config, provider=provider, node_id=node_id)

    # We need export the cloud storage
    node_type_config = get_node_type_config(config, provider, node_id)
    provider_envs = provider.with_environment_variables(node_type_config, node_id)
    runtime_envs.update(provider_envs)

    return runtime_envs


def _get_runtime_logs():
    logs_dir = os.path.join(os.getenv("TRINO_HOME"), "logs")
    all_logs = {"trino": logs_dir}
    return all_logs


def _validate_config(config: Dict[str, Any], provider):
    pass


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
    urls = [
        {"name": "Trino Web UI", "url": "http://{}:8080".format(cluster_head_ip)},
    ]
    return urls


def _with_memory_configurations(
        runtime_envs: Dict[str, Any], trino_config: Dict[str, Any],
        config: Dict[str, Any], provider, node_id: str):
    # Set query_max_memory
    query_max_memory = trino_config.get("query_max_memory", "50GB")
    runtime_envs["TRINO_QUERY_MAX_MEMORY"] = query_max_memory

    node_type = get_node_type(provider, node_id)
    if node_type is None:
        return

    # Get memory of node type
    resources = get_resource_of_node_type(config, node_type)
    if resources is None:
        return

    memory_in_mb = int(resources.get("memory", 0) / (1024 * 1024))
    if memory_in_mb == 0:
        return

    jvm_max_memory = get_jvm_max_memory(memory_in_mb)
    query_max_memory_per_node = get_query_max_memory_per_node(jvm_max_memory)

    runtime_envs["TRINO_JVM_MAX_MEMORY"] = jvm_max_memory
    runtime_envs["TRINO_MAX_MEMORY_PER_NODE"] = query_max_memory_per_node
    runtime_envs["TRINO_HEAP_HEADROOM_PER_NODE"] = \
        get_memory_heap_headroom_per_node(jvm_max_memory)


def configure_connectors(runtime_config: Dict[str, Any]):
    if runtime_config is None:
        return

    trino_config = runtime_config.get("trino")
    if trino_config is None:
        return

    catalogs = trino_config.get("catalogs")
    if catalogs is None:
        return

    for catalog in catalogs:
        catalog_config = catalogs[catalog]
        configure_connector(catalog, catalog_config)


def configure_connector(catalog: str, catalog_config: Dict[str, Any]):
    catalog_filename = f"{catalog}.properties"
    catalog_properties_file = os.path.join(
        os.getenv("TRINO_HOME"), "etc/catalog", catalog_filename)

    mode = 'a' if os.path.exists(catalog_properties_file) else 'w'
    with open(catalog_properties_file, mode) as f:
        for key, value in catalog_config.items():
            f.write("{}={}\n".format(key, value))
