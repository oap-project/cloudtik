import os
from typing import Any, Dict

from cloudtik.core._private.core_utils import double_quote
from cloudtik.core._private.providers import _get_workspace_provider
from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_METASTORE
from cloudtik.core._private.utils import is_runtime_enabled, \
    get_node_type, get_resource_of_node_type, RUNTIME_CONFIG_KEY, get_node_type_config, get_config_for_update

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["com.facebook.presto.server.PrestoServer", False, "PrestoServer", "node"],
]

PRESTO_RUNTIME_CONFIG_KEY = "presto"
PRESTO_HIVE_METASTORE_URI_KEY = "hive_metastore_uri"

JVM_MAX_MEMORY_RATIO = 0.8
QUERY_MAX_MEMORY_PER_NODE_RATIO = 0.5
QUERY_MAX_TOTAL_MEMORY_PER_NODE_RATIO = 0.7
MEMORY_HEAP_HEADROOM_PER_NODE_RATIO = 0.25


def get_jvm_max_memory(total_memory):
    return int(total_memory * JVM_MAX_MEMORY_RATIO)


def get_query_max_memory_per_node(jvm_max_memory):
    return int(jvm_max_memory * QUERY_MAX_MEMORY_PER_NODE_RATIO)


def get_query_max_total_memory_per_node(jvm_max_memory):
    return int(jvm_max_memory * QUERY_MAX_TOTAL_MEMORY_PER_NODE_RATIO)


def get_memory_heap_headroom_per_node(jvm_max_memory):
    return int(jvm_max_memory * MEMORY_HEAP_HEADROOM_PER_NODE_RATIO)


def _config_depended_services(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    workspace_name = cluster_config.get("workspace_name")
    if workspace_name is None:
        return cluster_config

    runtime_config = get_config_for_update(cluster_config, RUNTIME_CONFIG_KEY)
    presto_config = get_config_for_update(runtime_config, PRESTO_RUNTIME_CONFIG_KEY)

    workspace_provider = _get_workspace_provider(cluster_config["provider"], workspace_name)
    global_variables = workspace_provider.subscribe_global_variables(cluster_config)

    # Check metastore
    if not is_runtime_enabled(runtime_config, "metastore"):
        if presto_config.get(PRESTO_HIVE_METASTORE_URI_KEY) is None:
            if presto_config.get("auto_detect_metastore", True):
                hive_metastore_uri = global_variables.get("hive-metastore-uri")
                if hive_metastore_uri is not None:
                    presto_config[PRESTO_HIVE_METASTORE_URI_KEY] = hive_metastore_uri

    return cluster_config


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _is_runtime_scripts(script_file):
    if script_file.endswith(".presto.sql"):
        return True

    return False


def _get_runnable_command(target):
    command_parts = ["presto", "-f", double_quote(target)]
    return command_parts


def _with_runtime_environment_variables(runtime_config, config, provider, node_id: str):
    runtime_envs = {"PRESTO_ENABLED": True}
    presto_config = runtime_config.get(PRESTO_RUNTIME_CONFIG_KEY, {})
    cluster_runtime_config = config.get(RUNTIME_CONFIG_KEY)

    # 1) Try to use local metastore if there is one started;
    # 2) Try to use defined metastore_uri;
    if is_runtime_enabled(cluster_runtime_config, BUILT_IN_RUNTIME_METASTORE):
        runtime_envs["METASTORE_ENABLED"] = True
    elif presto_config.get(PRESTO_HIVE_METASTORE_URI_KEY) is not None:
        runtime_envs["HIVE_METASTORE_URI"] = presto_config.get(PRESTO_HIVE_METASTORE_URI_KEY)

    _with_memory_configurations(
        runtime_envs, presto_config=presto_config,
        config=config, provider=provider, node_id=node_id)

    # We need export the cloud storage
    node_type_config = get_node_type_config(config, provider, node_id)
    provider_envs = provider.with_environment_variables(node_type_config, node_id)
    runtime_envs.update(provider_envs)

    return runtime_envs


def _get_runtime_logs():
    logs_dir = os.path.join(os.getenv("PRESTO_HOME"), "logs")
    all_logs = {"presto": logs_dir}
    return all_logs


def _get_head_service_urls(cluster_head_ip):
    services = {
        "presto": {
            "name": "Presto Web UI",
            "url": "http://{}:8081".format(cluster_head_ip)
        },
    }
    return services


def _with_memory_configurations(
        runtime_envs: Dict[str, Any], presto_config: Dict[str, Any],
        config: Dict[str, Any], provider, node_id: str):
    # Set query_max_memory
    query_max_memory = presto_config.get("query_max_memory", "50GB")
    runtime_envs["PRESTO_QUERY_MAX_MEMORY"] = query_max_memory

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
    query_max_total_memory_per_node = get_query_max_total_memory_per_node(jvm_max_memory)

    runtime_envs["PRESTO_JVM_MAX_MEMORY"] = jvm_max_memory
    runtime_envs["PRESTO_MAX_MEMORY_PER_NODE"] = query_max_memory_per_node
    runtime_envs["PRESTO_MAX_TOTAL_MEMORY_PER_NODE"] = query_max_total_memory_per_node
    runtime_envs["PRESTO_HEAP_HEADROOM_PER_NODE"] = \
        get_memory_heap_headroom_per_node(jvm_max_memory)


def configure_connectors(runtime_config: Dict[str, Any]):
    if runtime_config is None:
        return

    presto_config = runtime_config.get(PRESTO_RUNTIME_CONFIG_KEY)
    if presto_config is None:
        return

    catalogs = presto_config.get("catalogs")
    if catalogs is None:
        return

    for catalog in catalogs:
        catalog_config = catalogs[catalog]
        configure_connector(catalog, catalog_config)


def configure_connector(catalog: str, catalog_config: Dict[str, Any]):
    catalog_filename = f"{catalog}.properties"
    catalog_properties_file = os.path.join(
        os.getenv("PRESTO_HOME"), "etc/catalog", catalog_filename)

    mode = 'a' if os.path.exists(catalog_properties_file) else 'w'
    with open(catalog_properties_file, mode) as f:
        for key, value in catalog_config.items():
            f.write("{}={}\n".format(key, value))


def _get_head_service_ports(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    service_ports = {
        "presto": {
            "protocol": "TCP",
            "port": 8081,
        },
    }
    return service_ports
