import os
from typing import Any, Dict

from cloudtik.core._private.core_utils import double_quote
from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_METASTORE, BUILT_IN_RUNTIME_PRESTO
from cloudtik.core._private.service_discovery.utils import get_canonical_service_name, define_runtime_service_on_head, \
    get_service_discovery_config, SERVICE_DISCOVERY_FEATURE_ANALYTICS
from cloudtik.core._private.utils import \
    get_node_type, get_resource_of_node_type, RUNTIME_CONFIG_KEY
from cloudtik.runtime.common.service_discovery.cluster import has_runtime_in_cluster
from cloudtik.runtime.common.service_discovery.runtime_discovery import \
    discover_metastore_from_workspace, discover_metastore_on_head, METASTORE_URI_KEY

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["com.facebook.presto.server.PrestoServer", False, "PrestoServer", "node"],
]

JVM_MAX_MEMORY_RATIO = 0.8
QUERY_MAX_MEMORY_PER_NODE_RATIO = 0.5
QUERY_MAX_TOTAL_MEMORY_PER_NODE_RATIO = 0.7
MEMORY_HEAP_HEADROOM_PER_NODE_RATIO = 0.25

PRESTO_SERVICE_NAME = BUILT_IN_RUNTIME_PRESTO
PRESTO_SERVICE_PORT = 8081


def _get_config(runtime_config: Dict[str, Any]):
    return runtime_config.get(BUILT_IN_RUNTIME_PRESTO, {})


def get_jvm_max_memory(total_memory):
    return int(total_memory * JVM_MAX_MEMORY_RATIO)


def get_query_max_memory_per_node(jvm_max_memory):
    return int(jvm_max_memory * QUERY_MAX_MEMORY_PER_NODE_RATIO)


def get_query_max_total_memory_per_node(jvm_max_memory):
    return int(jvm_max_memory * QUERY_MAX_TOTAL_MEMORY_PER_NODE_RATIO)


def get_memory_heap_headroom_per_node(jvm_max_memory):
    return int(jvm_max_memory * MEMORY_HEAP_HEADROOM_PER_NODE_RATIO)


def _config_depended_services(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    cluster_config = discover_metastore_from_workspace(
        cluster_config, BUILT_IN_RUNTIME_PRESTO)
    return cluster_config


def _prepare_config_on_head(cluster_config: Dict[str, Any]):
    cluster_config = discover_metastore_on_head(
        cluster_config, BUILT_IN_RUNTIME_PRESTO)
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


def _with_runtime_environment_variables(
        runtime_config, config, provider, node_id: str):
    runtime_envs = {"PRESTO_ENABLED": True}
    presto_config = _get_config(runtime_config)
    cluster_runtime_config = config.get(RUNTIME_CONFIG_KEY)

    if has_runtime_in_cluster(
            cluster_runtime_config, BUILT_IN_RUNTIME_METASTORE):
        runtime_envs["METASTORE_ENABLED"] = True

    _with_memory_configurations(
        runtime_envs, presto_config=presto_config,
        config=config, provider=provider, node_id=node_id)

    return runtime_envs


def _configure(runtime_config, head: bool):
    # TODO: move more runtime specific environment_variables to here
    presto_config = _get_config(runtime_config)
    metastore_uri = presto_config.get(METASTORE_URI_KEY)
    if metastore_uri:
        os.environ["HIVE_METASTORE_URI"] = metastore_uri


def _get_runtime_logs():
    logs_dir = os.path.join(os.getenv("PRESTO_HOME"), "logs")
    all_logs = {"presto": logs_dir}
    return all_logs


def _get_runtime_endpoints(cluster_head_ip):
    endpoints = {
        "presto": {
            "name": "Presto Web UI",
            "url": "http://{}:{}".format(cluster_head_ip, PRESTO_SERVICE_PORT)
        },
    }
    return endpoints


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

    presto_config = runtime_config.get(BUILT_IN_RUNTIME_PRESTO)
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
            "port": PRESTO_SERVICE_PORT,
        },
    }
    return service_ports


def _get_runtime_services(
        runtime_config: Dict[str, Any], cluster_name: str) -> Dict[str, Any]:
    presto_config = _get_config(runtime_config)
    service_discovery_config = get_service_discovery_config(presto_config)
    service_name = get_canonical_service_name(
        service_discovery_config, cluster_name, PRESTO_SERVICE_NAME)
    services = {
        service_name: define_runtime_service_on_head(
            service_discovery_config, PRESTO_SERVICE_PORT,
            features=[SERVICE_DISCOVERY_FEATURE_ANALYTICS]),
    }
    return services
