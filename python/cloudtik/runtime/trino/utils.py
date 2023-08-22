import os
from typing import Any, Dict

from cloudtik.core._private.core_utils import double_quote
from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_METASTORE, BUILT_IN_RUNTIME_TRINO
from cloudtik.core._private.service_discovery.utils import get_canonical_service_name, define_runtime_service_on_head, \
    get_service_discovery_config, SERVICE_DISCOVERY_FEATURE_ANALYTICS
from cloudtik.core._private.utils import \
    get_node_type, get_resource_of_node_type, RUNTIME_CONFIG_KEY, get_node_type_config, get_config_for_update, \
    get_runtime_config
from cloudtik.runtime.common.service_discovery.cluster import has_runtime_in_cluster
from cloudtik.runtime.common.service_discovery.discovery import DiscoveryType
from cloudtik.runtime.common.service_discovery.runtime_discovery import discover_metastore

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["io.trino.server.TrinoServer", False, "TrinoServer", "node"],
]

TRINO_HIVE_METASTORE_URI_KEY = "hive_metastore_uri"
TRINO_METASTORE_SERVICE_SELECTOR_KEY = "metastore_service_selector"

JVM_MAX_MEMORY_RATIO = 0.8
QUERY_MAX_MEMORY_PER_NODE_RATIO = 0.5
MEMORY_HEAP_HEADROOM_PER_NODE_RATIO = 0.25

TRINO_SERVICE_NAME = BUILT_IN_RUNTIME_TRINO
TRINO_SERVICE_PORT = 8081


def _get_config(runtime_config: Dict[str, Any]):
    return runtime_config.get(BUILT_IN_RUNTIME_TRINO, {})


def _is_metastore_service_discovery(trino_config):
    return trino_config.get("metastore_service_discovery", True)


def get_jvm_max_memory(total_memory):
    return int(total_memory * JVM_MAX_MEMORY_RATIO)


def get_query_max_memory_per_node(jvm_max_memory):
    return int(jvm_max_memory * QUERY_MAX_MEMORY_PER_NODE_RATIO)


def get_memory_heap_headroom_per_node(jvm_max_memory):
    return int(jvm_max_memory * MEMORY_HEAP_HEADROOM_PER_NODE_RATIO)


def _config_depended_services(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    runtime_config = get_config_for_update(cluster_config, RUNTIME_CONFIG_KEY)
    trino_config = get_config_for_update(runtime_config, BUILT_IN_RUNTIME_TRINO)

    # Check metastore
    if (not trino_config.get(TRINO_HIVE_METASTORE_URI_KEY) and
            not has_runtime_in_cluster(
                runtime_config, BUILT_IN_RUNTIME_METASTORE)):
        if _is_metastore_service_discovery(trino_config):
            hive_metastore_uri = discover_metastore(
                trino_config, TRINO_METASTORE_SERVICE_SELECTOR_KEY,
                cluster_config=cluster_config,
                discovery_type=DiscoveryType.WORKSPACE)
            if hive_metastore_uri:
                trino_config[TRINO_HIVE_METASTORE_URI_KEY] = hive_metastore_uri

    return cluster_config


def _prepare_config_on_head(cluster_config: Dict[str, Any]):
    cluster_config = _discover_metastore_on_head(cluster_config)
    return cluster_config


def _discover_metastore_on_head(cluster_config: Dict[str, Any]):
    runtime_config = get_runtime_config(cluster_config)
    trino_config = _get_config(runtime_config)
    if not _is_metastore_service_discovery(trino_config):
        return cluster_config

    hive_metastore_uri = trino_config.get(TRINO_HIVE_METASTORE_URI_KEY)
    if hive_metastore_uri:
        # Metastore already configured
        return cluster_config

    if has_runtime_in_cluster(
            runtime_config, BUILT_IN_RUNTIME_METASTORE):
        # There is a metastore
        return cluster_config

    # There is service discovery to come here
    hive_metastore_uri = discover_metastore(
        trino_config, TRINO_METASTORE_SERVICE_SELECTOR_KEY,
        cluster_config=cluster_config,
        discovery_type=DiscoveryType.CLUSTER)
    if hive_metastore_uri:
        trino_config = get_config_for_update(
            runtime_config, BUILT_IN_RUNTIME_TRINO)
        trino_config[TRINO_HIVE_METASTORE_URI_KEY] = hive_metastore_uri
    return cluster_config


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _is_runtime_scripts(script_file):
    if script_file.endswith(".trino.sql"):
        return True

    return False


def _get_runnable_command(target):
    command_parts = ["trino", "-f", double_quote(target)]
    return command_parts


def _with_runtime_environment_variables(runtime_config, config, provider, node_id: str):
    runtime_envs = {"TRINO_ENABLED": True}
    trino_config = _get_config(runtime_config)
    cluster_runtime_config = config.get(RUNTIME_CONFIG_KEY)

    if has_runtime_in_cluster(
            cluster_runtime_config, BUILT_IN_RUNTIME_METASTORE):
        runtime_envs["METASTORE_ENABLED"] = True

    _with_memory_configurations(
        runtime_envs, trino_config=trino_config,
        config=config, provider=provider, node_id=node_id)

    # We need export the cloud storage
    node_type_config = get_node_type_config(config, provider, node_id)
    provider_envs = provider.with_environment_variables(node_type_config, node_id)
    runtime_envs.update(provider_envs)

    return runtime_envs


def _configure(runtime_config, head: bool):
    # TODO: move more runtime specific environment_variables to here
    # only needed for applying head service discovery settings
    trino_config = _get_config(runtime_config)
    hive_metastore_uri = trino_config.get(TRINO_HIVE_METASTORE_URI_KEY)
    if hive_metastore_uri:
        os.environ["HIVE_METASTORE_URI"] = hive_metastore_uri


def _get_runtime_logs():
    logs_dir = os.path.join(os.getenv("TRINO_HOME"), "logs")
    all_logs = {"trino": logs_dir}
    return all_logs


def _get_runtime_endpoints(cluster_head_ip):
    endpoints = {
        "trino": {
            "name": "Trino Web UI",
            "url": "http://{}:{}".format(cluster_head_ip, TRINO_SERVICE_PORT)
        },
    }
    return endpoints


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

    trino_config = runtime_config.get(BUILT_IN_RUNTIME_TRINO)
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


def _get_head_service_ports(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    service_ports = {
        "trino": {
            "protocol": "TCP",
            "port": TRINO_SERVICE_PORT,
        },
    }
    return service_ports


def _get_runtime_services(
        runtime_config: Dict[str, Any], cluster_name: str) -> Dict[str, Any]:
    trino_config = _get_config(runtime_config)
    service_discovery_config = get_service_discovery_config(trino_config)
    service_name = get_canonical_service_name(
        service_discovery_config, cluster_name, TRINO_SERVICE_NAME)
    services = {
        service_name: define_runtime_service_on_head(
            service_discovery_config, TRINO_SERVICE_PORT,
            features=[SERVICE_DISCOVERY_FEATURE_ANALYTICS]),
    }
    return services
