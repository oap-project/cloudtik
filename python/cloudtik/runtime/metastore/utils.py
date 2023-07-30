import os
from typing import Any, Dict

from cloudtik.core._private.providers import _get_node_provider, _get_workspace_provider
from cloudtik.core._private.service_discovery.utils import SERVICE_DISCOVERY_PROTOCOL, SERVICE_DISCOVERY_PORT, \
    SERVICE_DISCOVERY_NODE_KIND, SERVICE_DISCOVERY_NODE_KIND_HEAD, SERVICE_DISCOVERY_PROTOCOL_TCP, \
    get_canonical_service_name
from cloudtik.core._private.utils import export_runtime_flags

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["proc_metastore", False, "Metastore", "head"],
    ["mysql", False, "MySQL", "head"],
]

METASTORE_RUNTIME_CONFIG_KEY = "metastore"

METASTORE_SERVICE_NAME = "metastore"
METASTORE_SERVICE_PORT = 9083


def _get_config(runtime_config: Dict[str, Any]):
    return runtime_config.get(METASTORE_RUNTIME_CONFIG_KEY, {})


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _with_runtime_environment_variables(runtime_config, config, provider, node_id: str):
    runtime_envs = {"METASTORE_ENABLED": True}

    metastore_config = _get_config(runtime_config)
    export_runtime_flags(
        metastore_config, METASTORE_RUNTIME_CONFIG_KEY, runtime_envs)
    return runtime_envs


def publish_service_endpoint(cluster_config: Dict[str, Any], head_node_id: str) -> None:
    workspace_name = cluster_config.get("workspace_name")
    if workspace_name is None:
        return

    provider = _get_node_provider(cluster_config["provider"], cluster_config["cluster_name"])
    head_internal_ip = provider.internal_ip(head_node_id)
    service_endpoints = {"hive-metastore-uri": "thrift://{}:{}".format(
        head_internal_ip, METASTORE_SERVICE_PORT)}

    workspace_provider = _get_workspace_provider(cluster_config["provider"], workspace_name)
    workspace_provider.publish_global_variables(cluster_config, service_endpoints)


def _get_runtime_logs():
    hive_logs_dir = os.path.join(os.getenv("METASTORE_HOME"), "logs")
    all_logs = {"metastore": hive_logs_dir}
    return all_logs


def _get_runtime_endpoints(cluster_head_ip):
    endpoints = {
        "metastore": {
            "name": "Metastore Uri",
            "url": "thrift://{}:{}".format(cluster_head_ip, METASTORE_SERVICE_PORT)
        },
    }
    return endpoints


def _get_head_service_ports(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    service_ports = {
        "metastore": {
            "protocol": "TCP",
            "port": METASTORE_SERVICE_PORT,
        },
    }
    return service_ports


def _get_runtime_services(
        runtime_config: Dict[str, Any], cluster_name: str) -> Dict[str, Any]:
    metastore_config = _get_config(runtime_config)
    service_name = get_canonical_service_name(
        metastore_config, cluster_name, METASTORE_SERVICE_NAME)
    services = {
        service_name: {
            SERVICE_DISCOVERY_PROTOCOL: SERVICE_DISCOVERY_PROTOCOL_TCP,
            SERVICE_DISCOVERY_PORT: METASTORE_SERVICE_PORT,
            SERVICE_DISCOVERY_NODE_KIND: SERVICE_DISCOVERY_NODE_KIND_HEAD
        },
    }
    return services
