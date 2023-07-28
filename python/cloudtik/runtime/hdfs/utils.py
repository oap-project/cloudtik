import os
from typing import Any, Dict

from cloudtik.core._private.service_discovery.utils import SERVICE_DISCOVERY_PROTOCOL, SERVICE_DISCOVERY_PORT, \
    SERVICE_DISCOVERY_NODE_KIND, SERVICE_DISCOVERY_NODE_KIND_HEAD, SERVICE_DISCOVERY_PROTOCOL_TCP, \
    get_canonical_service_name
from cloudtik.core._private.utils import get_node_type_config
from cloudtik.core._private.workspace.workspace_operator import _get_workspace_provider
from cloudtik.core._private.providers import _get_node_provider

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["proc_namenode", False, "NameNode", "head"],
    ["proc_datanode", False, "DataNode", "worker"],
]

HDFS_RUNTIME_CONFIG_KEY = "hdfs"

HDFS_WEB_PORT = 9870

HDFS_SERVICE_NAME = "hdfs"
HDFS_SERVICE_PORT = 9000


def _get_config(runtime_config: Dict[str, Any]):
    return runtime_config.get(HDFS_RUNTIME_CONFIG_KEY, {})


def publish_service_uri(cluster_config: Dict[str, Any], head_node_id: str) -> None:
    workspace_name = cluster_config.get("workspace_name")
    if workspace_name is None:
        return

    provider = _get_node_provider(cluster_config["provider"], cluster_config["cluster_name"])
    head_internal_ip = provider.internal_ip(head_node_id)
    service_uris = {"hdfs-namenode-uri": "hdfs://{}:{}".format(
        head_internal_ip, HDFS_SERVICE_PORT)}

    workspace_provider = _get_workspace_provider(cluster_config["provider"], workspace_name)
    workspace_provider.publish_global_variables(cluster_config, service_uris)


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _with_runtime_environment_variables(runtime_config, config, provider, node_id: str):
    runtime_envs = {"HDFS_ENABLED": True}

    # We always export the cloud storage even for local HDFS case
    node_type_config = get_node_type_config(config, provider, node_id)
    provider_envs = provider.with_environment_variables(node_type_config, node_id)
    runtime_envs.update(provider_envs)

    return runtime_envs


def _get_runtime_logs():
    hadoop_logs_dir = os.path.join(os.getenv("HADOOP_HOME"), "logs")
    all_logs = {"hadoop": hadoop_logs_dir}
    return all_logs


def _get_runtime_endpoints(cluster_head_ip):
    endpoints = {
        "hdfs-web": {
            "name": "HDFS Web UI",
            "url": "http://{}:{}".format(cluster_head_ip, HDFS_WEB_PORT)
        },
        "hdfs": {
            "name": "HDFS Service",
            "url": "hdfs://{}:{}".format(cluster_head_ip, HDFS_SERVICE_PORT)
        },
    }
    return endpoints


def _get_head_service_ports(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    service_ports = {
        "hdfs-web": {
            "protocol": "TCP",
            "port": HDFS_WEB_PORT,
        },
        "hdfs-nn": {
            "protocol": "TCP",
            "port": HDFS_SERVICE_PORT,
        },
    }
    return service_ports


def _get_runtime_services(
        runtime_config: Dict[str, Any], cluster_name: str) -> Dict[str, Any]:
    # service name is decided by the runtime itself
    # For in services backed by the collection of nodes of the cluster
    # service name is a combination of cluster_name + runtime_service_name
    hdfs_config = _get_config(runtime_config)
    service_name = get_canonical_service_name(
        hdfs_config, cluster_name, HDFS_SERVICE_NAME)
    services = {
        service_name: {
            SERVICE_DISCOVERY_PROTOCOL: SERVICE_DISCOVERY_PROTOCOL_TCP,
            SERVICE_DISCOVERY_PORT: HDFS_SERVICE_PORT,
            SERVICE_DISCOVERY_NODE_KIND: SERVICE_DISCOVERY_NODE_KIND_HEAD
        },
    }
    return services
