import os
from typing import Any, Dict

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


def publish_service_uri(cluster_config: Dict[str, Any], head_node_id: str) -> None:
    workspace_name = cluster_config.get("workspace_name")
    if workspace_name is None:
        return

    provider = _get_node_provider(cluster_config["provider"], cluster_config["cluster_name"])
    head_internal_ip = provider.internal_ip(head_node_id)
    service_uris = {"hdfs-namenode-uri": "hdfs://{}:9000".format(head_internal_ip)}

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


def _get_runtime_services(cluster_head_ip):
    services = {
        "hdfs-web": {
            "name": "HDFS Web UI",
            "url": "http://{}:9870".format(cluster_head_ip)
        },
        "hdfs": {
            "name": "HDFS Service",
            "url": "hdfs://{}:9000".format(cluster_head_ip)
        },
    }
    return services


def _get_runtime_service_ports(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    service_ports = {
        "hdfs-web": {
            "protocol": "TCP",
            "port": 9870,
        },
        "hdfs-nn": {
            "protocol": "TCP",
            "port": 9000,
        },
    }
    return service_ports
