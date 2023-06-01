import os
from typing import Any, Dict

from cloudtik.core._private.providers import _get_node_provider, _get_workspace_provider
from cloudtik.core._private.utils import export_runtime_flags

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["proc_metastore", False, "Metastore", "head"],
    ["mysql", False, "MySQL", "head"],
]

RUNTIME_CONFIG_KEY = "metastore"


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _with_runtime_environment_variables(runtime_config, config, provider, node_id: str):
    runtime_envs = {"METASTORE_ENABLED": True}

    metastore_config = runtime_config.get(RUNTIME_CONFIG_KEY, {})
    export_runtime_flags(
        metastore_config, RUNTIME_CONFIG_KEY, runtime_envs)
    return runtime_envs


def publish_service_uri(cluster_config: Dict[str, Any], head_node_id: str) -> None:
    workspace_name = cluster_config.get("workspace_name")
    if workspace_name is None:
        return

    provider = _get_node_provider(cluster_config["provider"], cluster_config["cluster_name"])
    head_internal_ip = provider.internal_ip(head_node_id)
    service_uris = {"hive-metastore-uri": "thrift://{}:9083".format(head_internal_ip)}

    workspace_provider = _get_workspace_provider(cluster_config["provider"], workspace_name)
    workspace_provider.publish_global_variables(cluster_config, service_uris)


def _get_runtime_logs():
    hive_logs_dir = os.path.join(os.getenv("METASTORE_HOME"), "logs")
    all_logs = {"metastore": hive_logs_dir}
    return all_logs


def _get_runtime_services(cluster_head_ip):
    services = {
        "metastore": {
            "name": "Metastore Uri",
            "url": "thrift://{}:9083".format(cluster_head_ip)
        },
    }
    return services


def _get_runtime_service_ports(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    service_ports = {
        "metastore": {
            "protocol": "TCP",
            "port": 9083,
        },
    }
    return service_ports
