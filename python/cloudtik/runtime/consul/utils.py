import os
from typing import Any, Dict

from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_CONSUL
from cloudtik.core._private.utils import \
    publish_cluster_variable, RUNTIME_TYPES_CONFIG_KEY, _get_node_type_specific_runtime_config
from cloudtik.core._private.workspace.workspace_operator import _get_workspace_provider

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["consul", True, "Consul", "node"],
]

CONSUL_RUNTIME_CONFIG_KEY = "consul"
CONSUL_SERVER_RPC_PORT = 8300
CONSUL_SERVER_HTTP_PORT = 8500


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _with_runtime_environment_variables(runtime_config, config, provider, node_id: str):
    runtime_envs = {"CONSUL_ENABLED": True}

    # get the number of the workers plus head
    minimal_workers = _get_consul_minimal_workers(config)
    runtime_envs["CONSUL_SERVERS"] = minimal_workers + 1
    return runtime_envs


def _get_runtime_logs():
    consul_logs_dir = os.path.join(os.getenv("HOME"), "runtime", "consul", "logs")
    all_logs = {"consul": consul_logs_dir}
    return all_logs


def _get_runtime_services(cluster_head_ip):
    services = {
        "consul": {
            "name": "Consul RPC",
            "url": "{}:{}".format(cluster_head_ip, CONSUL_SERVER_RPC_PORT)
        },
        "consul_ui": {
            "name": "Consul UI",
            "url": "http://{}:{}".format(cluster_head_ip, CONSUL_SERVER_HTTP_PORT)
        },
    }
    return services


def _get_runtime_service_ports(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    service_ports = {
        "consul": {
            "protocol": "TCP",
            "port": CONSUL_SERVER_RPC_PORT,
        },
        "consul_ui": {
            "protocol": "TCP",
            "port": CONSUL_SERVER_HTTP_PORT,
        },
    }
    return service_ports


def _server_ensemble_from_nodes_info(nodes_info: Dict[str, Any]):
    server_ensemble = []
    for node_id, node_info in nodes_info.items():
        if "node_ip" not in node_info:
            raise RuntimeError("Missing node ip for node {}.".format(node_id))
        if "node_number" not in node_info:
            raise RuntimeError("Missing node number for node {}.".format(node_id))
        server_ensemble += [node_info]

    def node_info_sort(node_info):
        return node_info["node_number"]

    server_ensemble.sort(key=node_info_sort)
    return server_ensemble


def _handle_minimal_nodes_reached(
        runtime_config: Dict[str, Any], cluster_config: Dict[str, Any],
        node_type: str, head_info: Dict[str, Any], nodes_info: Dict[str, Any]):
    # We know this is called in the cluster scaler context
    server_ensemble = _server_ensemble_from_nodes_info(nodes_info)
    service_uri = "{}:{}".format(
        head_info["node_ip"], CONSUL_SERVER_RPC_PORT)

    for node_info in server_ensemble:
        node_address = "{}:{}".format(
            node_info["node_ip"], CONSUL_SERVER_RPC_PORT)
        if len(service_uri) > 0:
            service_uri += ","
        service_uri += node_address

    _publish_service_uri_to_cluster(service_uri)
    _publish_service_uri_to_workspace(cluster_config, service_uri)


def _publish_service_uri_to_cluster(service_uri: str) -> None:
    publish_cluster_variable("consul-uri", service_uri)


def _publish_service_uri_to_workspace(cluster_config: Dict[str, Any], service_uri: str) -> None:
    workspace_name = cluster_config["workspace_name"]
    if workspace_name is None:
        return

    service_uris = {"consul-uri": service_uri}
    workspace_provider = _get_workspace_provider(cluster_config["provider"], workspace_name)
    workspace_provider.publish_global_variables(cluster_config, service_uris)


def _get_consul_minimal_workers(config: Dict[str, Any]):
    available_node_types = config["available_node_types"]
    head_node_type = config["head_node_type"]
    for node_type in available_node_types:
        if node_type == head_node_type:
            # Exclude the head
            continue
        # Check the runtimes of the node type whether it needs to wait minimal before update
        runtime_config = _get_node_type_specific_runtime_config(config, node_type)
        if runtime_config:
            runtime_types = runtime_config.get(RUNTIME_TYPES_CONFIG_KEY, [])
            if BUILT_IN_RUNTIME_CONSUL in runtime_types:
                node_type_config = available_node_types[node_type]
                min_workers = node_type_config.get("min_workers", 0)
                return min_workers
    return 0
