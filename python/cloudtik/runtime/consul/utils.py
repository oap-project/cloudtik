import os
from typing import Any, Dict

from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_CONSUL
from cloudtik.core._private.utils import \
    publish_cluster_variable, RUNTIME_TYPES_CONFIG_KEY, _get_node_type_specific_runtime_config, RUNTIME_CONFIG_KEY, \
    get_config_for_update
from cloudtik.core._private.workspace.workspace_operator import _get_workspace_provider

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["consul", True, "Consul", "node"],
]

CONSUL_RUNTIME_CONFIG_KEY = "consul"
CONSUL_JOIN_LIST = "consul_join_list"
CONSUL_RPC_PORT = "consul_rpc_port"

CONSUL_SERVER_RPC_PORT = 8300
CONSUL_SERVER_HTTP_PORT = 8500


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _is_agent_server_mode(runtime_config):
    # Whether this is a consul server cluster or deploy at client
    consul_config = runtime_config.get(CONSUL_RUNTIME_CONFIG_KEY, {})
    return consul_config.get("server", False)


def _to_joint_list(sever_uri):
    hosts = []
    port = CONSUL_SERVER_RPC_PORT
    host_port_list = [x.strip() for x in sever_uri.split(",")]
    for host_port in host_port_list:
        parts = [x.strip() for x in host_port.split(":")]
        n = len(parts)
        if n == 1:
            host = parts[0]
        elif n == 2:
            host = parts[0]
            port = int(parts[1])
        else:
            raise ValueError(
                "Invalid Consul server uri: {}".format(sever_uri))
        hosts.append(host)

    join_list = ",".join(['"{}"'.format(host) for host in hosts])
    return join_list, port


def _bootstrap_join_list(cluster_config: Dict[str, Any]):
    workspace_name = cluster_config.get("workspace_name")
    if not workspace_name:
        raise ValueError("Workspace name should be configured for cluster.")

    # The consul server cluster must be running and registered
    # discovered with bootstrap methods (workspace global variables)
    workspace_provider = _get_workspace_provider(cluster_config["provider"], workspace_name)
    global_variables = workspace_provider.subscribe_global_variables(cluster_config)
    consul_uri = global_variables.get("consul-uri")
    if not consul_uri:
        raise RuntimeError("No running consul server cluster is detected.")

    runtime_config = get_config_for_update(cluster_config, RUNTIME_CONFIG_KEY)
    consul_config = get_config_for_update(runtime_config, CONSUL_RUNTIME_CONFIG_KEY)

    join_list, rpc_port = _to_joint_list(consul_uri)
    consul_config[CONSUL_JOIN_LIST] = join_list
    # current we don't use it
    consul_config[CONSUL_RPC_PORT] = rpc_port


def _with_runtime_environment_variables(
        server_mode, runtime_config, config,
        provider, node_id: str):
    runtime_envs = {}

    if server_mode:
        runtime_envs["CONSUL_SERVER"] = True

        # get the number of the workers plus head
        minimal_workers = _get_consul_minimal_workers(config)
        runtime_envs["CONSUL_NUM_SERVERS"] = minimal_workers + 1
    else:
        runtime_envs["CONSUL_CLIENT"] = True

        consul_config = runtime_config.get(CONSUL_RUNTIME_CONFIG_KEY, {})
        join_list = consul_config.get(CONSUL_JOIN_LIST)
        if not join_list:
            raise RuntimeError("Invalid join list. No running consul server cluster is detected.")
        runtime_envs["CONSUL_JOIN_LIST"] = join_list

    return runtime_envs


def _get_runtime_logs():
    consul_logs_dir = os.path.join(os.getenv("HOME"), "runtime", "consul", "logs")
    all_logs = {"consul": consul_logs_dir}
    return all_logs


def _get_runtime_services(server_mode, cluster_head_ip):
    services = {
        "consul": {
            "name": "Consul RPC",
            "url": "{}:{}".format(cluster_head_ip, CONSUL_SERVER_RPC_PORT)
        },
    }
    if server_mode:
        services["consul_ui"] = {
            "name": "Consul UI",
            "url": "http://{}:{}".format(cluster_head_ip, CONSUL_SERVER_HTTP_PORT)
        }
    return services


def _get_runtime_service_ports(server_mode, runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    service_ports = {
        "consul": {
            "protocol": "TCP",
            "port": CONSUL_SERVER_RPC_PORT,
        },
    }

    if server_mode:
        service_ports["consul_ui"] = {
            "protocol": "TCP",
            "port": CONSUL_SERVER_HTTP_PORT,
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
