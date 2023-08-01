import copy
import json
import os
from typing import Any, Dict

from cloudtik.core._private.constants import CLOUDTIK_RUNTIME_ENV_NODE_IP, CLOUDTIK_RUNTIME_ENV_QUORUM_JOIN, \
    CLOUDTIK_RUNTIME_ENV_HEAD_IP
from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_CONSUL, _get_runtime
from cloudtik.core._private.runtime_utils import get_runtime_node_type, get_runtime_node_ip, \
    get_runtime_config_from_node, RUNTIME_NODE_SEQ_ID, RUNTIME_NODE_IP, subscribe_nodes_info, sort_nodes_by_seq_id, \
    load_and_save_json
from cloudtik.core._private.service_discovery.utils import SERVICE_DISCOVERY_PORT, \
    SERVICE_DISCOVERY_TAGS, SERVICE_DISCOVERY_META, SERVICE_DISCOVERY_META_RUNTIME, \
    SERVICE_DISCOVERY_CHECK_INTERVAL, SERVICE_DISCOVERY_CHECK_TIMEOUT, SERVICE_DISCOVERY_META_CLUSTER, \
    match_service_node
from cloudtik.core._private.utils import \
    publish_cluster_variable, RUNTIME_TYPES_CONFIG_KEY, _get_node_type_specific_runtime_config, \
    RUNTIME_CONFIG_KEY, get_config_for_update
from cloudtik.core._private.workspace.workspace_operator import _get_workspace_provider
from cloudtik.core.tags import QUORUM_JOIN_STATUS_INIT

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["consul", True, "Consul", "node"],
]

CONSUL_RUNTIME_CONFIG_KEY = "consul"
CONFIG_KEY_JOIN_LIST = "join_list"
CONFIG_KEY_RPC_PORT = "rpc_port"
CONFIG_KEY_SERVICES = "services"

CONSUL_SERVER_RPC_PORT = 8300
CONSUL_SERVER_HTTP_PORT = 8500

SERVICE_CHECK_INTERVAL_DEFAULT = 10
SERVICE_CHECK_TIMEOUT_DEFAULT = 5


def _get_config(runtime_config: Dict[str, Any]):
    return runtime_config.get(CONSUL_RUNTIME_CONFIG_KEY, {})


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _is_agent_server_mode(runtime_config):
    # Whether this is a consul server cluster or deploy at client
    consul_config = _get_config(runtime_config)
    return consul_config.get("server", False)


def _get_consul_config_for_update(cluster_config):
    runtime_config = get_config_for_update(cluster_config, RUNTIME_CONFIG_KEY)
    return get_config_for_update(runtime_config, CONSUL_RUNTIME_CONFIG_KEY)


def _to_joint_list(endpoint_uri):
    hosts = []
    port = CONSUL_SERVER_RPC_PORT
    host_port_list = [x.strip() for x in endpoint_uri.split(",")]
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
                "Invalid Consul endpoint uri: {}".format(endpoint_uri))
        hosts.append(host)

    join_list = ",".join(hosts)
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

    consul_config = _get_consul_config_for_update(cluster_config)

    join_list, rpc_port = _to_joint_list(consul_uri)
    consul_config[CONFIG_KEY_JOIN_LIST] = join_list
    # current we don't use it
    consul_config[CONFIG_KEY_RPC_PORT] = rpc_port

    return cluster_config


def _bootstrap_runtime_services(config: Dict[str, Any]):
    # for all the runtimes, query its services per node type
    services_map = {}
    cluster_name = config["cluster_name"]
    available_node_types = config["available_node_types"]
    head_node_type = config["head_node_type"]
    for node_type in available_node_types:
        runtime_config = _get_node_type_specific_runtime_config(config, node_type)
        if not runtime_config:
            continue

        head = True if node_type == head_node_type else False

        services_for_node_type = {}
        runtime_types = runtime_config.get(RUNTIME_TYPES_CONFIG_KEY, [])
        for runtime_type in runtime_types:
            if runtime_type == BUILT_IN_RUNTIME_CONSUL:
                continue

            runtime = _get_runtime(runtime_type, runtime_config)
            services = runtime.get_runtime_services(cluster_name)
            if not services:
                continue

            for service_name, runtime_service in services.items():
                if match_service_node(runtime_service, head):
                    # conversion between the data formats
                    service_config = _generate_service_config(
                        cluster_name, runtime_type, runtime_service)
                    services_for_node_type[service_name] = service_config
        if services_for_node_type:
            services_map[node_type] = services_for_node_type

    if services_map:
        consul_config = _get_consul_config_for_update(config)
        consul_config[CONFIG_KEY_SERVICES] = services_map

    return config


def _generate_service_config(cluster_name, runtime_type, runtime_service):
    # We utilize all the standard service discovery properties
    service_config = copy.deepcopy(runtime_service)
    meta_config = get_config_for_update(
        service_config, SERVICE_DISCOVERY_META)

    meta_config[SERVICE_DISCOVERY_META_CLUSTER] = cluster_name
    meta_config[SERVICE_DISCOVERY_META_RUNTIME] = runtime_type
    return service_config


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

        consul_config = _get_config(runtime_config)
        join_list = consul_config.get(CONFIG_KEY_JOIN_LIST)
        if not join_list:
            raise RuntimeError("Invalid join list. No running consul server cluster is detected.")
        runtime_envs["CONSUL_JOIN_LIST"] = join_list

    return runtime_envs


def _get_home_dir():
    return os.path.join(os.getenv("HOME"), "runtime", "consul")


def _get_runtime_logs():
    home_dir = _get_home_dir()
    logs_dir = os.path.join(home_dir, "logs")
    return {"consul": logs_dir}


def _get_runtime_endpoints(server_mode, cluster_head_ip):
    endpoints = {
        "consul": {
            "name": "Consul RPC",
            "url": "{}:{}".format(cluster_head_ip, CONSUL_SERVER_RPC_PORT)
        },
    }
    if server_mode:
        endpoints["consul_ui"] = {
            "name": "Consul UI",
            "url": "http://{}:{}".format(cluster_head_ip, CONSUL_SERVER_HTTP_PORT)
        }
    return endpoints


def _get_head_service_ports(server_mode, runtime_config: Dict[str, Any]) -> Dict[str, Any]:
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
        if RUNTIME_NODE_IP not in node_info:
            raise RuntimeError("Missing node ip for node {}.".format(node_id))
        if RUNTIME_NODE_SEQ_ID not in node_info:
            raise RuntimeError("Missing node sequence id for node {}.".format(node_id))
        server_ensemble += [node_info]

    def node_info_sort(node_info):
        return node_info[RUNTIME_NODE_SEQ_ID]

    server_ensemble.sort(key=node_info_sort)
    return server_ensemble


def _handle_node_constraints_reached(
        runtime_config: Dict[str, Any], cluster_config: Dict[str, Any],
        node_type: str, head_info: Dict[str, Any], nodes_info: Dict[str, Any]):
    # We know this is called in the cluster scaler context
    server_ensemble = _server_ensemble_from_nodes_info(nodes_info)
    endpoint_uri = "{}:{}".format(
        head_info[RUNTIME_NODE_IP], CONSUL_SERVER_RPC_PORT)

    for node_info in server_ensemble:
        node_address = "{}:{}".format(
            node_info[RUNTIME_NODE_IP], CONSUL_SERVER_RPC_PORT)
        if len(endpoint_uri) > 0:
            endpoint_uri += ","
        endpoint_uri += node_address

    _publish_service_endpoint_to_cluster(endpoint_uri)
    _publish_service_endpoint_to_workspace(cluster_config, endpoint_uri)


def _publish_service_endpoint_to_cluster(endpoint_uri: str) -> None:
    publish_cluster_variable("consul-uri", endpoint_uri)


def _publish_service_endpoint_to_workspace(
        cluster_config: Dict[str, Any], endpoint_uri: str) -> None:
    workspace_name = cluster_config["workspace_name"]
    if workspace_name is None:
        return

    service_endpoints = {"consul-uri": endpoint_uri}
    workspace_provider = _get_workspace_provider(cluster_config["provider"], workspace_name)
    workspace_provider.publish_global_variables(cluster_config, service_endpoints)


def _get_consul_minimal_workers(config: Dict[str, Any]):
    available_node_types = config["available_node_types"]
    head_node_type = config["head_node_type"]
    for node_type in available_node_types:
        if node_type == head_node_type:
            # Exclude the head
            continue
        # Check the runtimes of the node type whether it needs to wait minimal before update
        runtime_config = _get_node_type_specific_runtime_config(config, node_type)
        if not runtime_config:
            continue
        runtime_types = runtime_config.get(RUNTIME_TYPES_CONFIG_KEY, [])
        if BUILT_IN_RUNTIME_CONSUL not in runtime_types:
            continue
        node_type_config = available_node_types[node_type]
        min_workers = node_type_config.get("min_workers", 0)
        return min_workers
    return 0


def _get_services_of_node_type(runtime_config, node_type):
    consul_config = _get_config(runtime_config)
    services_map = consul_config.get(CONFIG_KEY_SERVICES)
    if not services_map:
        return None
    return services_map.get(node_type)


###################################
# Calls from node when configuring
###################################


def configure_join(head):
    consul_server = os.environ.get("CONSUL_SERVER")
    server_mode = True if consul_server == "true" else False
    _configure_join_list(server_mode, head)

    if server_mode:
        quorum_join = os.environ.get(CLOUDTIK_RUNTIME_ENV_QUORUM_JOIN)
        if quorum_join == QUORUM_JOIN_STATUS_INIT:
            _update_server_config_for_join()


def _configure_join_list(server_mode, head):
    # Configure the retry join list for all the cases

    if server_mode:
        # join list for servers
        if head:
            # for head, use its own address
            node_ip = os.environ.get(CLOUDTIK_RUNTIME_ENV_NODE_IP)
            if not node_ip:
                raise RuntimeError("Missing node ip environment variable for the running node.")
            join_list = [node_ip]
        else:
            # getting from the quorum nodes info
            join_list = _get_join_list_from_nodes_info()
    else:
        # client mode, get from the CONSUL_JOIN_LIST environments
        join_list_str = os.environ.get("CONSUL_JOIN_LIST")
        if not join_list_str:
            raise RuntimeError("Missing join list environment variable for the running node.")
        join_list = join_list_str.split(',')

    _update_join_list_config(join_list)


def _get_join_list_from_nodes_info():
    nodes_info = subscribe_nodes_info()
    join_nodes = sort_nodes_by_seq_id(nodes_info)
    head_node_ip = os.environ.get(CLOUDTIK_RUNTIME_ENV_HEAD_IP)
    if not head_node_ip:
        raise RuntimeError("Missing head node ip environment variable for the running node.")

    join_list = [head_node_ip]
    join_list += [node[RUNTIME_NODE_IP] for node in join_nodes]
    return join_list


def _update_join_list_config(join_list):
    home_dir = _get_home_dir()
    config_file = os.path.join(home_dir, "consul.d", "consul.json")

    def update_retry_join(config_object):
        config_object["retry_join"] = join_list

    load_and_save_json(config_file, update_retry_join)


def _update_server_config_for_join():
    home_dir = _get_home_dir()
    config_file = os.path.join(home_dir, "consul.d", "server.json")

    def update_server_config(config_object):
        config_object.pop("bootstrap_expect", None)

    load_and_save_json(config_file, update_server_config)


def configure_services(head):
    """This method is called from configure.py script which is running on node.
    """
    node_type = get_runtime_node_type()
    runtime_config = get_runtime_config_from_node(head)
    services_config = _get_services_of_node_type(runtime_config, node_type)

    home_dir = _get_home_dir()
    config_dir = os.path.join(home_dir, "consul.d")
    services_file = os.path.join(config_dir, "services.json")
    if not services_config:
        # no services, remove the services file
        if os.path.isfile(services_file):
            os.remove(services_file)
    else:
        # generate the services configuration file
        os.makedirs(config_dir, exist_ok=True)
        services = _generate_services_def(services_config)
        with open(services_file, "w") as f:
            f.write(json.dumps(services, indent=4))


def _generate_service_def(service_name, service_config):
    node_ip = get_runtime_node_ip()
    port = service_config[SERVICE_DISCOVERY_PORT]
    check_interval = service_config.get(
        SERVICE_DISCOVERY_CHECK_INTERVAL, SERVICE_CHECK_INTERVAL_DEFAULT)
    check_timeout = service_config.get(
        SERVICE_DISCOVERY_CHECK_TIMEOUT, SERVICE_CHECK_TIMEOUT_DEFAULT)
    service_def = {
            "name": service_name,
            "address": node_ip,
            "port": port,
            "checks": [
                {
                    "tcp": "{}:{}".format(node_ip, port),
                    "interval": "{}s".format(check_interval),
                    "timeout": "{}s".format(check_timeout),
                }
            ]
        }

    tags = service_config.get(SERVICE_DISCOVERY_TAGS)
    if tags:
        service_def["tags"] = tags

    meta = service_config.get(SERVICE_DISCOVERY_META)
    if meta:
        service_def["meta"] = meta

    return service_def


def _generate_services_def(services_config):
    services = []
    for service_name, service_config in services_config.items():
        service_def = _generate_service_def(service_name, service_config)
        services.append(service_def)

    services_config = {
        "services": services
    }
    return services_config
