import os
from typing import Any, Dict

import yaml

from cloudtik.core._private.providers import _get_workspace_provider
from cloudtik.core._private.service_discovery.utils import SERVICE_DISCOVERY_PROTOCOL, SERVICE_DISCOVERY_PORT, \
    SERVICE_DISCOVERY_PROTOCOL_TCP, get_canonical_service_name, SERVICE_DISCOVERY_NODE_KIND, \
    SERVICE_DISCOVERY_NODE_KIND_WORKER

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["etcd", True, "etcd", "worker"],
]

ETCD_RUNTIME_CONFIG_KEY = "etcd"

ETCD_SERVICE_NAME = "etcd"
ETCD_SERVICE_PORT = 2379
ETCD_PEER_PORT = 2380


def _get_config(runtime_config: Dict[str, Any]):
    return runtime_config.get(ETCD_RUNTIME_CONFIG_KEY, {})


def _get_home_dir():
    return os.path.join(os.getenv("HOME"), "runtime", ETCD_SERVICE_NAME)


def _get_runtime_logs():
    home_dir = _get_home_dir()
    logs_dir = os.path.join(home_dir, "logs")
    return {"etcd": logs_dir}


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _with_runtime_environment_variables(
        runtime_config, config,
        provider, node_id: str):
    runtime_envs = {
        "ETCD_CLUSTER_NAME": config["cluster_name"]
    }
    return runtime_envs


def _get_endpoints(nodes):
    return ",".join(
        ["http://{}:{}".format(
            node["node_number"], node["node_ip"], ETCD_PEER_PORT) for node in nodes])


def _handle_minimal_nodes_reached(
        runtime_config: Dict[str, Any], cluster_config: Dict[str, Any],
        node_type: str, head_info: Dict[str, Any], nodes_info: Dict[str, Any]):
    # We know this is called in the cluster scaler context
    initial_cluster = _initial_cluster_from_nodes_info(nodes_info)
    endpoints = _get_endpoints(initial_cluster)

    # public the endpoint to workspace
    _publish_endpoints_to_workspace(cluster_config, endpoints)


def _publish_endpoints_to_workspace(cluster_config: Dict[str, Any], endpoints: str) -> None:
    workspace_name = cluster_config["workspace_name"]
    if workspace_name is None:
        return

    service_endpoints = {"etcd-endpoints": endpoints}
    workspace_provider = _get_workspace_provider(cluster_config["provider"], workspace_name)
    workspace_provider.publish_global_variables(cluster_config, service_endpoints)


def _get_runtime_endpoints(runtime_config: Dict[str, Any], cluster_head_ip):
    # TODO: future to retrieve the endpoints from service discovery
    return None


def _get_runtime_services(
        runtime_config: Dict[str, Any], cluster_name: str) -> Dict[str, Any]:
    etcd_config = _get_config(runtime_config)
    service_name = get_canonical_service_name(
        etcd_config, cluster_name, ETCD_SERVICE_NAME)
    services = {
        service_name: {
            SERVICE_DISCOVERY_PROTOCOL: SERVICE_DISCOVERY_PROTOCOL_TCP,
            SERVICE_DISCOVERY_PORT: ETCD_SERVICE_PORT,
            SERVICE_DISCOVERY_NODE_KIND: SERVICE_DISCOVERY_NODE_KIND_WORKER
        },
    }
    return services


def _initial_cluster_from_nodes_info(nodes_info: Dict[str, Any]):
    initial_cluster = []
    for node_id, node_info in nodes_info.items():
        if "node_ip" not in node_info:
            raise RuntimeError("Missing node ip for node {}.".format(node_id))
        if "node_number" not in node_info:
            raise RuntimeError("Missing node number for node {}.".format(node_id))
        initial_cluster += [node_info]

    def node_info_sort(node_info):
        return node_info["node_number"]

    initial_cluster.sort(key=node_info_sort)
    return initial_cluster


###################################
# Calls from node when configuring
###################################

def configure_initial_cluster(nodes_info: Dict[str, Any]):

    if nodes_info is None:
        raise RuntimeError("Missing nodes info for configuring server ensemble.")

    initial_cluster = _initial_cluster_from_nodes_info(nodes_info)
    initial_cluster_str = ",".join(
        ["server{}=http://{}:{}".format(
            node["node_number"], node["node_ip"], ETCD_PEER_PORT) for node in initial_cluster])

    home_dir = _get_home_dir()
    config_file = os.path.join(home_dir, "conf", "etcd.yaml")
    # load and save yaml
    with open(config_file) as f:
        config_object = yaml.safe_load(f)

    config_object["initial-cluster"] = initial_cluster_str

    with open(config_file, "w") as f:
        yaml.dump(config_object, f, default_flow_style=False)
