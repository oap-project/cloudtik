import os
from typing import Any, Dict

from cloudtik.core._private.constants import CLOUDTIK_RUNTIME_ENV_NODE_IP, CLOUDTIK_RUNTIME_ENV_NODE_SEQ_ID
from cloudtik.core._private.core_utils import exec_with_output, strip_quote
from cloudtik.core._private.providers import _get_workspace_provider
from cloudtik.core._private.runtime_utils import RUNTIME_NODE_SEQ_ID, RUNTIME_NODE_IP, sort_nodes_by_seq_id, \
    load_and_save_yaml
from cloudtik.core._private.service_discovery.utils import get_canonical_service_name, define_runtime_service_on_worker, \
    get_service_discovery_config

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
            node[RUNTIME_NODE_SEQ_ID], node[RUNTIME_NODE_IP], ETCD_PEER_PORT) for node in nodes])


def _handle_node_constraints_reached(
        runtime_config: Dict[str, Any], cluster_config: Dict[str, Any],
        node_type: str, head_info: Dict[str, Any], nodes_info: Dict[str, Any]):
    # We know this is called in the cluster scaler context
    initial_cluster = sort_nodes_by_seq_id(nodes_info)
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
    service_discovery_config = get_service_discovery_config(etcd_config)
    service_name = get_canonical_service_name(
        service_discovery_config, cluster_name, ETCD_SERVICE_NAME)
    services = {
        service_name: define_runtime_service_on_worker(
            service_discovery_config, ETCD_SERVICE_PORT),
    }
    return services


###################################
# Calls from node when configuring
###################################


def _get_initial_cluster_from_nodes_info(initial_cluster):
    return ",".join(
        ["server{}=http://{}:{}".format(
            node[RUNTIME_NODE_SEQ_ID], node[RUNTIME_NODE_IP], ETCD_PEER_PORT) for node in initial_cluster])


def configure_initial_cluster(nodes_info: Dict[str, Any]):
    if nodes_info is None:
        raise RuntimeError("Missing nodes info for configuring server ensemble.")

    initial_cluster = sort_nodes_by_seq_id(nodes_info)
    initial_cluster_str = _get_initial_cluster_from_nodes_info(initial_cluster)

    _update_initial_cluster_config(initial_cluster_str)


def _update_initial_cluster_config(initial_cluster_str):
    home_dir = _get_home_dir()
    config_file = os.path.join(home_dir, "conf", "etcd.yaml")

    def update_initial_cluster(config_object):
        config_object["initial-cluster"] = initial_cluster_str

    load_and_save_yaml(config_file, update_initial_cluster)


def request_to_join_cluster(nodes_info: Dict[str, Any]):
    if nodes_info is None:
        raise RuntimeError("Missing nodes info for join to the cluster.")

    initial_cluster = sort_nodes_by_seq_id(nodes_info)

    node_ip = os.environ.get(CLOUDTIK_RUNTIME_ENV_NODE_IP)
    if not node_ip:
        raise RuntimeError("Missing node ip environment variable for this node.")

    # exclude my own address from the initial cluster as endpoints
    endpoints = [node for node in initial_cluster if node[RUNTIME_NODE_IP] != node_ip]
    if not endpoints:
        raise RuntimeError("No exiting nodes found for contacting to join the cluster.")

    seq_id = os.environ.get(CLOUDTIK_RUNTIME_ENV_NODE_SEQ_ID)
    if not seq_id:
        raise RuntimeError("Missing sequence ip environment variable for this node.")

    _request_member_add(endpoints, node_ip, seq_id)


def _get_initial_cluster_from_output(output):
    output_lines = output.split('\n')
    initial_cluster_mark = "ETCD_INITIAL_CLUSTER="
    for output_line in output_lines:
        if output_line.startswith(initial_cluster_mark):
            return strip_quote(output_line[len(initial_cluster_mark):])


def _request_member_add(endpoints, node_ip, seq_id):
    # etcdctl --endpoints=http://existing_node_ip:2379 member add server --peer-urls=http://node_ip:2380
    cmd = ["etcdctl"]
    endpoints_str = ",".join(
        ["http://{}:{}".format(
            node[RUNTIME_NODE_IP], ETCD_SERVICE_PORT) for node in endpoints])
    cmd += ["--endpoints=" + endpoints_str]
    cmd += ["member", "add"]
    node_name = "server{}".format(seq_id)
    cmd += [node_name]
    peer_urls = "--peer-urls=http://{}:{}".format(node_ip, ETCD_PEER_PORT)
    cmd += [peer_urls]

    cmd_str = " ".join(cmd)
    output = exec_with_output(cmd_str).decode().strip()
    initial_cluster_str = _get_initial_cluster_from_output(output)
    if initial_cluster_str:
        # succeed
        _update_initial_cluster_config(initial_cluster_str)
