import os
import subprocess
from shlex import quote
from typing import Any, Dict, List

from cloudtik.core._private.constants import CLOUDTIK_RUNTIME_ENV_NODE_IP, CLOUDTIK_RUNTIME_ENV_NODE_SEQ_ID
from cloudtik.core._private.runtime_utils import subscribe_runtime_config, RUNTIME_NODE_SEQ_ID, RUNTIME_NODE_IP, \
    sort_nodes_by_seq_id
from cloudtik.core._private.service_discovery.utils import SERVICE_DISCOVERY_PROTOCOL, SERVICE_DISCOVERY_PORT, \
    SERVICE_DISCOVERY_NODE_KIND, SERVICE_DISCOVERY_PROTOCOL_TCP, SERVICE_DISCOVERY_NODE_KIND_WORKER, \
    get_canonical_service_name
from cloudtik.core._private.utils import \
    publish_cluster_variable, load_properties_file, save_properties_file
from cloudtik.core._private.workspace.workspace_operator import _get_workspace_provider

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["org.apache.zookeeper.server.quorum.QuorumPeerMain", False, "ZooKeeper", "worker"],
]

ZOOKEEPER_RUNTIME_CONFIG_KEY = "zookeeper"

ZOOKEEPER_SERVICE_NAME = "zookeeper"
ZOOKEEPER_SERVICE_PORT = 2181

ZOOKEEPER_QUORUM_RETRY = 30
ZOOKEEPER_QUORUM_RETRY_INTERVAL = 5


class NoQuorumError(RuntimeError):
    pass


def _get_config(runtime_config: Dict[str, Any]):
    return runtime_config.get(ZOOKEEPER_RUNTIME_CONFIG_KEY, {})


def _get_home_dir():
    return os.getenv("ZOOKEEPER_HOME")


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _with_runtime_environment_variables(runtime_config, config, provider, node_id: str):
    runtime_envs = {"ZOOKEEPER_ENABLED": True}
    return runtime_envs


def _get_runtime_logs():
    home_dir = _get_home_dir()
    zookeeper_logs_dir = os.path.join(home_dir, "logs")
    all_logs = {"zookeeper": zookeeper_logs_dir}
    return all_logs


def _get_runtime_endpoints(cluster_head_ip):
    # TODO: future to retrieve the endpoints from service discovery
    return None


def _handle_node_constraints_reached(
        runtime_config: Dict[str, Any], cluster_config: Dict[str, Any],
        node_type: str, head_info: Dict[str, Any], nodes_info: Dict[str, Any]):
    # We know this is called in the cluster scaler context
    server_ensemble = sort_nodes_by_seq_id(nodes_info)
    endpoint_uri = ""

    for node_info in server_ensemble:
        node_address = "{}:{}".format(
            node_info[RUNTIME_NODE_IP], ZOOKEEPER_SERVICE_PORT)
        if len(endpoint_uri) > 0:
            endpoint_uri += ","
        endpoint_uri += node_address

    _publish_service_endpoint_to_cluster(endpoint_uri)
    _publish_service_endpoint_to_workspace(cluster_config, endpoint_uri)


def _publish_service_endpoint_to_cluster(endpoint_uri: str) -> None:
    publish_cluster_variable("zookeeper-uri", endpoint_uri)


def _publish_service_endpoint_to_workspace(cluster_config: Dict[str, Any], endpoint_uri: str) -> None:
    workspace_name = cluster_config["workspace_name"]
    if workspace_name is None:
        return

    service_endpoints = {"zookeeper-uri": endpoint_uri}
    workspace_provider = _get_workspace_provider(cluster_config["provider"], workspace_name)
    workspace_provider.publish_global_variables(cluster_config, service_endpoints)


def _get_server_config(runtime_config: Dict[str, Any]):
    zookeeper_config = runtime_config.get(ZOOKEEPER_RUNTIME_CONFIG_KEY)
    if not zookeeper_config:
        return None

    return zookeeper_config.get("config")


def _get_runtime_services(
        runtime_config: Dict[str, Any], cluster_name: str) -> Dict[str, Any]:
    zookeeper_config = _get_config(runtime_config)
    service_name = get_canonical_service_name(
        zookeeper_config, cluster_name, ZOOKEEPER_SERVICE_NAME)
    services = {
        service_name: {
            SERVICE_DISCOVERY_PROTOCOL: SERVICE_DISCOVERY_PROTOCOL_TCP,
            SERVICE_DISCOVERY_PORT: ZOOKEEPER_SERVICE_PORT,
            SERVICE_DISCOVERY_NODE_KIND: SERVICE_DISCOVERY_NODE_KIND_WORKER
        },
    }
    return services


def _format_server_line(node_ip, seq_id):
    # below two lines are equivalent
    # server.id=node_ip:2888:3888;2181
    # server.id=node_ip:2888:3888:participant;0.0.0.0:2181
    return "server.{}={}:2888:3888;{}".format(seq_id, node_ip, ZOOKEEPER_SERVICE_PORT)


###################################
# Calls from node when configuring
###################################


def update_configurations():
    # Merge user specified configuration and default configuration
    runtime_config = subscribe_runtime_config()
    server_config = _get_server_config(runtime_config)
    if not server_config:
        return

    # Read in the existing configurations
    home_dir = _get_home_dir()
    server_properties_file = os.path.join(home_dir, "conf", "zoo.cfg")
    server_properties, comments = load_properties_file(server_properties_file)

    # Merge with the user configurations
    server_properties.update(server_config)

    # Write back the configuration file
    save_properties_file(server_properties_file, server_properties, comments=comments)


def configure_server_ensemble(nodes_info: Dict[str, Any]):
    # This method calls from node when configuring
    if nodes_info is None:
        raise RuntimeError("Missing nodes info for configuring server ensemble.")

    server_ensemble = sort_nodes_by_seq_id(nodes_info)
    _write_server_ensemble(server_ensemble)


def _write_server_ensemble(server_ensemble: List[Dict[str, Any]]):
    home_dir = _get_home_dir()
    zoo_cfg_file = os.path.join(home_dir, "conf", "zoo.cfg")

    mode = 'a' if os.path.exists(zoo_cfg_file) else 'w'
    with open(zoo_cfg_file, mode) as f:
        for node_info in server_ensemble:
            server_line = _format_server_line(
                node_info[RUNTIME_NODE_IP], node_info[RUNTIME_NODE_SEQ_ID])
            f.write("{}\n".format(server_line))


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


def _request_member_add(endpoints, node_ip, seq_id):
    home_dir = _get_home_dir()
    zk_cli = os.path.join(home_dir, "bin", "zkCli.sh")

    server_to_add = _format_server_line(node_ip, seq_id)
    # trying each node if failed
    last_error = None
    for endpoint in endpoints:
        try:
            _try_member_add(endpoint, zk_cli, server_to_add)
            # output should contain: Committed new configuration
            # success without exception
            return
        except NoQuorumError as quorum_error:
            raise quorum_error
        except Exception as e:
            # Other error retrying other endpoints
            print("Failed to add member through endpoint: "
                  "{}. Retrying with other endpoints...".format(
                    endpoint[RUNTIME_NODE_IP]))
            last_error = e
            continue

    if last_error is not None:
        raise last_error


def _try_member_add(endpoint, zk_cli, server_to_add):
    # zkCli.sh -server existing_server_ip:2181 reconfig -add server.id=node_ip:2888:3888;2181
    cmd = ["bash", zk_cli]
    endpoints_str = "{}:{}".format(
        endpoint[RUNTIME_NODE_IP], ZOOKEEPER_SERVICE_PORT)
    cmd += ["-server", endpoints_str]
    cmd += ["reconfig", "-add"]
    cmd += [quote(server_to_add)]

    cmd_str = " ".join(cmd)
    retries = ZOOKEEPER_QUORUM_RETRY
    env = os.environ.copy()
    env["ZOO_LOG4J_PROP"] = "ERROR,ROLLINGFILE"
    while retries > 0:
        try:
            return subprocess.check_output(
                cmd_str,
                shell=True,
                stderr=subprocess.STDOUT,
                env=env
            )
        except subprocess.CalledProcessError as e:
            retries -= 1
            output = e.output
            if output is not None:
                output_str = output.decode().strip()
                if "No quorum of new config is connected" in output_str:
                    # only retry for waiting for quorum
                    if retries == 0:
                        raise NoQuorumError("No quorum of new config is connected")
                    print("No quorum of new config is connected. "
                          "Waiting {} seconds and retrying...".format(
                            ZOOKEEPER_QUORUM_RETRY_INTERVAL))
                    continue
            raise e
