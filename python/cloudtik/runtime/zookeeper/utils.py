import os
from typing import Any, Dict, List

from cloudtik.core._private.runtime_utils import subscribe_runtime_config
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


def _get_config(runtime_config: Dict[str, Any]):
    return runtime_config.get(ZOOKEEPER_RUNTIME_CONFIG_KEY, {})


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _with_runtime_environment_variables(runtime_config, config, provider, node_id: str):
    runtime_envs = {"ZOOKEEPER_ENABLED": True}
    return runtime_envs


def _get_runtime_logs():
    zookeeper_logs_dir = os.path.join(os.getenv("ZOOKEEPER_HOME"), "logs")
    all_logs = {"zookeeper": zookeeper_logs_dir}
    return all_logs


def _get_head_service_urls(cluster_head_ip):
    # TODO: how to get the ZooKeeper service address which established after head node
    return None


def _configure_server_ensemble(nodes_info: Dict[str, Any]):
    if nodes_info is None:
        raise RuntimeError("Missing nodes info for configuring server ensemble.")

    server_ensemble = _server_ensemble_from_nodes_info(nodes_info)
    _write_server_ensemble(server_ensemble)


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


def _write_server_ensemble(server_ensemble: List[Dict[str, Any]]):
    zoo_cfg_file = os.path.join(
        os.getenv("ZOOKEEPER_HOME"), "conf", "zoo.cfg")

    mode = 'a' if os.path.exists(zoo_cfg_file) else 'w'
    with open(zoo_cfg_file, mode) as f:
        for node_info in server_ensemble:
            f.write("server.{}={}:2888:3888\n".format(
                node_info["node_number"], node_info["node_ip"]))


def _handle_minimal_nodes_reached(
        runtime_config: Dict[str, Any], cluster_config: Dict[str, Any],
        node_type: str, head_info: Dict[str, Any], nodes_info: Dict[str, Any]):
    # We know this is called in the cluster scaler context
    server_ensemble = _server_ensemble_from_nodes_info(nodes_info)
    service_uri = ""

    for node_info in server_ensemble:
        node_address = "{}:{}".format(
            node_info["node_ip"], ZOOKEEPER_SERVICE_PORT)
        if len(service_uri) > 0:
            service_uri += ","
        service_uri += node_address

    _publish_service_uri_to_cluster(service_uri)
    _publish_service_uri_to_workspace(cluster_config, service_uri)


def _publish_service_uri_to_cluster(service_uri: str) -> None:
    publish_cluster_variable("zookeeper-uri", service_uri)


def _publish_service_uri_to_workspace(cluster_config: Dict[str, Any], service_uri: str) -> None:
    workspace_name = cluster_config["workspace_name"]
    if workspace_name is None:
        return

    service_uris = {"zookeeper-uri": service_uri}
    workspace_provider = _get_workspace_provider(cluster_config["provider"], workspace_name)
    workspace_provider.publish_global_variables(cluster_config, service_uris)


def _get_server_config(runtime_config: Dict[str, Any]):
    zookeeper_config = runtime_config.get(ZOOKEEPER_RUNTIME_CONFIG_KEY)
    if not zookeeper_config:
        return None

    return zookeeper_config.get("config")


def update_configurations():
    # Merge user specified configuration and default configuration
    runtime_config = subscribe_runtime_config()
    server_config = _get_server_config(runtime_config)
    if not server_config:
        return

    # Read in the existing configurations
    server_properties_file = os.path.join(os.getenv("ZOOKEEPER_HOME"), "conf/zoo.cfg")
    server_properties, comments = load_properties_file(server_properties_file)

    # Merge with the user configurations
    server_properties.update(server_config)

    # Write back the configuration file
    save_properties_file(server_properties_file, server_properties, comments=comments)


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
