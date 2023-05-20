import copy
import yaml
from filelock import FileLock
from threading import RLock
import json
import os
import socket
import logging
from typing import Any, Dict

from cloudtik.core.node_provider import NodeProvider

from cloudtik.providers._private.onprem.config import get_cloud_simulator_lock_path, \
    get_cloud_simulator_state_path, _get_instance_types, \
    get_list_of_node_ips, _get_request_instance_type, _get_node_id_mapping, _get_node_instance_type, \
    set_node_types_resources

logger = logging.getLogger(__name__)


def load_provider_config(config_file):
    with open(config_file) as f:
        config_object = yaml.safe_load(f) or {}

    return config_object


class ClusterState:
    def __init__(self, lock_path, state_path, provider_config):
        self.lock = RLock()
        self.file_lock = FileLock(lock_path)
        self.state_path = state_path
        self.cached_nodes = {}
        self.load_config(provider_config)

    def get(self):
        with self.lock:
            with self.file_lock:
                return copy.deepcopy(self.cached_nodes)

    def get_safe(self):
        return self.cached_nodes

    def get_node(self, node_id):
        with self.lock:
            with self.file_lock:
                nodes = self.cached_nodes
                if node_id not in nodes:
                    return None
                return copy.deepcopy(nodes[node_id])

    def put(self, node_id, node):
        assert "tags" in node
        assert "state" in node
        with self.lock:
            with self.file_lock:
                nodes = self.cached_nodes
                nodes[node_id] = node
                with open(self.state_path, "w") as f:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Writing cluster state: {}".format(list(nodes)))
                    f.write(json.dumps(nodes))

    def load_config(self, provider_config):
        with self.lock:
            with self.file_lock:
                list_of_node_ips = get_list_of_node_ips(provider_config)
                if os.path.exists(self.state_path):
                    nodes = json.loads(open(self.state_path).read())
                else:
                    nodes = {}
                logger.info(
                    "Loaded cluster state: {}".format(nodes))

                # Filter removed node ips.
                for node_ip in list(nodes):
                    if node_ip not in list_of_node_ips:
                        del nodes[node_ip]

                for node_ip in list_of_node_ips:
                    if node_ip not in nodes:
                        nodes[node_ip] = {
                            "tags": {},
                            "state": "terminated",
                        }
                assert len(nodes) == len(list_of_node_ips)
                with open(self.state_path, "w") as f:
                    logger.info("Writing cluster state: {}".format(nodes))
                    f.write(json.dumps(nodes))
                self.cached_nodes = nodes


class CloudSimulatorNodeProvider(NodeProvider):
    """NodeProvider for private/on-promise clusters.
    This is not a real Node Provider that will be used by core (which
    may be instanced multiple times and on different environment)

    `node_id` is overloaded to also be `node_ip` in this class.

    It manages multiple clusters in a unified state file that requires each node to be tagged with
    CLOUDTIK_TAG_CLUSTER_NAME in create and non_terminated_nodes function calls to
    associate each node with the right cluster.

    The current use case of managing multiple clusters is by
    CloudSimulator which receives node provider HTTP requests
    from OnPremNodeProvider and uses CloudSimulatorNodeProvider to get
    the responses.
    """

    def __init__(self, provider_config, cluster_name):
        NodeProvider.__init__(self, provider_config, cluster_name)

        # CloudSimulatorNodeProvider with a Cloud Simulator.
        self.state = ClusterState(
            get_cloud_simulator_lock_path(), get_cloud_simulator_state_path(),
            provider_config)
        self.node_id_mapping = _get_node_id_mapping(provider_config)

    def non_terminated_nodes(self, tag_filters):
        nodes = self.state.get()
        matching_ips = []
        for node_ip, node in nodes.items():
            if node["state"] == "terminated":
                continue
            ok = True
            for k, v in tag_filters.items():
                if node["tags"].get(k) != v:
                    ok = False
                    break
            if ok:
                matching_ips.append(node_ip)
        return matching_ips

    def is_running(self, node_id):
        node = self.state.get_node(node_id)
        if node is None:
            return False
        return node["state"] == "running"

    def is_terminated(self, node_id):
        return not self.is_running(node_id)

    def node_tags(self, node_id):
        node = self.state.get_node(node_id)
        if node is None:
            raise RuntimeError("Node with id {} doesn't exist.".format(node_id))
        return node["tags"]

    def external_ip(self, node_id):
        """Returns an external ip if the user has supplied one.
        Otherwise, use the same logic as internal_ip below.

        This can be used to call cloudtik up from outside the network, for example
        if the cluster exists in an AWS VPC and we're interacting with
        the cluster from a laptop (where using an internal_ip will not work).

        Useful for debugging the on-prem node provider with cloud VMs."""

        node = self.node_id_mapping[node_id]
        ext_ip = node.get("external_ip")
        if ext_ip:
            return ext_ip
        else:
            return socket.gethostbyname(node_id)

    def internal_ip(self, node_id):
        return socket.gethostbyname(node_id)

    def set_node_tags(self, node_id, tags):
        with self.state.file_lock:
            node = self.state.get_node(node_id)
            if node is None:
                raise RuntimeError("Node with id {} doesn't exist.".format(node_id))
            node["tags"].update(tags)
            self.state.put(node_id, node)

    def create_node(self, node_config, tags, count):
        """Creates min(count, currently available) nodes."""
        launched = 0
        instance_type = _get_request_instance_type(node_config)
        with self.state.file_lock:
            nodes = self.state.get_safe()
            for node_id, node in nodes.items():
                if node["state"] != "terminated":
                    continue

                node_instance_type = self.get_node_instance_type(node_id)
                if instance_type != node_instance_type:
                    continue

                node["tags"] = tags
                node["state"] = "running"
                self.state.put(node_id, node)
                launched = launched + 1
                if count == launched:
                    return
        if launched < count:
            raise RuntimeError(
                "No enough free nodes. {} nodes requested / {} launched.".format(
                    count, launched))

    def terminate_node(self, node_id):
        node = self.state.get_node(node_id)
        if node is None:
            raise RuntimeError("Node with id {} doesn't exist.".format(node_id))
        node["state"] = "terminated"
        self.state.put(node_id, node)

    def get_node_info(self, node_id):
        node = self.state.get_node(node_id)
        if node is None:
            raise RuntimeError("Node with id {} doesn't exist.".format(node_id))
        node_instance_type = self.get_node_instance_type(node_id)
        node_info = {"node_id": node_id,
                     "instance_type": node_instance_type,
                     "private_ip": self.internal_ip(node_id),
                     "public_ip": self.external_ip(node_id),
                     "instance_status": node["state"]}
        node_info.update(node["tags"])
        return node_info

    def with_environment_variables(self, node_type_config: Dict[str, Any], node_id: str):
        return {}

    @staticmethod
    def post_prepare(
            cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Fills out missing fields after the user config is merged with defaults and before validate"""
        instance_types = _get_instance_types(cluster_config["provider"])
        set_node_types_resources(cluster_config, instance_types)
        return cluster_config

    def get_instance_types(self):
        """Return the all instance types information"""
        return _get_instance_types(self.provider_config)

    def get_node_instance_type(self, node_id):
        return _get_node_instance_type(self.node_id_mapping, node_id)

    def reload(self, config_file):
        provider_config = load_provider_config(config_file)
        self.state.load_config(provider_config)
