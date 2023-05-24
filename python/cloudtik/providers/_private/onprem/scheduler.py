import logging
import socket
from typing import Any, Dict

import yaml

from cloudtik.core.node_provider import NodeProvider
from cloudtik.providers._private.onprem.config import get_cloud_simulator_lock_path, \
    get_cloud_simulator_state_path, _get_instance_types, \
    _get_request_instance_type, _get_node_id_mapping, _get_node_instance_type, \
    set_node_types_resources
from cloudtik.providers._private.onprem.state_store import FileStateStore

logger = logging.getLogger(__name__)


def load_provider_config(config_file):
    with open(config_file) as f:
        config_object = yaml.safe_load(f) or {}

    return config_object


class CloudSimulatorScheduler(NodeProvider):
    """NodeProvider for private/on-promise clusters.
    This is not a real Node Provider that will be used by core (which
    may be instanced multiple times and on different environment)

    `node_id` is overloaded to also be `node_ip` in this class.

    It manages multiple clusters in a unified state file that requires each node to be tagged with
    CLOUDTIK_TAG_CLUSTER_NAME in create and non_terminated_nodes function calls to
    associate each node with the right cluster.

    The current use case of managing multiple clusters is by
    CloudSimulator which receives node provider HTTP requests
    from OnPremNodeProvider and uses CloudSimulatorScheduler to get
    the responses.
    """

    def __init__(self, provider_config, cluster_name):
        NodeProvider.__init__(self, provider_config, cluster_name)

        # CloudSimulatorScheduler with a Cloud Simulator.
        self.state = FileStateStore(
            provider_config,
            get_cloud_simulator_lock_path(),
            get_cloud_simulator_state_path())
        self.node_id_mapping = _get_node_id_mapping(provider_config)

    def non_terminated_nodes(self, tag_filters):
        nodes = self.state.get_nodes()
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
        with self.state.transaction():
            node = self.state.get_node_safe(node_id)
            if node is None:
                raise RuntimeError("Node with id {} doesn't exist.".format(node_id))
            node["tags"].update(tags)
            self.state.put_node_safe(node_id, node)

    def create_node(self, node_config, tags, count):
        """Creates min(count, currently available) nodes."""
        launched = 0
        instance_type = _get_request_instance_type(node_config)
        with self.state.transaction():
            nodes = self.state.get_nodes_safe()
            for node_id, node in nodes.items():
                if node["state"] != "terminated":
                    continue

                node_instance_type = self.get_node_instance_type(node_id)
                if instance_type != node_instance_type:
                    continue

                node["tags"] = tags
                node["state"] = "running"
                self.state.put_node_safe(node_id, node)
                launched = launched + 1
                if count == launched:
                    return
        if launched < count:
            raise RuntimeError(
                "No enough free nodes. {} nodes requested / {} launched.".format(
                    count, launched))

    def terminate_node(self, node_id):
        with self.state.transaction():
            node = self.state.get_node_safe(node_id)
            if node is None:
                raise RuntimeError("Node with id {} doesn't exist.".format(node_id))
            if node["state"] != "running":
                raise RuntimeError("Node with id {} is not running.".format(node_id))
            node["state"] = "terminated"
            self.state.put_node_safe(node_id, node)

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

    def create_workspace(self, workspace_name):
        self.state.create_workspace(workspace_name)
        return {"name": workspace_name}

    def delete_workspace(self, workspace_name):
        self.state.delete_workspace(workspace_name)
        return {"name": workspace_name}

    def get_workspace(self, workspace_name):
        return self.state.get_workspace(workspace_name)
