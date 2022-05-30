from filelock import FileLock
from threading import RLock
import json
import os
import socket
import logging
from typing import Any, Dict

from cloudtik.core.node_provider import NodeProvider

from cloudtik.providers._private.local.config import get_cloud_simulator_lock_path, \
    get_cloud_simulator_state_path, _get_instance_types, \
    get_list_of_node_ips, _get_request_instance_type, _get_node_id_mapping, _get_node_instance_type

logger = logging.getLogger(__name__)

filelock_logger = logging.getLogger("filelock")
filelock_logger.setLevel(logging.WARNING)


class ClusterState:
    def __init__(self, lock_path, save_path, provider_config):
        self.lock = RLock()
        self.file_lock = FileLock(lock_path)
        self.save_path = save_path

        with self.lock:
            with self.file_lock:
                list_of_node_ips = get_list_of_node_ips(provider_config)
                if os.path.exists(self.save_path):
                    nodes = json.loads(open(self.save_path).read())
                else:
                    nodes = {}
                logger.info(
                    "Cluster State: "
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
                with open(self.save_path, "w") as f:
                    logger.info(
                        "Cluster State: "
                        "Writing cluster state: {}".format(nodes))
                    f.write(json.dumps(nodes))

    def get(self):
        with self.lock:
            with self.file_lock:
                nodes = json.loads(open(self.save_path).read())
                return nodes

    def put(self, node_id, info):
        assert "tags" in info
        assert "state" in info
        with self.lock:
            with self.file_lock:
                nodes = self.get()
                nodes[node_id] = info
                with open(self.save_path, "w") as f:
                    logger.debug("Cluster State: "
                                 "Writing cluster state: {}".format(
                                    list(nodes)))
                    f.write(json.dumps(nodes))


class LocalNodeProvider(NodeProvider):
    """NodeProvider for private/local clusters.

    `node_id` is overloaded to also be `node_ip` in this class.

    It manages multiple clusters in a unified state file that requires each node to be tagged with
    CLOUDTIK_TAG_CLUSTER_NAME in create and non_terminated_nodes function calls to
    associate each node with the right cluster.

    The current use case of managing multiple clusters is by
    CloudSimulator which receives node provider HTTP requests
    from CloudSimulatorNodeProvider and uses LocalNodeProvider to get
    the responses.
    """

    def __init__(self, provider_config, cluster_name):
        NodeProvider.__init__(self, provider_config, cluster_name)

        # LocalNodeProvider with a Cloud Simulator.
        self.state = ClusterState(
            get_cloud_simulator_lock_path(), get_cloud_simulator_state_path(),
            provider_config)
        self.node_id_mapping = _get_node_id_mapping(provider_config)

    def non_terminated_nodes(self, tag_filters):
        nodes = self.state.get()
        matching_ips = []
        for node_ip, info in nodes.items():
            if info["state"] == "terminated":
                continue
            ok = True
            for k, v in tag_filters.items():
                if info["tags"].get(k) != v:
                    ok = False
                    break
            if ok:
                matching_ips.append(node_ip)
        return matching_ips

    def is_running(self, node_id):
        return self.state.get()[node_id]["state"] == "running"

    def is_terminated(self, node_id):
        return not self.is_running(node_id)

    def node_tags(self, node_id):
        return self.state.get()[node_id]["tags"]

    def external_ip(self, node_id):
        """Returns an external ip if the user has supplied one.
        Otherwise, use the same logic as internal_ip below.

        This can be used to call cloudtik up from outside the network, for example
        if the cluster exists in an AWS VPC and we're interacting with
        the cluster from a laptop (where using an internal_ip will not work).

        Useful for debugging the local node provider with cloud VMs."""

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
            info = self.state.get()[node_id]
            info["tags"].update(tags)
            self.state.put(node_id, info)

    def create_node(self, node_config, tags, count):
        """Creates min(count, currently available) nodes."""
        instance_type = _get_request_instance_type(node_config)
        with self.state.file_lock:
            nodes = self.state.get()
            for node_id, info in nodes.items():
                if info["state"] != "terminated":
                    continue

                node_instance_type = self.get_node_instance_type(node_id)
                if instance_type != node_instance_type:
                    continue

                info["tags"] = tags
                info["state"] = "running"
                self.state.put(node_id, info)
                count = count - 1
                if count == 0:
                    return

    def terminate_node(self, node_id):
        nodes = self.state.get()
        info = nodes[node_id]
        info["state"] = "terminated"
        self.state.put(node_id, info)

    def get_node_info(self, node_id):
        node = self.state.get()[node_id]
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

    def get_instance_types(self):
        """Return the all instance types information"""
        return _get_instance_types(self.provider_config)

    def get_node_instance_type(self, node_id):
        return _get_node_instance_type(self.node_id_mapping, node_id)
