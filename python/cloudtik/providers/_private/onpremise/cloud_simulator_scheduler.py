import logging
from typing import Any, Dict, Optional, List

import yaml

from cloudtik.core._private.utils import is_head_node_by_tags
from cloudtik.core.tags import CLOUDTIK_TAG_WORKSPACE_NAME
from cloudtik.providers._private.onpremise.config import get_cloud_simulator_lock_path, \
    get_cloud_simulator_state_path, _get_instance_types, \
    _get_request_instance_type, _get_node_id_mapping, _get_node_instance_type
from cloudtik.providers._private.onpremise.state_store import FileStateStore

logger = logging.getLogger(__name__)


def load_provider_config(config_file):
    with open(config_file) as f:
        config_object = yaml.safe_load(f) or {}

    return config_object


class CloudSimulatorScheduler:
    def __init__(self, provider_config, cluster_name):
        self.provider_config = provider_config
        self.cluster_name = cluster_name

        self.state = FileStateStore(
            provider_config,
            get_cloud_simulator_lock_path(),
            get_cloud_simulator_state_path())
        self.node_id_mapping = _get_node_id_mapping(provider_config)

    def list_nodes(self, workspace_name, tag_filters):
        # List nodes that are not cluster specific, ignoring the cluster name
        tag_filters = {} if tag_filters is None else tag_filters
        tag_filters[CLOUDTIK_TAG_WORKSPACE_NAME] = workspace_name
        return self._list_nodes(tag_filters)

    def _list_nodes(self, tag_filters):
        nodes = self.state.get_nodes()
        matching_nodes = []
        for node_id, node in nodes.items():
            if node["state"] == "terminated":
                continue
            ok = True
            for k, v in tag_filters.items():
                if node["tags"].get(k) != v:
                    ok = False
                    break
            if ok:
                matching_nodes.append(node)
        return matching_nodes

    def describe_node(self, node_id):
        node = self.state.get_node(node_id)
        return node

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
            # head node prefer with node specified with external IP
            # first trying node with external ip specified
            if is_head_node_by_tags(tags):
                launched = self._launch_node(
                    nodes, tags, count, launched, instance_type, True)
                if count == launched:
                    return
            launched = self._launch_node(
                nodes, tags, count, launched, instance_type)

        if launched < count:
            raise RuntimeError(
                "No enough free nodes. {} nodes requested / {} launched.".format(
                    count, launched))

    def _launch_node(
            self, nodes, tags, count, launched,
            instance_type, with_external_ip=False):
        for node_id, node in nodes.items():
            if node["state"] != "terminated":
                continue

            node_instance_type = self.get_node_instance_type(node_id)
            if instance_type != node_instance_type:
                continue

            if with_external_ip:
                # A previous running node was removed
                provider_node = self.node_id_mapping.get(node_id)
                if not provider_node:
                    continue
                external_ip = provider_node.get("external_ip")
                if not external_ip:
                    continue

            node["tags"] = tags
            node["state"] = "running"
            self.state.put_node_safe(node_id, node)
            launched = launched + 1
            if count == launched:
                return launched
        return launched

    def terminate_nodes(self, node_ids: List[str]) -> Optional[Dict[str, Any]]:
        """Terminates a set of nodes.
        May be overridden with a batch method, which optionally may return a
        mapping from deleted node ids to node metadata.
        """
        for node_id in node_ids:
            self.terminate_node(node_id)
        return None

    def terminate_node(self, node_id):
        with self.state.transaction():
            node = self.state.get_node_safe(node_id)
            if node is None:
                raise RuntimeError("Node with id {} doesn't exist.".format(node_id))
            if node["state"] != "running":
                raise RuntimeError("Node with id {} is not running.".format(node_id))
            node["state"] = "terminated"
            self.state.put_node_safe(node_id, node)

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
