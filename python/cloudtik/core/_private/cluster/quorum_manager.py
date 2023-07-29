import hashlib
import json
import logging

from cloudtik.core._private.runtime_utils import RUNTIME_NODE_SEQ_ID, RUNTIME_NODE_IP, RUNTIME_NODE_ID
from cloudtik.core._private.state.kv_store import kv_put
from cloudtik.core._private.utils import _get_minimal_nodes_before_update, CLOUDTIK_CLUSTER_NODES_INFO_NODE_TYPE, \
    _notify_minimal_nodes_reached
from cloudtik.core.tags import (
    CLOUDTIK_TAG_USER_NODE_TYPE, CLOUDTIK_TAG_NODE_SEQ_ID, CLOUDTIK_TAG_HEAD_NODE_SEQ_ID,
    CLOUDTIK_TAG_QUORUM_ID)

logger = logging.getLogger(__name__)


class QuorumManager:
    """Quorum Manager is in charge of managing a cluster nodes to form a quorum.
    A quorum cluster of nodes usually different from a normal cluster in which each
    node is horizontally independently scale. While quorum cluster forms a consistency
    protocol which needs agreement to scale new nodes after the cluster is bootstrapped.
    """

    def __init__(
            self, config, provider
    ):
        self.config = config
        self.provider = provider
        self.available_node_types = config["available_node_types"] if config else None

        self.published_nodes_info_hashes = {}

        # These are initialized for each config change with reset
        self.minimal_nodes_before_update = {}
        self.node_types_quorum_id_to_nodes = {}

        # Set at each update by calling update
        self.non_terminated_nodes = None

    def reset(self, config, provider):
        self.config = config
        self.provider = provider
        self.available_node_types = self.config["available_node_types"]

        # Collect the minimal nodes before update requirements
        self._collect_minimal_nodes_before_update()

    def update(self, non_terminated_nodes):
        self.non_terminated_nodes = non_terminated_nodes
        self.collect_quorum_minimal_nodes()

    def _collect_minimal_nodes_before_update(self):
        # Push global runtime config
        minimal_nodes = {}
        for node_type in self.available_node_types:
            minimal_nodes_for_node_type = _get_minimal_nodes_before_update(
                self.config, node_type)
            if minimal_nodes_for_node_type:
                minimal_nodes[node_type] = minimal_nodes_for_node_type
        self.minimal_nodes_before_update = minimal_nodes

    def _collect_nodes_info(self):
        nodes_info_map = {}
        for node_id in self.non_terminated_nodes.all_node_ids:
            tags = self.provider.node_tags(node_id)
            if CLOUDTIK_TAG_USER_NODE_TYPE in tags:
                node_type = tags[CLOUDTIK_TAG_USER_NODE_TYPE]
                if node_type not in nodes_info_map:
                    nodes_info_map[node_type] = {}
                nodes_info = nodes_info_map[node_type]

                node_info = {RUNTIME_NODE_IP: self.provider.internal_ip(node_id)}
                if CLOUDTIK_TAG_NODE_SEQ_ID in tags:
                    node_info[RUNTIME_NODE_SEQ_ID] = int(tags[CLOUDTIK_TAG_NODE_SEQ_ID])
                nodes_info[node_id] = node_info

        return nodes_info_map

    def wait_for_minimal_nodes_before_update(self):
        if not self.minimal_nodes_before_update:
            # No need to wait
            return False

        # Make sure only minimal requirement > 0 will appear in self.minimal_nodes_before_update
        nodes_info_map = self._collect_nodes_info()
        for node_type in self.minimal_nodes_before_update:
            minimal_nodes_info = self.minimal_nodes_before_update[node_type]
            if node_type not in nodes_info_map:
                self._print_info_waiting_for(minimal_nodes_info, 0, "minimal")
                return True

            if minimal_nodes_info["quorum"] and self._exists_a_quorum(node_type):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Cluster Controller: Quorum exists for operating {}. No waiting.".format(
                        node_type))
                continue

            nodes_info = nodes_info_map[node_type]
            nodes_number = len(nodes_info)
            if minimal_nodes_info["minimal"] > nodes_number:
                self._print_info_waiting_for(minimal_nodes_info, nodes_number, "minimal")
                return True

            # Check whether the internal ip are all available
            for node_id, node_info in nodes_info.items():
                if node_info.get(RUNTIME_NODE_IP) is None:
                    self._print_info_waiting_for(minimal_nodes_info, nodes_number, "IP available")
                    return True

            logger.info(
                "Cluster Controller: Minimal nodes requirement satisfied for {}: {}.".format(
                    node_type, minimal_nodes_info["minimal"]))
            # publish nodes will check whether it has changed since last publish
            self._publish_nodes_info(node_type, nodes_info, minimal_nodes_info)

        # All satisfied if come to here
        return False

    def _publish_nodes_info(self, node_type: str, nodes_info, minimal_nodes_info):
        nodes_info_data = json.dumps(nodes_info, sort_keys=True)

        hasher = hashlib.sha1()
        hasher.update(nodes_info_data.encode("utf-8"))
        new_nodes_info_hash = hasher.hexdigest()

        published_nodes_info_hash = self.published_nodes_info_hashes.get(node_type)
        if published_nodes_info_hash and new_nodes_info_hash == published_nodes_info_hash:
            return False
        self.published_nodes_info_hashes[node_type] = new_nodes_info_hash

        # Minimal number of the nodes reached, set the quorum id of the new joined nodes
        self._form_a_quorum(node_type, new_nodes_info_hash)

        logger.info(
            "Cluster Controller: Publish and notify nodes info for {}".format(
                node_type))

        nodes_info_key = CLOUDTIK_CLUSTER_NODES_INFO_NODE_TYPE.format(node_type)
        kv_put(nodes_info_key, nodes_info_data, overwrite=True)

        # Notify runtime of these
        self._notify_minimal_nodes_reached(node_type, nodes_info, minimal_nodes_info)
        return True

    def _notify_minimal_nodes_reached(
            self, node_type: str, nodes_info, minimal_nodes_info):
        head_id = self.non_terminated_nodes.head_id
        head_node_ip = self.provider.internal_ip(head_id)
        head_info = {
            RUNTIME_NODE_ID: head_id,
            RUNTIME_NODE_IP: head_node_ip,
            RUNTIME_NODE_SEQ_ID: CLOUDTIK_TAG_HEAD_NODE_SEQ_ID
        }
        _notify_minimal_nodes_reached(
            self.config, node_type, head_info,
            nodes_info, minimal_nodes_info)

    def is_launch_allowed(self, node_type: str):
        if self._is_minimal_nodes(node_type) and (
                self._is_quorum_minimal_nodes(node_type)) and (
                not self._is_quorum_minimal_nodes_in_launch(node_type)):
            return False
        return True

    def _is_minimal_nodes(self, node_type: str):
        if not self.minimal_nodes_before_update:
            return False
        if node_type not in self.minimal_nodes_before_update:
            return False
        return True

    def _is_quorum_minimal_nodes(self, node_type: str):
        # Usually, for the cases to control minimal nodes for update
        # We are targeting for fixed nodes once it started
        minimal_nodes_info = self.minimal_nodes_before_update[node_type]
        return minimal_nodes_info["quorum"]

    def _exists_a_quorum(self, node_type: str):
        quorum, minimal = self._get_quorum(node_type)
        quorum_id_to_nodes = self.node_types_quorum_id_to_nodes.get(
            node_type, {})
        for node_quorum_id in quorum_id_to_nodes:
            quorum_id_nodes = quorum_id_to_nodes[node_quorum_id]
            remaining = len(quorum_id_nodes)
            if remaining >= quorum:
                return True
        return False

    def _form_a_quorum(self, node_type: str, quorum_id):
        if not self._is_quorum_minimal_nodes(node_type):
            return

        for node_id in self.non_terminated_nodes.worker_ids:
            tags = self.provider.node_tags(node_id)
            this_node_type = tags.get(CLOUDTIK_TAG_USER_NODE_TYPE)
            if node_type != this_node_type:
                continue
            node_quorum_id = tags.get(CLOUDTIK_TAG_QUORUM_ID)
            if node_quorum_id:
                continue

            # New node, assign the quorum_id
            self.provider.set_node_tags(
                node_id, {CLOUDTIK_TAG_QUORUM_ID: quorum_id})
            self._update_quorum_id_to_nodes(
                node_type, node_quorum_id, node_id)

    def _get_quorum(self, node_type: str):
        minimal_nodes_info = self.minimal_nodes_before_update[node_type]
        minimal = minimal_nodes_info["minimal"]
        quorum = int(minimal / 2) + 1
        return quorum, minimal

    def terminate_for_quorum(self, node_type: str, node_id):
        if (not self._is_minimal_nodes(node_type)) or (
                not self._is_quorum_minimal_nodes(node_type)):
            return False

        quorum, minimal = self._get_quorum(node_type)
        # Check whether the node is an invalid quorum member
        quorum_id_to_nodes = self.node_types_quorum_id_to_nodes.get(
            node_type, {})
        for node_quorum_id in quorum_id_to_nodes:
            quorum_id_nodes = quorum_id_to_nodes[node_quorum_id]
            if node_id in quorum_id_nodes:
                remaining = len(quorum_id_nodes)
                if remaining < quorum:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Node {} is not bad quorum member: {} ({}/{}). Will be terminated".format(
                            node_id, node_quorum_id, remaining, minimal))
                    return True
                return False
        return False

    def _update_quorum_id_to_nodes(
            self, node_type: str, node_quorum_id: str, node_id: str):
        if node_type not in self.node_types_quorum_id_to_nodes:
            self.node_types_quorum_id_to_nodes[node_type] = {}
        quorum_id_to_nodes = self.node_types_quorum_id_to_nodes[node_type]
        if node_quorum_id not in quorum_id_to_nodes:
            quorum_id_to_nodes[node_quorum_id] = set()
        quorum_id_nodes = quorum_id_to_nodes[node_quorum_id]
        quorum_id_nodes.add(node_id)

    def collect_quorum_minimal_nodes(self):
        if not self.minimal_nodes_before_update:
            # No need
            return

        self.node_types_quorum_id_to_nodes = {}
        for node_id in self.non_terminated_nodes.worker_ids:
            tags = self.provider.node_tags(node_id)
            node_type = tags.get(CLOUDTIK_TAG_USER_NODE_TYPE)
            if not node_type:
                continue
            if node_type not in self.minimal_nodes_before_update:
                continue
            node_quorum_id = tags.get(CLOUDTIK_TAG_QUORUM_ID)
            if not node_quorum_id:
                continue

            self._update_quorum_id_to_nodes(
                node_type, node_quorum_id, node_id)

    def _is_quorum_minimal_nodes_in_launch(self, node_type: str):
        quorum, minimal = self._get_quorum(node_type)
        # Only when a quorum of the minimal nodes dead,
        # we can launch new nodes and form a new quorum
        quorum_id_to_nodes = self.node_types_quorum_id_to_nodes.get(
            node_type, {})
        for node_quorum_id in quorum_id_to_nodes:
            quorum_id_nodes = quorum_id_to_nodes[node_quorum_id]
            remaining = len(quorum_id_nodes)
            if remaining >= quorum:
                # One quorum id exceed the quorum
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("No new node launch allowed with the existence of a valid quorum: {} ({}/{}).".format(
                        node_quorum_id, remaining, minimal))
                return False

        # none of the quorum_id exceeding a quorum
        logger.info(
            "Cluster Controller: None of the quorum id of {} forms a quorum ({}/{})."
            " Quorum launch.".format(node_type, quorum, minimal))
        return True

    @staticmethod
    def _print_info_waiting_for(minimal_nodes_info, nodes_number, for_what):
        logger.info("Cluster Controller: waiting for {} of {}/{} nodes required by runtimes: {}".format(
            for_what, nodes_number, minimal_nodes_info["minimal"], minimal_nodes_info["runtimes"]))
