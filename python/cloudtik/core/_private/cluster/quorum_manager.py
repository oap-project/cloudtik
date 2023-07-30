import hashlib
import json
import logging
from typing import Optional, List

from cloudtik.core._private.runtime_utils import RUNTIME_NODE_SEQ_ID, RUNTIME_NODE_IP, RUNTIME_NODE_ID, \
    RUNTIME_NODE_QUORUM_JOIN
from cloudtik.core._private.state.kv_store import kv_put
from cloudtik.core._private.utils import _get_node_constraints_for_node_type, CLOUDTIK_CLUSTER_NODES_INFO_NODE_TYPE, \
    _notify_node_constraints_reached
from cloudtik.core.tags import (
    CLOUDTIK_TAG_USER_NODE_TYPE, CLOUDTIK_TAG_NODE_SEQ_ID, CLOUDTIK_TAG_HEAD_NODE_SEQ_ID,
    CLOUDTIK_TAG_QUORUM_ID, CLOUDTIK_TAG_QUORUM_JOIN, QUORUM_JOIN_STATUS_INIT)

logger = logging.getLogger(__name__)


class NodeConstraints:
    def __init__(
            self, minimal: int, quorum: bool, scalable: bool, runtimes: List[str]
    ):
        self.minimal = minimal
        self.quorum = quorum
        self.scalable = scalable
        self.runtimes = runtimes


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
        self.node_constraints_by_node_type = {}
        self.quorum_id_to_nodes_by_node_type = {}

        # Set at each update by calling update
        self.non_terminated_nodes = None

        # Refresh at each wait for update
        self.nodes_info_by_node_type = None

    def reset(self, config, provider):
        self.config = config
        self.provider = provider
        self.available_node_types = self.config["available_node_types"]

        # Collect the nodes constraints
        self._collect_node_constraints()

    def update(self, non_terminated_nodes):
        self.non_terminated_nodes = non_terminated_nodes
        self._collect_quorum_nodes()

    def _collect_node_constraints(self):
        # Push global runtime config
        node_constraints_by_type = {}
        for node_type in self.available_node_types:
            node_constraints_of_node_type = _get_node_constraints_for_node_type(
                self.config, node_type)
            if node_constraints_of_node_type is not None:
                minimal, quorum, scalable, runtimes = node_constraints_of_node_type
                node_constraints_by_type[node_type] = NodeConstraints(
                    minimal, quorum, scalable, runtimes)
        self.node_constraints_by_node_type = node_constraints_by_type

    def _collect_nodes_info(self):
        # We only collect nodes of related node types
        nodes_info_of_node_type = {}
        for node_id in self.non_terminated_nodes.worker_ids:
            tags = self.provider.node_tags(node_id)
            node_type = tags.get(CLOUDTIK_TAG_USER_NODE_TYPE)
            if not node_type:
                continue
            if node_type not in self.node_constraints_by_node_type:
                continue

            if node_type not in nodes_info_of_node_type:
                nodes_info_of_node_type[node_type] = {}
            nodes_info = nodes_info_of_node_type[node_type]

            node_info = {RUNTIME_NODE_IP: self.provider.internal_ip(node_id)}
            if CLOUDTIK_TAG_NODE_SEQ_ID in tags:
                node_info[RUNTIME_NODE_SEQ_ID] = int(tags[CLOUDTIK_TAG_NODE_SEQ_ID])
            if CLOUDTIK_TAG_QUORUM_JOIN in tags:
                node_info[RUNTIME_NODE_QUORUM_JOIN] = tags[CLOUDTIK_TAG_QUORUM_JOIN]

            nodes_info[node_id] = node_info

        return nodes_info_of_node_type

    def wait_for_update(self):
        if not self.node_constraints_by_node_type:
            # No need to wait for most cases, fast return
            return False

        # Make sure only minimal requirement > 0 will appear in self.node_constraints_by_node_type
        self.nodes_info_by_node_type = self._collect_nodes_info()
        for node_type in self.node_constraints_by_node_type:
            node_constraints = self.node_constraints_by_node_type[node_type]
            if node_type not in self.nodes_info_by_node_type:
                self._print_info_waiting_for(node_constraints, 0, "minimal")
                return True

            if node_constraints.quorum:
                quorum_id = self._get_running_quorum(node_type)
                if quorum_id:
                    if not node_constraints.scalable:
                        # not have dynamic join cases
                        continue
                    node_quorum_join = self._get_quorum_join_in_progress(node_type)
                    if node_quorum_join is None:
                        # There is running quorum and no join in progress, no need to wait
                        continue

                    # quorum join in progress, make sure the node has ip ready,
                    node_id, node_info = node_quorum_join
                    if node_info.get(RUNTIME_NODE_IP) is None:
                        logger.info("Cluster Controller: waiting for")
                        return True

                    # we publish the nodes info for quorum
                    self._publish_nodes_for_quorum(
                        node_type, quorum_id, node_constraints)
                    continue

            nodes_info = self.nodes_info_by_node_type[node_type]
            number_of_nodes = len(nodes_info)
            if node_constraints.minimal > number_of_nodes:
                self._print_info_waiting_for(node_constraints, number_of_nodes, "minimal")
                return True

            # Check whether the internal ip are all available
            for node_id, node_info in nodes_info.items():
                if node_info.get(RUNTIME_NODE_IP) is None:
                    self._print_info_waiting_for(node_constraints, number_of_nodes, "IP available")
                    return True

            logger.info(
                "Cluster Controller: Node constraints satisfied for {}: minimal nodes = {}.".format(
                    node_type, node_constraints.minimal))
            # publish nodes will check whether it has changed since last publish
            self._publish_nodes(node_type, nodes_info, node_constraints)

        # All satisfied if come to here
        return False

    def _publish_nodes(
            self, node_type: str, nodes_info, node_constraints,
            form_quorum=True, quorum_id=None):
        nodes_info_data = json.dumps(nodes_info, sort_keys=True)

        hasher = hashlib.sha1()
        hasher.update(nodes_info_data.encode("utf-8"))
        new_nodes_info_hash = hasher.hexdigest()

        published_nodes_info_hash = self.published_nodes_info_hashes.get(node_type)
        if published_nodes_info_hash and new_nodes_info_hash == published_nodes_info_hash:
            return False
        self.published_nodes_info_hashes[node_type] = new_nodes_info_hash

        if form_quorum:
            # Minimal number of the nodes reached, set the quorum id of the new joined nodes
            quorum_id = self._form_a_quorum(node_type, new_nodes_info_hash)

        if quorum_id:
            logger.info(
                "Cluster Controller: Publish and notify nodes for {} with quorum {}".format(
                    node_type, quorum_id))
        else:
            logger.info(
                "Cluster Controller: Publish and notify nodes for {}".format(
                    node_type))

        nodes_info_key = CLOUDTIK_CLUSTER_NODES_INFO_NODE_TYPE.format(node_type)
        kv_put(nodes_info_key, nodes_info_data, overwrite=True)

        # Notify runtime of these
        self._notify_node_constraints_reached(
            node_type, nodes_info, node_constraints,
            quorum_id=quorum_id)
        return True

    def _publish_nodes_for_quorum(
            self, node_type: str, quorum_id: str, node_constraints):
        nodes_info = self._get_nodes_info_for_quorum(node_type, quorum_id)
        if not nodes_info:
            # quorum has no nodes
            logger.warning(
                "The quorum {} based on node type has no nodes.".format(
                    quorum_id, node_type))
            return False

        return self._publish_nodes(
            node_type, nodes_info, node_constraints,
            form_quorum=False, quorum_id=quorum_id)

    def _get_nodes_info_for_quorum(self, node_type: str, quorum_id: str):
        quorum_id_to_nodes = self.quorum_id_to_nodes_by_node_type.get(
            node_type, {})
        quorum_nodes = quorum_id_to_nodes.get(quorum_id)
        if not quorum_nodes:
            return None

        nodes_info = self.nodes_info_by_node_type.get(node_type)
        if not nodes_info:
            return None

        # get nodes info by node id from nodes_info_of_node_type
        quorum_nodes_info = {}
        for node_id in quorum_nodes:
            node_info = nodes_info.get(node_id)
            if node_info:
                quorum_nodes_info[node_id] = node_info
        return quorum_nodes_info

    def _notify_node_constraints_reached(
            self, node_type: str, nodes_info, node_constraints,
            quorum_id: Optional[str] = None):
        head_id = self.non_terminated_nodes.head_id
        head_node_ip = self.provider.internal_ip(head_id)
        head_info = {
            RUNTIME_NODE_ID: head_id,
            RUNTIME_NODE_IP: head_node_ip,
            RUNTIME_NODE_SEQ_ID: CLOUDTIK_TAG_HEAD_NODE_SEQ_ID
        }
        _notify_node_constraints_reached(
            self.config, node_type, head_info, nodes_info,
            runtimes_to_notify=node_constraints.runtimes,
            quorum_id=quorum_id)

    def is_launch_allowed(self, node_type: str):
        if self._has_node_constraints(node_type) and (
                self._has_quorum_node_constraints(node_type)):
            running_quorum = self._get_running_quorum(node_type)
            if running_quorum:
                # if there is a running quorum
                if self._is_quorum_scalable(node_type):
                    if self._is_quorum_join_in_progress(node_type):
                        logger.info(
                            "Cluster Controller: Node in progress of quorum join. "
                            "Pause launching new nodes for type: {}.".format(
                                node_type))
                        return False, None
                    else:
                        return True, running_quorum
                else:
                    return False, None
        return True, None

    def _has_node_constraints(self, node_type: str):
        if not self.node_constraints_by_node_type:
            return False
        if node_type not in self.node_constraints_by_node_type:
            return False
        return True

    def _has_quorum_node_constraints(self, node_type: str):
        node_constraints = self.node_constraints_by_node_type[node_type]
        return node_constraints.quorum

    def _is_quorum_scalable(self, node_type: str):
        # Suggest to scale one node a time for a scalable quorum
        node_constraints = self.node_constraints_by_node_type[node_type]
        return node_constraints.scalable

    def _form_a_quorum(self, node_type: str, quorum_id):
        if not self._has_quorum_node_constraints(node_type):
            return None

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

        # get the running quorum
        running_quorum_id = self._get_running_quorum(node_type)
        if quorum_id != running_quorum_id:
            logger.warning(
                "The {} running quorum {} doesn't match to the target quorum {}.".format(
                    node_type, running_quorum_id, quorum_id))
        return running_quorum_id

    def _get_quorum(self, node_type: str):
        node_constraints = self.node_constraints_by_node_type[node_type]
        minimal = node_constraints.minimal
        quorum = int(minimal / 2) + 1
        return quorum, minimal

    def terminate_for_quorum(self, node_type: str, node_id):
        if (not self._has_node_constraints(node_type)) or (
                not self._has_quorum_node_constraints(node_type)):
            return False

        quorum, minimal = self._get_quorum(node_type)
        # Check whether the node is an invalid quorum member
        quorum_id_to_nodes = self.quorum_id_to_nodes_by_node_type.get(
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
        if node_type not in self.quorum_id_to_nodes_by_node_type:
            self.quorum_id_to_nodes_by_node_type[node_type] = {}
        quorum_id_to_nodes = self.quorum_id_to_nodes_by_node_type[node_type]
        if node_quorum_id not in quorum_id_to_nodes:
            quorum_id_to_nodes[node_quorum_id] = set()
        quorum_id_nodes = quorum_id_to_nodes[node_quorum_id]
        quorum_id_nodes.add(node_id)

    def _collect_quorum_nodes(self):
        if not self.node_constraints_by_node_type:
            # No need
            return

        self.quorum_id_to_nodes_by_node_type = {}
        for node_id in self.non_terminated_nodes.worker_ids:
            tags = self.provider.node_tags(node_id)
            node_type = tags.get(CLOUDTIK_TAG_USER_NODE_TYPE)
            if not node_type:
                continue
            if node_type not in self.node_constraints_by_node_type:
                continue
            node_quorum_id = tags.get(CLOUDTIK_TAG_QUORUM_ID)
            if not node_quorum_id:
                continue

            self._update_quorum_id_to_nodes(
                node_type, node_quorum_id, node_id)

    def _get_running_quorum(self, node_type: str):
        quorum, minimal = self._get_quorum(node_type)
        # Only when a quorum of the minimal nodes dead,
        # we can launch new nodes and form a new quorum
        quorum_id_to_nodes = self.quorum_id_to_nodes_by_node_type.get(
            node_type, {})
        for quorum_id in quorum_id_to_nodes:
            quorum_id_nodes = quorum_id_to_nodes[quorum_id]
            remaining = len(quorum_id_nodes)
            if remaining >= quorum:
                # One quorum id exceed the quorum
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("No new node launch allowed with the existence of a valid quorum: {} ({}/{}).".format(
                        quorum_id, remaining, minimal))
                return quorum_id

        # none of the quorum_id exceeding a quorum
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Cluster Controller: None of the quorum id of {} forms a quorum ({}/{})."
                " Quorum launch.".format(node_type, quorum, minimal))
        return None

    def _is_quorum_join_in_progress(self, node_type):
        node_quorum_join = self._get_quorum_join_in_progress(node_type)
        return False if node_quorum_join is None else True

    def _get_quorum_join_in_progress(self, node_type):
        # Make sure this call are after wait_for_update
        # because self.nodes_info_of_node_type is set at beginning
        if node_type not in self.nodes_info_by_node_type:
            return None

        nodes_info = self.nodes_info_by_node_type[node_type]
        # Check whether the node which is in progress
        for node_id, node_info in nodes_info.items():
            quorum_join = node_info.get(RUNTIME_NODE_QUORUM_JOIN)
            if quorum_join and quorum_join == QUORUM_JOIN_STATUS_INIT:
                return node_id, node_info
        return None

    @staticmethod
    def _print_info_waiting_for(node_constraints, number_of_nodes, for_what):
        logger.info("Cluster Controller: waiting for {} of {}/{} nodes required by runtimes: {}".format(
            for_what, number_of_nodes, node_constraints.minimal, node_constraints.runtimes))
