import json
import logging
import time

from cloudtik.core._private.constants import CLOUDTIK_HEARTBEAT_TIMEOUT_S, CLOUDTIK_SCALING_STATE_TIMEOUT_S, \
    CLOUDTIK_NODE_RESOURCE_STATE_TIMEOUT_S
from cloudtik.core._private.state.control_state import ControlState
from cloudtik.core._private.state.kv_store import kv_put, kv_get
from cloudtik.core._private.state.state_utils import NODE_STATE_NODE_ID, NODE_STATE_NODE_IP, NODE_STATE_HEARTBEAT_TIME
from cloudtik.core.scaling_policy import ScalingState

CLOUDTIK_AUTOSCALING_INSTRUCTIONS = "autoscaling_instructions"
STATE_FETCH_TIMEOUT = 60
RESOURCE_STATE_TABLE = "resource_state"

logger = logging.getLogger(__name__)


class NodeHeartbeatState:
    def __init__(self, node_id, node_ip, last_heartbeat_time):
        self.node_id = node_id
        self.node_ip = node_ip
        self.last_heartbeat_time = last_heartbeat_time


class ClusterHeartbeatState:
    def __init__(self):
        self.node_heartbeat_states = {}

    def add_heartbeat_state(self, node_id, node_heartbeat_state):
        self.node_heartbeat_states[node_id] = node_heartbeat_state


class ScalingStateClient:
    """Client to read resource information from Redis"""

    def __init__(self,
                 control_state: ControlState,
                 nums_reconnect_retry: int = 5):
        self._control_state = control_state
        self._nums_reconnect_retry = nums_reconnect_retry

    def get_cluster_heartbeat_state(self, timeout: int = STATE_FETCH_TIMEOUT):
        node_table = self._control_state.get_node_table()
        cluster_heartbeat_state = ClusterHeartbeatState()
        for node_info_as_json in node_table.get_all().values():
            node_info = json.loads(node_info_as_json)
            # Filter out the stale record in the node table
            delta = time.time() - node_info.get(NODE_STATE_HEARTBEAT_TIME, 0)
            if delta < CLOUDTIK_HEARTBEAT_TIMEOUT_S:
                node_id = node_info[NODE_STATE_NODE_ID]
                node_heartbeat_state = NodeHeartbeatState(
                    node_id, node_info[NODE_STATE_NODE_IP],
                    node_info.get(NODE_STATE_HEARTBEAT_TIME))
                cluster_heartbeat_state.add_heartbeat_state(node_id, node_heartbeat_state)
        return cluster_heartbeat_state

    def get_scaling_state(self, timeout: int = STATE_FETCH_TIMEOUT):
        now = time.time()
        scaling_state = ScalingState()

        # Get resource demands
        as_json = kv_get(CLOUDTIK_AUTOSCALING_INSTRUCTIONS)
        if as_json is not None:
            autoscaling_instructions = json.loads(as_json)
            scaling_time = autoscaling_instructions.get("scaling_time", 0)
            delta = now - scaling_time
            if delta < CLOUDTIK_SCALING_STATE_TIMEOUT_S:
                scaling_state.set_autoscaling_instructions(autoscaling_instructions)

        # Get resource state of nodes
        resource_state_table = self._control_state.get_user_state_table(RESOURCE_STATE_TABLE)
        for resource_state_as_json in resource_state_table.get_all().values():
            resource_state = json.loads(resource_state_as_json)
            # Filter out the stale record in the node table
            resource_time = resource_state.get("resource_time", 0)
            delta = now - resource_time
            if delta < CLOUDTIK_NODE_RESOURCE_STATE_TIMEOUT_S:
                node_id = resource_state[NODE_STATE_NODE_ID]
                scaling_state.add_node_resource_state(node_id, resource_state)
        return scaling_state

    def update_scaling_state(self, scaling_state: ScalingState):
        autoscaling_instructions = scaling_state.autoscaling_instructions
        if autoscaling_instructions is not None:
            as_json = json.dumps(autoscaling_instructions)
            kv_put(CLOUDTIK_AUTOSCALING_INSTRUCTIONS, as_json)

        node_resource_states = scaling_state.node_resource_states
        lost_nodes = scaling_state.lost_nodes
        if node_resource_states is not None or lost_nodes is not None:
            resource_state_table = self._control_state.get_user_state_table(RESOURCE_STATE_TABLE)
            if node_resource_states:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Publish node resource states for: {}".format(
                        [node_id for node_id in node_resource_states]))
                for node_id, node_resource_state in node_resource_states.items():
                    resource_state_as_json = json.dumps(node_resource_state)
                    resource_state_table.put(node_id, resource_state_as_json)
            if lost_nodes:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Delete node resource states for: {}".format(
                        [node_id for node_id in lost_nodes]))
                for node_id in lost_nodes:
                    resource_state_table.delete(node_id)

    @staticmethod
    def create_from(control_state):
        return ScalingStateClient(control_state=control_state)
