import json
import logging
import time

from cloudtik.core._private.constants import CLOUDTIK_HEARTBEAT_TIMEOUT_S
from cloudtik.core._private.state.control_state import ControlState
from cloudtik.core._private.state.kv_store import kv_put, kv_get

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


class ClusterResourceState:
    def __init__(self):
        self.node_resource_states = {}
        self.autoscaling_instructions = None

    def add_node_resource_state(self, node_id, node_resource_state):
        self.node_resource_states[node_id] = node_resource_state

    def set_node_resource_states(self, node_resource_states):
        self.node_resource_states = node_resource_states

    def set_autoscaling_instructions(self, autoscaling_instructions):
        self.autoscaling_instructions = autoscaling_instructions


class ResourceStateClient:
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
            delta = time.time() - node_info.get("last_heartbeat_time", 0)
            if delta < CLOUDTIK_HEARTBEAT_TIMEOUT_S:
                node_id = node_info["node_id"]
                node_heartbeat_state = NodeHeartbeatState(
                    node_id, node_info["node_ip"], node_info.get("last_heartbeat_time"))
                cluster_heartbeat_state.add_heartbeat_state(node_id, node_heartbeat_state)
        return cluster_heartbeat_state

    def get_cluster_resource_state(self, timeout: int = STATE_FETCH_TIMEOUT):
        cluster_resource_state = ClusterResourceState()

        # Get resource demands
        as_json = kv_get(CLOUDTIK_AUTOSCALING_INSTRUCTIONS)
        if as_json is not None:
            autoscaling_instructions = json.loads(as_json)
            cluster_resource_state.set_autoscaling_instructions(autoscaling_instructions)

        # Get resource state of nodes
        resource_state_table = self._control_state.get_user_state_table(RESOURCE_STATE_TABLE)
        for resource_state_as_json in resource_state_table.get_all().values():
            resource_state = json.loads(resource_state_as_json)
            # Filter out the stale record in the node table
            resource_time = resource_state.get("resource_time", 0)
            delta = time.time() - resource_time
            if delta < CLOUDTIK_HEARTBEAT_TIMEOUT_S:
                node_id = resource_state["node_id"]
                cluster_resource_state.add_node_resource_state(node_id, resource_state)
        return cluster_resource_state

    def update_cluster_resource_state(self, cluster_resource_state: ClusterResourceState):
        autoscaling_instructions = cluster_resource_state.autoscaling_instructions
        if autoscaling_instructions is not None:
            as_json = json.dumps(autoscaling_instructions)
            kv_put(CLOUDTIK_AUTOSCALING_INSTRUCTIONS, as_json)

        node_resource_states = cluster_resource_state.node_resource_states
        if node_resource_states is not None:
            resource_state_table = self._control_state.get_user_state_table(RESOURCE_STATE_TABLE)
            for node_id, node_resource_state in node_resource_states.items():
                resource_state_as_json = json.dumps(node_resource_state)
                resource_state_table.put(node_id, resource_state_as_json)

    @staticmethod
    def create_from(control_state):
        return ResourceStateClient(control_state=control_state)
