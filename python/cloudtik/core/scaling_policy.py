import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ScalingState:
    def __init__(self):
        self.node_resource_states = {}
        self.lost_nodes = {}
        self.autoscaling_instructions = None

    def add_node_resource_state(self, node_id, node_resource_state):
        self.node_resource_states[node_id] = node_resource_state

    def set_node_resource_states(self, node_resource_states):
        self.node_resource_states = node_resource_states

    def add_lost_node(self, node_id):
        self.node_resource_states[node_id] = node_id

    def set_lost_nodes(self, lost_nodes):
        self.lost_nodes = lost_nodes

    def set_autoscaling_instructions(self, autoscaling_instructions):
        self.autoscaling_instructions = autoscaling_instructions


class ScalingPolicy:
    """Interface for plugin automatically scale policy.

    **Important**: This is an INTERNAL API that is only exposed for the purpose
    of implementing custom scaling policies. It is not allowed to call into
    ScalingPolicy methods from any package outside, only to
    define new implementations of ScalingPolicy for use with the "external" scale
    policy option.
    """

    def __init__(self,
                 config: Dict[str, Any],
                 head_ip: str) -> None:
        self.config = config
        self.head_ip = head_ip

    def reset(self, config):
        self.config = config

    def get_scaling_state(self) -> Optional[ScalingState]:
        raise NotImplementedError
