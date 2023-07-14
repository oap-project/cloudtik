import logging
from typing import Any, Dict, Optional

from cloudtik.core._private.annotations import DeveloperAPI

logger = logging.getLogger(__name__)


@DeveloperAPI
class ScalingState:
    def __init__(self,
                 autoscaling_instructions=None,
                 node_resource_states=None,
                 lost_nodes=None):
        self.autoscaling_instructions = autoscaling_instructions
        self.node_resource_states = node_resource_states
        self.lost_nodes = lost_nodes

    def set_autoscaling_instructions(self, autoscaling_instructions):
        self.autoscaling_instructions = autoscaling_instructions

    def add_node_resource_state(self, node_id, node_resource_state):
        if self.node_resource_states is None:
            self.node_resource_states = {}
        self.node_resource_states[node_id] = node_resource_state

    def set_node_resource_states(self, node_resource_states):
        self.node_resource_states = node_resource_states

    def add_lost_node(self, node_id):
        if self.lost_nodes is None:
            self.lost_nodes = {}
        self.lost_nodes[node_id] = node_id

    def set_lost_nodes(self, lost_nodes):
        self.lost_nodes = lost_nodes


@DeveloperAPI
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

    def name(self):
        """Return the name of the scaling policy"""
        raise NotImplementedError

    def reset(self, config):
        self.config = config

    def get_scaling_state(self) -> Optional[ScalingState]:
        """Return the scaling state including scaling instructions and resource states"""
        raise NotImplementedError
