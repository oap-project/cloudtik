import logging
from typing import Optional

from cloudtik.core._private.state.scaling_state import ScalingStateClient, ScalingState
from cloudtik.core._private.utils import _get_runtime_scaling_policies, merge_scaling_state

logger = logging.getLogger(__name__)


class ResourceScalingPolicy:
    def __init__(self,
                 head_ip,
                 scaling_state_client: ScalingStateClient):
        self.head_ip = head_ip
        self.scaling_state_client = scaling_state_client
        self.config = None
        # We support multiple scaling policies
        self.scaling_policies = None

    def reset(self, config):
        self.config = config
        # Reset is called when the configuration changed
        # Always recreate the scaling policy when config is changed
        # in the case that the scaling policy is disabled in the change
        self.scaling_policies = self._create_scaling_policies(self.config)

    def _create_scaling_policies(self, config):
        return _get_runtime_scaling_policies(config, self.head_ip)

    def update(self):
        # Pulling data from resource management system
        scaling_state = self.get_scaling_state()
        if scaling_state is not None:
            self.scaling_state_client.update_scaling_state(
                scaling_state)

    def get_scaling_state(self) -> Optional[ScalingState]:
        if self.scaling_policies is None:
            return None

        num_policies = len(self.scaling_policies)
        if num_policies == 0:
            return None
        elif num_policies == 1:
            return self.scaling_policies[0].get_scaling_state()
        else:
            # We each scaling policies, we call to get scaling state
            # We merge the scaling state from each of the scaling policy
            scaling_state = self.scaling_policies[0].get_scaling_state()
            for i in range(1, num_policies):
                new_scaling_state = self.scaling_policies[i].get_scaling_state()
                # TODO: currently we use a simple override method for scaling state merge
                scaling_state = merge_scaling_state(scaling_state, new_scaling_state)
            return scaling_state
