import logging
from typing import Optional

from cloudtik.core._private.state.scaling_state import ScalingStateClient, ScalingState
from cloudtik.core._private.utils import _get_runtime_scaling_policy

logger = logging.getLogger(__name__)


class ResourceScalingPolicy:
    def __init__(self,
                 head_ip,
                 scaling_state_client: ScalingStateClient):
        self.head_ip = head_ip
        self.scaling_state_client = scaling_state_client
        self.config = None
        self.scaling_policy = None

    def reset(self, config):
        self.config = config
        # Reset is called when the configuration changed
        # Always recreate the scaling policy when config is changed
        # in the case that the scaling policy is disabled in the change
        self.scaling_policy = self._create_scaling_policy(self.config)

    def has_scaling_policy(self):
        return False if self.scaling_policy is None else True

    def _create_scaling_policy(self, config):
        return _get_runtime_scaling_policy(config, self.head_ip)

    def update(self):
        # Pulling data from resource management system
        scaling_state = self.get_scaling_state()
        if scaling_state is not None:
            self.scaling_state_client.update_scaling_state(
                scaling_state)

    def get_scaling_state(self) -> Optional[ScalingState]:
        if self.scaling_policy is None:
            return None
        return self.scaling_policy.get_scaling_state()
