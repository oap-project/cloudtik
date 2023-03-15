import logging
from typing import Optional

from cloudtik.core._private.cluster.scaling_policies import _create_scaling_policy
from cloudtik.core._private.state.scaling_state import ScalingStateClient, ScalingState
from cloudtik.core._private.utils import merge_scaling_state, RUNTIME_CONFIG_KEY, \
    _get_runtime_scaling_policy

logger = logging.getLogger(__name__)


class ResourceScalingPolicy:
    def __init__(self,
                 head_ip,
                 scaling_state_client: ScalingStateClient):
        self.head_ip = head_ip
        self.scaling_state_client = scaling_state_client
        self.config = None
        # Multiple scaling policies will cause confusion
        self.scaling_policy = None

    def reset(self, config):
        self.config = config
        # Reset is called when the configuration changed
        # Always recreate the scaling policy when config is changed
        # in the case that the scaling policy is disabled in the change
        self.scaling_policy = self._create_scaling_policies(self.config)
        if self.scaling_policy is not None:
            logger.info(f"CloudTik scaling with: {self.scaling_policy.name()}")
        else:
            logger.info("CloudTik: No scaling policy is used.")

    def _create_scaling_policies(self, config):
        scaling_policy = _get_runtime_scaling_policy(config, self.head_ip)
        if scaling_policy is not None:
            return scaling_policy

        # Check whether there are any built-in scaling policies configured
        system_scaling_policy = self._get_system_scaling_policy(config, self.head_ip)
        if system_scaling_policy is not None:
            return system_scaling_policy

        return None

    def update(self):
        # Pulling data from resource management system
        scaling_state = self.get_scaling_state()
        if scaling_state is not None:
            self.scaling_state_client.update_scaling_state(
                scaling_state)

    def get_scaling_state(self) -> Optional[ScalingState]:
        if self.scaling_policies is None:
            return None

        return self.scaling_policy.get_scaling_state()

    def _get_system_scaling_policy(self, config, head_ip):
        runtime_config = config.get(RUNTIME_CONFIG_KEY)
        if runtime_config is None:
            return None

        if "scaling" not in runtime_config:
            return None

        scaling_config = runtime_config["scaling"]
        scaling_policy_name = scaling_config.get("scaling_policy")
        if not scaling_policy_name:
            return None

        return _create_scaling_policy(scaling_policy_name, config, head_ip)
