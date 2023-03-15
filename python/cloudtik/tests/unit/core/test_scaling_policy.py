import copy
import time
from typing import Dict, Optional, Any

import pytest

from cloudtik.core._private.cluster.resource_scaling_policy import ResourceScalingPolicy
from cloudtik.core._private.utils import merge_scaling_state
from cloudtik.core.scaling_policy import ScalingState, ScalingPolicy

test_now = time.time()

autoscaling_instructions = {
    "demanding_time": test_now,
    "resource_demands": [
        {"CPU": 4},
        {"CPU": 4}
    ]
}

node_resource_states = {
    "node-1.1.1.1": {
        "node_id": "node-1.1.1.1",
        "node_ip": "1.1.1.1",
        "resource_time": test_now,
        "total_resources": {
            "CPU": 4
        },
        "available_resources": {
            "CPU": 2
        },
        "resource_load": {
            "in_use": True
        }
    }
}

lost_nodes = {
    "node-1.1.1.2": "1.1.1.2"
}


class ScalingPolicyForTest(ScalingPolicy):
    def __init__(self,
                 config: Dict[str, Any],
                 head_ip: str) -> None:
        self.config = config
        self.head_ip = head_ip

    def name(self):
        return "scaling-for-test"

    def reset(self, config):
        self.config = config

    def get_scaling_state(self) -> Optional[ScalingState]:
        return ScalingState(autoscaling_instructions, node_resource_states, lost_nodes)


class TestScalingPolicy:

    def test_user_scaling_policy(self):
        resource_scaling_policy = ResourceScalingPolicy("127.0.0.1", None)
        config = {
            "runtime": {
                "types": ["ganglia"],
                "scaling": {
                    "scaling_policy_class": "cloudtik.tests.unit.core.test_scaling_policy.ScalingPolicyForTest"
                }
            },
        }
        resource_scaling_policy.reset(config)

        scaling_state = resource_scaling_policy.get_scaling_state()
        assert scaling_state is not None
        assert scaling_state.autoscaling_instructions["demanding_time"] == test_now
        assert len(scaling_state.autoscaling_instructions["resource_demands"]) == 2
        assert len(scaling_state.node_resource_states) == 1
        assert len(scaling_state.lost_nodes) == 1

    def test_scaling_state_merge(self):
        scaling_sate = ScalingState(autoscaling_instructions, node_resource_states, lost_nodes)
        autoscaling_instructions_copy = copy.deepcopy(autoscaling_instructions)
        autoscaling_instructions_copy["resource_demands"] = [
            {"CPU": 4},
        ]
        node_resource_states_copy = copy.deepcopy(node_resource_states)
        node_resource_states_copy["node-1.1.1.1"]["available_resources"]["CPU"] = 0
        new_scaling_state = ScalingState(autoscaling_instructions_copy, node_resource_states_copy, lost_nodes)

        result_scaling_sate = merge_scaling_state(scaling_sate, new_scaling_state)
        assert result_scaling_sate is not None
        assert result_scaling_sate.autoscaling_instructions["demanding_time"] == test_now
        assert len(result_scaling_sate.autoscaling_instructions["resource_demands"]) == 1
        assert result_scaling_sate.node_resource_states["node-1.1.1.1"]["available_resources"]["CPU"] == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
