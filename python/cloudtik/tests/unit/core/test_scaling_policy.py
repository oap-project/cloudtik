import copy
import time
from typing import Dict, Optional, Any

import pytest

from cloudtik.core._private.cluster.resource_scaling_policy import ResourceScalingPolicy
from cloudtik.core._private.cluster.scaling_policies import ScalingWithTime
from cloudtik.core._private.utils import merge_scaling_state
from cloudtik.core.scaling_policy import ScalingState, ScalingPolicy

SCALING_POLICY_TEST_RUNTIME = "prometheus"

test_now = time.time()

autoscaling_instructions = {
    "scaling_time": test_now,
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

CONFIG = {
    "head_node_type": "head.default",
    "available_node_types": {
        "head.default": {
            "resources": {
                "CPU": 4
            }
        },
        "worker.default": {
            "min_workers": 3,
            "resources": {
                "CPU": 4
            }
        }
    }
}

SECONDS_OF_A_DAY = 3600 * 24


class ScalingPolicyForTest(ScalingPolicy):
    def __init__(self, config: Dict[str, Any], head_ip: str) -> None:
        super().__init__(config, head_ip)

    def name(self):
        return "scaling-for-test"

    def reset(self, config):
        super().reset(config)

    def get_scaling_state(self) -> Optional[ScalingState]:
        return ScalingState(autoscaling_instructions, node_resource_states, lost_nodes)


class ScalingWithTimeTest(ScalingWithTime):
    def __init__(self,
                 config: Dict[str, Any],
                 head_ip: str) -> None:
        ScalingWithTime.__init__(self, config, head_ip)

    def get_scaling_state(self) -> Optional[ScalingState]:
        self.last_state_time = time.time()

        all_node_metrics = {}
        _autoscaling_instructions = self._get_autoscaling_instructions(
            all_node_metrics)
        _node_resource_states = None

        scaling_state = ScalingState()
        scaling_state.set_autoscaling_instructions(_autoscaling_instructions)
        scaling_state.set_node_resource_states(_node_resource_states)
        scaling_state.set_lost_nodes(lost_nodes)
        return scaling_state


class TestScalingPolicy:

    def test_user_scaling_policy(self):
        resource_scaling_policy = ResourceScalingPolicy("127.0.0.1", None)
        config = {
            "runtime": {
                "types": [SCALING_POLICY_TEST_RUNTIME],
                "scaling": {
                    "scaling_policy_class": "cloudtik.tests.unit.core.test_scaling_policy.ScalingPolicyForTest"
                }
            },
        }
        resource_scaling_policy.reset(config)

        scaling_state = resource_scaling_policy.get_scaling_state()
        assert scaling_state is not None
        assert scaling_state.autoscaling_instructions["scaling_time"] == test_now
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
        assert result_scaling_sate.autoscaling_instructions["scaling_time"] == test_now
        assert len(result_scaling_sate.autoscaling_instructions["resource_demands"]) == 1
        assert result_scaling_sate.node_resource_states["node-1.1.1.1"]["available_resources"]["CPU"] == 0

    def test_scaling_with_time(self):
        config = copy.deepcopy(CONFIG)
        config["runtime"] = {
            "types": [SCALING_POLICY_TEST_RUNTIME],
            "scaling": {
                "scaling_policy": "scale-with-time",
                "scaling_math_base": "on-previous-time",
                "scaling_time_table": {
                    "00:00:01": 2,
                    "00:00:02": 3,
                    "00:00:03": "*2",
                    "00:00:04": "*3.0"
                }
            }
        }

        scaling_with_time = ScalingWithTimeTest(config, "127.0.0.1")
        result_scaling_sate = scaling_with_time.get_scaling_state()

        assert result_scaling_sate is not None
        assert scaling_with_time.min_workers == 3
        assert scaling_with_time.scaling_time_table[0][0] == 1
        assert scaling_with_time.scaling_time_table[1][0] == 2
        assert scaling_with_time.scaling_time_table[2][0] == 3
        assert scaling_with_time.scaling_time_table[3][0] == 4
        assert scaling_with_time.scaling_time_table[0][1] == 2
        assert scaling_with_time.scaling_time_table[1][1] == 3
        assert scaling_with_time.scaling_time_table[2][1] == 6
        assert scaling_with_time.scaling_time_table[3][1] == 18

    def test_scaling_with_time_reorder(self):
        config = copy.deepcopy(CONFIG)
        config["runtime"] = {
            "types": [SCALING_POLICY_TEST_RUNTIME],
            "scaling": {
                "scaling_policy": "scale-with-time",
                "scaling_math_base": "on-previous-time",
                "scaling_time_table": {
                    "00:00:04": "*3.0",
                    "00:00:03": "*2",
                    "00:00:02": 3,
                    "00:00:01": 2,
                }
            }
        }

        scaling_with_time = ScalingWithTimeTest(config, "127.0.0.1")
        result_scaling_sate = scaling_with_time.get_scaling_state()

        assert result_scaling_sate is not None
        assert scaling_with_time.min_workers == 3
        assert scaling_with_time.scaling_time_table[0][0] == 1
        assert scaling_with_time.scaling_time_table[1][0] == 2
        assert scaling_with_time.scaling_time_table[2][0] == 3
        assert scaling_with_time.scaling_time_table[3][0] == 4
        assert scaling_with_time.scaling_time_table[0][1] == 2
        assert scaling_with_time.scaling_time_table[1][1] == 3
        assert scaling_with_time.scaling_time_table[2][1] == 6
        assert scaling_with_time.scaling_time_table[3][1] == 18

    def test_scaling_with_time_expanding(self):
        config = copy.deepcopy(CONFIG)
        config["runtime"] = {
            "types": [SCALING_POLICY_TEST_RUNTIME],
            "scaling": {
                "scaling_policy": "scale-with-time",
                "scaling_math_base": "on-previous-time",
                "scaling_time_table": {
                    "00:00:01": "*2",
                    "00:00:02": 0,
                    "00:00:03": "*3.0",
                    "00:00:04": "3"
                }
            }
        }

        scaling_with_time = ScalingWithTimeTest(config, "127.0.0.1")
        result_scaling_sate = scaling_with_time.get_scaling_state()

        assert result_scaling_sate is not None
        assert scaling_with_time.min_workers == 3
        assert scaling_with_time.scaling_time_table[0][0] == 1
        assert scaling_with_time.scaling_time_table[1][0] == 2
        assert scaling_with_time.scaling_time_table[2][0] == 3
        assert scaling_with_time.scaling_time_table[3][0] == 4
        assert scaling_with_time.scaling_time_table[0][1] == 6
        assert scaling_with_time.scaling_time_table[1][1] == 3
        assert scaling_with_time.scaling_time_table[2][1] == 9
        assert scaling_with_time.scaling_time_table[3][1] == 3

    def test_scaling_with_time_base_on_min_workers(self):
        config = copy.deepcopy(CONFIG)
        config["runtime"] = {
            "types": [SCALING_POLICY_TEST_RUNTIME],
            "scaling": {
                "scaling_policy": "scale-with-time",
                "scaling_math_base": "on-min-workers",
                "scaling_time_table": {
                    "00:00:01": "*2",
                    "00:00:02": "+2",
                    "00:00:03": "*3.0",
                    "00:00:04": "-1"
                }
            }
        }

        scaling_with_time = ScalingWithTimeTest(config, "127.0.0.1")
        result_scaling_sate = scaling_with_time.get_scaling_state()

        assert result_scaling_sate is not None
        assert scaling_with_time.min_workers == 3
        assert scaling_with_time.scaling_time_table[0][0] == 1
        assert scaling_with_time.scaling_time_table[1][0] == 2
        assert scaling_with_time.scaling_time_table[2][0] == 3
        assert scaling_with_time.scaling_time_table[3][0] == 4
        assert scaling_with_time.scaling_time_table[0][1] == 6
        assert scaling_with_time.scaling_time_table[1][1] == 5
        assert scaling_with_time.scaling_time_table[2][1] == 9
        assert scaling_with_time.scaling_time_table[3][1] == 2

    def test_scaling_with_time_nodes_request(self):
        config = copy.deepcopy(CONFIG)
        config["runtime"] = {
            "types": [SCALING_POLICY_TEST_RUNTIME],
            "scaling": {
                "scaling_policy": "scale-with-time",
                "scaling_math_base": "on-previous-time",
                "scaling_time_table": {
                    "00:00:03": "*2",
                    "00:00:06": 0,
                    "00:00:09": "*3.0",
                    "00:00:12": "3"
                }
            }
        }

        scaling_with_time = ScalingWithTimeTest(config, "127.0.0.1")
        result_scaling_sate = scaling_with_time.get_scaling_state()

        assert result_scaling_sate is not None
        assert scaling_with_time.min_workers == 3
        assert scaling_with_time.scaling_time_table[0][0] == 3
        assert scaling_with_time.scaling_time_table[1][0] == 6
        assert scaling_with_time.scaling_time_table[2][0] == 9
        assert scaling_with_time.scaling_time_table[3][0] == 12
        assert scaling_with_time.scaling_time_table[0][1] == 6
        assert scaling_with_time.scaling_time_table[1][1] == 3
        assert scaling_with_time.scaling_time_table[2][1] == 9
        assert scaling_with_time.scaling_time_table[3][1] == 3

        assert scaling_with_time._get_nodes_request(1) == 3
        assert scaling_with_time._get_nodes_request(2) == 3
        assert scaling_with_time._get_nodes_request(3) == 6
        assert scaling_with_time._get_nodes_request(4) == 6
        assert scaling_with_time._get_nodes_request(6) == 3
        assert scaling_with_time._get_nodes_request(7) == 3
        assert scaling_with_time._get_nodes_request(9) == 9
        assert scaling_with_time._get_nodes_request(10) == 9
        assert scaling_with_time._get_nodes_request(12) == 3
        assert scaling_with_time._get_nodes_request(13) == 3

        resource_requests = scaling_with_time._get_resource_requests_at_seconds(10)
        assert resource_requests is not None
        assert len(resource_requests) == 9 + 1

    def test_scaling_with_time_weekly(self):
        config = copy.deepcopy(CONFIG)
        config["runtime"] = {
            "types": [SCALING_POLICY_TEST_RUNTIME],
            "scaling": {
                "scaling_policy": "scale-with-time",
                "scaling_math_base": "on-min-workers",
                "scaling_periodic": "weekly",
                "scaling_time_table": {
                    "Mon 00:00:01": "*2",
                    "Tue 00:00:02": "+2",
                    "Wed 00:00:03": "*3.0",
                    "Thu 00:00:04": "-1"
                }
            }
        }

        scaling_with_time = ScalingWithTimeTest(config, "127.0.0.1")
        result_scaling_sate = scaling_with_time.get_scaling_state()
        assert result_scaling_sate is not None
        assert scaling_with_time.min_workers == 3
        assert scaling_with_time.scaling_time_table[0][0] == 1 + 0 * SECONDS_OF_A_DAY
        assert scaling_with_time.scaling_time_table[1][0] == 2 + 1 * SECONDS_OF_A_DAY
        assert scaling_with_time.scaling_time_table[2][0] == 3 + 2 * SECONDS_OF_A_DAY
        assert scaling_with_time.scaling_time_table[3][0] == 4 + 3 * SECONDS_OF_A_DAY
        assert scaling_with_time.scaling_time_table[0][1] == 6
        assert scaling_with_time.scaling_time_table[1][1] == 5
        assert scaling_with_time.scaling_time_table[2][1] == 9
        assert scaling_with_time.scaling_time_table[3][1] == 2

    def test_scaling_with_time_monthly(self):
        config = copy.deepcopy(CONFIG)
        config["runtime"] = {
            "types": [SCALING_POLICY_TEST_RUNTIME],
            "scaling": {
                "scaling_policy": "scale-with-time",
                "scaling_math_base": "on-min-workers",
                "scaling_periodic": "monthly",
                "scaling_time_table": {
                    "20 00:00:01": "*2",
                    "21 00:00:02": "+2",
                    "22 00:00:03": "*3.0",
                    "23 00:00:04": "-1"
                }
            }
        }

        scaling_with_time = ScalingWithTimeTest(config, "127.0.0.1")
        result_scaling_sate = scaling_with_time.get_scaling_state()
        assert result_scaling_sate is not None
        assert scaling_with_time.min_workers == 3
        assert scaling_with_time.scaling_time_table[0][0] == 1 + 19 * SECONDS_OF_A_DAY
        assert scaling_with_time.scaling_time_table[1][0] == 2 + 20 * SECONDS_OF_A_DAY
        assert scaling_with_time.scaling_time_table[2][0] == 3 + 21 * SECONDS_OF_A_DAY
        assert scaling_with_time.scaling_time_table[3][0] == 4 + 22 * SECONDS_OF_A_DAY
        assert scaling_with_time.scaling_time_table[0][1] == 6
        assert scaling_with_time.scaling_time_table[1][1] == 5
        assert scaling_with_time.scaling_time_table[2][1] == 9
        assert scaling_with_time.scaling_time_table[3][1] == 2

    def test_scaling_with_time_nodes_request_weekly(self):
        config = copy.deepcopy(CONFIG)
        config["runtime"] = {
            "types": [SCALING_POLICY_TEST_RUNTIME],
            "scaling": {
                "scaling_policy": "scale-with-time",
                "scaling_math_base": "on-previous-time",
                "scaling_periodic": "weekly",
                "scaling_time_table": {
                    "Mon 00:00:03": "*2",
                    "Tue 00:00:06": 0,
                    "Wed 00:00:09": "*3.0",
                    "Thu 00:00:12": "3"
                }
            }
        }

        scaling_with_time = ScalingWithTimeTest(config, "127.0.0.1")
        result_scaling_sate = scaling_with_time.get_scaling_state()

        assert result_scaling_sate is not None
        assert scaling_with_time.min_workers == 3
        assert scaling_with_time.scaling_time_table[0][0] == 3 + 0 * SECONDS_OF_A_DAY
        assert scaling_with_time.scaling_time_table[1][0] == 6 + 1 * SECONDS_OF_A_DAY
        assert scaling_with_time.scaling_time_table[2][0] == 9 + 2 * SECONDS_OF_A_DAY
        assert scaling_with_time.scaling_time_table[3][0] == 12 + 3 * SECONDS_OF_A_DAY
        assert scaling_with_time.scaling_time_table[0][1] == 6
        assert scaling_with_time.scaling_time_table[1][1] == 3
        assert scaling_with_time.scaling_time_table[2][1] == 9
        assert scaling_with_time.scaling_time_table[3][1] == 3

        resource_requests = scaling_with_time._get_resource_requests_at_seconds(
            10 + 2 * SECONDS_OF_A_DAY)
        assert resource_requests is not None
        assert len(resource_requests) == 9 + 1

    def test_scaling_with_time_nodes_request_monthly(self):
        config = copy.deepcopy(CONFIG)
        config["runtime"] = {
            "types": [SCALING_POLICY_TEST_RUNTIME],
            "scaling": {
                "scaling_policy": "scale-with-time",
                "scaling_math_base": "on-previous-time",
                "scaling_periodic": "monthly",
                "scaling_time_table": {
                    "21 00:00:03": "*2",
                    "22 00:00:06": 0,
                    "23 00:00:09": "*3.0",
                    "24 00:00:12": "3"
                }
            }
        }

        scaling_with_time = ScalingWithTimeTest(config, "127.0.0.1")
        result_scaling_sate = scaling_with_time.get_scaling_state()

        assert result_scaling_sate is not None
        assert scaling_with_time.min_workers == 3
        assert scaling_with_time.scaling_time_table[0][0] == 3 + 20 * SECONDS_OF_A_DAY
        assert scaling_with_time.scaling_time_table[1][0] == 6 + 21 * SECONDS_OF_A_DAY
        assert scaling_with_time.scaling_time_table[2][0] == 9 + 22 * SECONDS_OF_A_DAY
        assert scaling_with_time.scaling_time_table[3][0] == 12 + 23 * SECONDS_OF_A_DAY
        assert scaling_with_time.scaling_time_table[0][1] == 6
        assert scaling_with_time.scaling_time_table[1][1] == 3
        assert scaling_with_time.scaling_time_table[2][1] == 9
        assert scaling_with_time.scaling_time_table[3][1] == 3

        resource_requests = scaling_with_time._get_resource_requests_at_seconds(
            10 + 22 * SECONDS_OF_A_DAY)
        assert resource_requests is not None
        assert len(resource_requests) == 9 + 1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
