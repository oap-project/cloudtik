from typing import List

import pytest

from cloudtik.providers._private.gcp.node import GCPCompute
from cloudtik.providers._private.gcp.node_provider import _retry

_PROJECT_NAME = "project-one"
_AZ = "us-west1-b"


class MockGCPNodeProvider:
    def __init__(self, errors: List[type]):
        # List off errors to raise while retrying self.mock_method
        self.errors = errors
        # Incremented during each retry via self._construct_client
        self.error_index = -1
        # Mirrors the __init__ of GCPNodeProvider
        # Also called during each retry in _retry
        self._construct_clients()

    def _construct_clients(self):
        # In real life, called during each retry to reinitializes api clients.
        # Here, increments index in list of errors passed into test.
        self.error_index += 1

    @_retry
    def mock_method(self, *args, **kwargs):
        error = self.errors[self.error_index]
        if error:
            raise error
        return (args, kwargs)


# Short names for two types of errors
B, V = BrokenPipeError, ValueError


# BrokenPipeError is supposed to caught with up to 5 tries.
# ValueError is an arbitrarily chosen exception which should not be caught.


@pytest.mark.parametrize(
    "error_input,expected_error_raised",
    [
        ([None], None),
        ([B, B, B, B, None], None),
        ([B, B, V, B, None], V),
        ([B, B, B, B, B, None], B),
        ([B, B, B, B, B, B, None], B),
    ],
)
def test_gcp_broken_pipe_retry(error_input, expected_error_raised):
    """Tests retries of BrokenPipeError in GCPNodeProvider.

    Args:
        error_input: List of exceptions hit during retries of test mock_method.
            None means no exception.
        expected_error_raised: Expected exception raised.
            None means no exception.
    """
    provider = MockGCPNodeProvider(error_input)
    if expected_error_raised:
        with pytest.raises(expected_error_raised):
            provider.mock_method(1, 2, a=4, b=5)
    else:
        ret = provider.mock_method(1, 2, a=4, b=5)
        assert ret == ((1, 2), {"a": 4, "b": 5})


@pytest.mark.parametrize(
    "test_case", [("n1-standard-4", f"zones/{_AZ}/machineTypes/n1-standard-4"),
                  (f"zones/{_AZ}/machineTypes/n1-standard-4",
                   f"zones/{_AZ}/machineTypes/n1-standard-4")])
def test_convert_resources_to_urls_machine(test_case):
    gcp_compute = GCPCompute(None, _PROJECT_NAME, _AZ, "cluster_name")
    base_machine, result_machine = test_case
    modified_config = gcp_compute._convert_resources_to_urls({
        "machineType": base_machine
    })
    assert modified_config["machineType"] == result_machine


@pytest.mark.parametrize("test_case", [
    ("nvidia-tesla-k80",
     f"projects/{_PROJECT_NAME}/zones/{_AZ}/acceleratorTypes/nvidia-tesla-k80"
     ),
    (f"projects/{_PROJECT_NAME}/zones/{_AZ}/acceleratorTypes/nvidia-tesla-k80",
     f"projects/{_PROJECT_NAME}/zones/{_AZ}/acceleratorTypes/nvidia-tesla-k80"
     ),
])
def test_convert_resources_to_urls_accelerators(test_case):
    gcp_compute = GCPCompute(None, _PROJECT_NAME, _AZ, "cluster_name")
    base_accel, result_accel = test_case

    base_config = {
        "machineType": "n1-standard-4",
        "guestAccelerators": [{
            "acceleratorCount": 1,
            "acceleratorType": base_accel
        }]
    }
    modified_config = gcp_compute._convert_resources_to_urls(base_config)

    assert modified_config["guestAccelerators"][0][
               "acceleratorType"] == result_accel


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
