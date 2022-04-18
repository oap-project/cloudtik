import pytest

from cloudtik.providers._private.aws.utils import resource_cache, BOTO_MAX_RETRIES

from botocore.stub import Stubber


@pytest.fixture()
def iam_client_stub():
    resource = resource_cache("iam", "us-west-2")
    with Stubber(resource.meta.client) as stubber:
        yield stubber
        stubber.assert_no_pending_responses()


@pytest.fixture()
def ec2_client_stub():
    resource = resource_cache("ec2", "us-west-2")
    with Stubber(resource.meta.client) as stubber:
        yield stubber
        stubber.assert_no_pending_responses()


@pytest.fixture()
def ec2_client_stub_fail_fast():
    resource = resource_cache("ec2", "us-west-2", 0)
    with Stubber(resource.meta.client) as stubber:
        yield stubber
        stubber.assert_no_pending_responses()


@pytest.fixture()
def ec2_client_stub_max_retries():
    resource = resource_cache("ec2", "us-west-2", BOTO_MAX_RETRIES)
    with Stubber(resource.meta.client) as stubber:
        yield stubber
        stubber.assert_no_pending_responses()
