from typing import Optional, Any, Dict

import pytest

from cloudtik.core._private.job_waiter_factory import _parse_built_in_chain, create_job_waiter
from cloudtik.core.job_waiter import JobWaiter


class JobWaiterTest(JobWaiter):
    def __init__(self,
                 config: Dict[str, Any]) -> None:
        JobWaiter.__init__(self, config)
        self.job_waiters_in_chain = []

    def wait_for_completion(self, cmd: str, timeout: Optional[int] = None):
        pass


class TestJobWaiter:
    def test_parse_built_in_chain(self):
        job_waiter_name = "chain[a, b, c, d]"
        names_in_chain = _parse_built_in_chain(job_waiter_name)
        assert len(names_in_chain) == 4

        job_waiter_name = "abc[a, b, c, d]"
        names_in_chain = _parse_built_in_chain(job_waiter_name)
        assert names_in_chain is None

    def test_create_job_waiter(self):
        config = {}

        job_waiter_name = "cloudtik.tests.unit.core.test_job_waiter.JobWaiterTest"
        job_waiter = create_job_waiter(config, job_waiter_name)
        assert job_waiter is not None

        job_waiter_name = "chain[cloudtik.tests.unit.core.test_job_waiter.JobWaiterTest]"
        job_waiter_chain = create_job_waiter(config, job_waiter_name)
        assert job_waiter_chain is not None
        assert len(job_waiter_chain.job_waiters_in_chain) == 1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
