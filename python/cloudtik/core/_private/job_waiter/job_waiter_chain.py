import logging
from typing import Any, Dict, Optional

from cloudtik.core.job_waiter import JobWaiter

logger = logging.getLogger(__name__)


class JobWaiterChain(JobWaiter):
    def __init__(self,
                 config: Dict[str, Any]) -> None:
        JobWaiter.__init__(self, config)
        self.job_waiters_in_chain = []

    def wait_for_completion(self, node_id: str, cmd: str, timestamp: int, timeout: Optional[int] = None):
        for job_waiter in self.job_waiters_in_chain:
            job_waiter.wait_for_completion(node_id, cmd, timestamp, timeout)

    def append_job_waiter(self, job_waiter: JobWaiter):
        self.job_waiters_in_chain.append(job_waiter)
