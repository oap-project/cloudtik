import logging
from typing import Any, Dict, Optional

from cloudtik.core._private.annotations import DeveloperAPI

logger = logging.getLogger(__name__)


@DeveloperAPI
class JobWaiter:
    """Interface for plugin the method to check an async job completion.

    **Important**: This is an INTERNAL API that is only exposed for the purpose
    of implementing custom method of job completion check. It is not allowed to call into
    JobWaiter methods from any package outside, only to
    define new implementations of JobWaiter for use with submit and exec async jobs
    and stop cluster when the job finishes.
    """

    def __init__(self,
                 config: Dict[str, Any]) -> None:
        self.config = config

    def wait_for_completion(self, node_id: str, cmd: str, session_name: str, timeout: Optional[int] = None):
        raise NotImplementedError
