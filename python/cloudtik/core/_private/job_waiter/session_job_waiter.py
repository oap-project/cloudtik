import logging
import os
import time
from typing import Any, Dict, Optional

from cloudtik.core._private.call_context import CallContext
from cloudtik.core._private.cli_logger import cli_logger
from cloudtik.core._private.cluster.cluster_utils import run_on_node
from cloudtik.core._private.constants import CLOUDTIK_WAIT_FOR_JOB_FINISHED_INTERVAL_S, CLOUDTIK_JOB_WAITER_TIMEOUT_MAX
from cloudtik.core._private.utils import get_command_session_name
from cloudtik.core.job_waiter import JobWaiter

logger = logging.getLogger(__name__)

TMUX_SESSION_CHECK_SCRIPT = "core/_private/job_waiter/tmux-session.sh"
SCREEN_SESSION_CHECK_SCRIPT = "core/_private/job_waiter/screen-session.sh"

CHECK_SESSION_RETRY = 5
CHECK_SESSION_RETRY_DELAY_S = 5


class SessionJobWaiter(JobWaiter):
    def __init__(self,
                 config: Dict[str, Any], session_check_script: str) -> None:
        JobWaiter.__init__(self, config)
        self.session_check_script = session_check_script
        self.call_context = CallContext()
        self.call_context.set_call_from_api(True)

    def _check_session(self, node_id: str, session_name):
        cmd = "cloudtik run-script "
        cmd += self.session_check_script
        cmd += " "
        cmd += session_name

        retry = CHECK_SESSION_RETRY
        while retry > 0:
            try:
                output = run_on_node(
                    config=self.config,
                    call_context=self.call_context,
                    node_id=node_id,
                    cmd=cmd,
                    with_output=True)
                if output is not None:
                    if output.startswith(session_name + " found."):
                        return True
                return False
            except Exception as e:
                retry = retry - 1
                if retry > 0:
                    cli_logger.warning(f"Error when checking session. Retrying in {CHECK_SESSION_RETRY_DELAY_S} seconds.")
                    time.sleep(CHECK_SESSION_RETRY_DELAY_S)
                else:
                    cli_logger.error("Failed to request yarn api: {}", str(e))
                    raise e
        return False

    def wait_for_completion(self, node_id: str, cmd: str, timeout: Optional[int] = None):
        start_time = time.time()
        if timeout is None:
            timeout = CLOUDTIK_JOB_WAITER_TIMEOUT_MAX
        interval = CLOUDTIK_WAIT_FOR_JOB_FINISHED_INTERVAL_S

        session_name = get_command_session_name(cmd)
        session_exists = self._check_session(node_id, session_name)
        while time.time() - start_time < timeout:
            if not session_exists:
                cli_logger.print("Session {} finished.", session_name)
                return
            else:
                cli_logger.print(
                    "Waiting for session {} to finish: ({} seconds)...".format(
                        session_name,
                        interval))
                time.sleep(interval)
                session_exists = self._check_session(session_name)
        raise TimeoutError(
            "Timed out while waiting for session {} to finish.".format(session_name))


class TmuxJobWaiter(SessionJobWaiter):
    def __init__(self,
                 config: Dict[str, Any]) -> None:
        SessionJobWaiter.__init__(self, config, TMUX_SESSION_CHECK_SCRIPT)


class ScreenJobWaiter(SessionJobWaiter):
    def __init__(self,
                 config: Dict[str, Any]) -> None:
        SessionJobWaiter.__init__(self, config, SCREEN_SESSION_CHECK_SCRIPT)
