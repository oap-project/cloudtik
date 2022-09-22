import hashlib
import logging
import os
import time
from typing import Any, Dict, Optional

from cloudtik.core._private.cli_logger import cli_logger
from cloudtik.core._private.constants import CLOUDTIK_WAIT_FOR_JOB_FINISHED_INTERVAL_S, CLOUDTIK_JOB_WAITER_TIMEOUT_MAX
from cloudtik.core.job_waiter import JobWaiter

logger = logging.getLogger(__name__)

JOB_WAITER_SCRIPT_HOME = os.path.dirname(__file__)
TMUX_SESSION_CHECK_SCRIPT = os.path.join(JOB_WAITER_SCRIPT_HOME, "tmux-session.sh")
SCREEN_SESSION_CHECK_SCRIPT = os.path.join(JOB_WAITER_SCRIPT_HOME, "screen-session.sh")


def _get_command_session_name(cmd: str):
    hasher = hashlib.sha1()
    hasher.update(cmd.encode("utf-8"))
    return "cloudtik-" + hasher.hexdigest()


class SessionJobWaiter(JobWaiter):
    def __init__(self,
                 config: Dict[str, Any], session_check_script: str) -> None:
        JobWaiter.__init__(self, config)
        self.session_check_script = session_check_script

    def _check_session(self, session_name):
        return True

    def wait_for_completion(self, cmd: str, timeout: Optional[int] = None):
        start_time = time.time()
        if timeout is None:
            timeout = CLOUDTIK_JOB_WAITER_TIMEOUT_MAX
        interval = CLOUDTIK_WAIT_FOR_JOB_FINISHED_INTERVAL_S

        session_name = _get_command_session_name(cmd)
        session_exists = self._check_session(session_name)
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
