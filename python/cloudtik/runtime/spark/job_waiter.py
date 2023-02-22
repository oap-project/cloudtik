import json
import time
from typing import Optional

from cloudtik.core._private.cli_logger import cli_logger
from cloudtik.core._private.constants import CLOUDTIK_WAIT_FOR_JOB_FINISHED_INTERVAL_S, CLOUDTIK_JOB_WAITER_TIMEOUT_MAX
from cloudtik.core.job_waiter import JobWaiter
from cloudtik.runtime.spark.utils import request_rest_yarn_with_retry


class SparkJobWaiter(JobWaiter):
    def _get_on_going_yarn_apps(self):
        response = request_rest_yarn_with_retry(self.config, None)
        json_object = json.loads(response)
        return json_object["clusterMetrics"]["appsPending"], json_object["clusterMetrics"]["appsRunning"]

    def wait_for_completion(self, node_id: str, cmd: str, session_name: str, timeout: Optional[int] = None):
        start_time = time.time()
        if timeout is None:
            timeout = CLOUDTIK_JOB_WAITER_TIMEOUT_MAX
        interval = CLOUDTIK_WAIT_FOR_JOB_FINISHED_INTERVAL_S

        apps_pending, apps_running = self._get_on_going_yarn_apps()
        while time.time() - start_time < timeout:
            if apps_pending == 0 and apps_running == 0:
                cli_logger.print("All Spark jobs now finished.")
                return
            else:
                cli_logger.print(
                    "Waiting for spark jobs to finish: {} pending jobs, {} running jobs ({} seconds)...".format(
                        apps_pending,
                        apps_running,
                        interval))
                time.sleep(interval)
                apps_pending, apps_running = self._get_on_going_yarn_apps()
        raise TimeoutError(
            "Timed out while waiting for spark jobs to finish: remain {} pending jobs, {} running jobs.".format(
                apps_pending, apps_running))
