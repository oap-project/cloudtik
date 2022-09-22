import json
import time
from typing import Optional

from cloudtik.core._private import constants
from cloudtik.core._private.cli_logger import cli_logger
from cloudtik.core.job_waiter import JobWaiter
from cloudtik.runtime.spark.utils import request_rest_yarn


class SparkJobWaiter(JobWaiter):

    def _get_unfinished_yarn_jobs(self):
        response = request_rest_yarn(self.config, None)
        json_object = json.loads(response)
        return json_object["clusterMetrics"]["appsPending"], json_object["clusterMetrics"]["appsRunning"]

    def wait_for_completion(self, cmd: str, timeout: Optional[int] = None):
        start_time = time.time()
        interval = constants.CLOUDTIK_WAIT_FOR_JOB_FINISHED_INTERVAL_S
        apps_pending, apps_running = self._get_unfinished_yarn_jobs()
        if timeout is None:
            timeout = 60*60*3
        while time.time() - start_time < timeout:
            if apps_pending == 0 and apps_running == 0:
                cli_logger.print("All Spark jobs are finished! ")
                return
            else:
                cli_logger.print("Waiting for spark jobs to finish: {} pending jobs, {} running jobs ({} seconds)...", apps_pending,
                                 apps_running,
                                 interval)
                time.sleep(interval)
                apps_pending, apps_running = self._get_unfinished_yarn_jobs()
        raise TimeoutError("Timed out while waiting for spark jobs to finish: remain {} pending jobs, {} running jobs".format(apps_pending, apps_running))
