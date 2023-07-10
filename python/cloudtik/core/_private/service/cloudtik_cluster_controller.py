"""Cluster control loop daemon."""

import argparse
import logging.handlers
import os
import sys
import signal
import time
import traceback
from multiprocessing.synchronize import Event
from typing import Optional

from cloudtik.core._private.cluster.cluster_metrics_updater import ClusterMetricsUpdater
from cloudtik.core._private.cluster.resource_scaling_policy import ResourceScalingPolicy
from cloudtik.core._private.state.scaling_state import ScalingStateClient

try:
    import prometheus_client
except ImportError:
    prometheus_client = None

import cloudtik
from cloudtik.core._private.cluster.cluster_scaler import ClusterScaler
from cloudtik.core._private.cluster.cluster_operator import teardown_cluster
from cloudtik.core._private.constants import CLOUDTIK_UPDATE_INTERVAL_S, \
    CLOUDTIK_METRIC_PORT
from cloudtik.core._private.cluster.event_summarizer import EventSummarizer
from cloudtik.core._private.prometheus_metrics import ClusterPrometheusMetrics
from cloudtik.core._private.cluster.cluster_metrics import ClusterMetrics
from cloudtik.core._private.utils import CLOUDTIK_CLUSTER_SCALING_ERROR
from cloudtik.core._private import constants, services
from cloudtik.core._private.logging_utils import setup_component_logger
from cloudtik.core._private.state.kv_store import kv_initialize, \
    kv_put, kv_initialized, kv_del
from cloudtik.core._private.state.control_state import ControlState, StateClient
from cloudtik.core._private.services import validate_redis_address

logger = logging.getLogger(__name__)


class ClusterController:
    """Cluster Controller for scaling  workers

    This process periodically collects stats from the control state and triggers
    cluster scaler updates.

    Attributes:
        redis: A connection to the Redis server.
    """

    def __init__(self,
                 redis_address,
                 cluster_scaling_config,
                 redis_password=None,
                 controller_ip=None,
                 stop_event: Optional[Event] = None,
                 retry_on_failure: bool = True):

        self.controller_ip = controller_ip
        # Initialize the Redis clients.
        self.redis = services.create_redis_client(
            redis_address, password=redis_password)
        (ip, port) = redis_address.split(":")

        if prometheus_client:
            controller_addr = f"{controller_ip}:{CLOUDTIK_METRIC_PORT}"
            # TODO: handle metrics
            self.redis.set(constants.CLOUDTIK_METRIC_ADDRESS_KEY, controller_addr)

        control_state = ControlState()
        _, redis_ip_address, redis_port = validate_redis_address(redis_address)
        control_state.initialize_control_state(redis_ip_address, redis_port, redis_password)

        # initialize the global kv store client
        state_client = StateClient.create_from_redis(self.redis)
        kv_initialize(state_client)

        self._session_name = self.get_session_name(state_client)
        logger.info(f"session_name: {self._session_name}")

        self.scaling_state_client = ScalingStateClient.create_from(control_state)

        self.head_ip = redis_address.split(":")[0]
        self.redis_address = redis_address
        self.redis_password = redis_password

        self.cluster_metrics = ClusterMetrics()
        self.event_summarizer = EventSummarizer()
        # Can be used to signal graceful exit from controller loop.
        self.stop_event = stop_event  # type: Optional[Event]
        self.retry_on_failure = retry_on_failure
        self.cluster_scaling_config = cluster_scaling_config
        self.cluster_scaler = None
        self.resource_scaling_policy = ResourceScalingPolicy(
            self.head_ip, self.scaling_state_client)
        self.cluster_metrics_updater = ClusterMetricsUpdater(
            self.cluster_metrics, self.event_summarizer, self.scaling_state_client)

        self.prometheus_metrics = ClusterPrometheusMetrics(
            session_name=self._session_name)
        if prometheus_client:
            try:
                logger.info(
                    "Starting metrics server on port {}".format(
                        CLOUDTIK_METRIC_PORT))
                prometheus_client.start_http_server(
                    port=CLOUDTIK_METRIC_PORT,
                    addr=controller_ip,
                    registry=self.prometheus_metrics.registry)
                # Reset some gauges, since we don't know which labels have
                # leaked if the cluster controller restarted.
                self.prometheus_metrics.pending_nodes_of_type.clear()
                self.prometheus_metrics.active_nodes_of_type.clear()
            except Exception:
                logger.exception(
                    "An exception occurred while starting the metrics server.")
        else:
            logger.warning("`prometheus_client` not found, so metrics will "
                           "not be exported.")

        logger.info("Controller: Started")

    def get_session_name(self, state_client) -> Optional[str]:
        """Obtain the session name from the state store.
        """
        session_name = state_client.kv_get(
            b"session_name",
            constants.KV_NAMESPACE_SESSION
        )

        if session_name:
            session_name = session_name.decode()

        return session_name

    def _initialize_cluster_scaler(self):
        self.cluster_scaler = ClusterScaler(
            self.cluster_scaling_config,
            self.cluster_metrics,
            cluster_metrics_updater=self.cluster_metrics_updater,
            resource_scaling_policy=self.resource_scaling_policy,
            event_summarizer=self.event_summarizer,
            prometheus_metrics=self.prometheus_metrics)

    def _run(self):
        """Run the controller loop."""
        while True:
            try:
                if self.stop_event and self.stop_event.is_set():
                    break

                # Process autoscaling actions
                if self.cluster_scaler:
                    self.cluster_scaler.run()
            except Exception:
                # By default, do not exit the controller on failure.
                if self.retry_on_failure:
                    logger.exception("Controller: Execution exception. Trying again...")
                else:
                    raise

            # Wait for a cluster scaler update interval before processing the next
            # round of messages.
            time.sleep(CLOUDTIK_UPDATE_INTERVAL_S)

    def destroy_cluster_scaler_workers(self):
        """Cleanup the cluster scaler, in case of an exception in the run() method.

        We kill the worker nodes, but retain the head node in order to keep
        logs around, keeping costs minimal. This controller process runs on the
        head node anyway, so this is more reliable."""

        if self.cluster_scaler is None:
            return  # Nothing to clean up.

        if self.cluster_scaling_config is None:
            # This is a logic error in the program. Can't do anything.
            logger.error(
                "Controller: Cleanup failed due to lack of cluster config.")
            return

        logger.info("Controller: Exception caught. Taking down workers...")
        clean = False
        while not clean:
            try:
                teardown_cluster(
                    config_file=self.cluster_scaling_config,
                    yes=True,  # Non-interactive.
                    workers_only=True,  # Retain head node for logs.
                    override_cluster_name=None,
                    keep_min_workers=True,  # Retain minimal amount of workers.
                )
                clean = True
                logger.info("Controller: Workers taken down.")
            except Exception:
                logger.error("Controller: Cleanup exception. Trying again...")
                time.sleep(2)

    def _handle_failure(self, error):
        logger.exception("Error in controller loop")
        if self.cluster_scaler is not None and \
           os.environ.get("CLOUDTIK_FATESHARE_WORKERS", "") == "1":
            self.cluster_scaler.kill_workers()
            # Take down cluster workers if necessary.
            self.destroy_cluster_scaler_workers()

        # Something went wrong, so push an error
        message = f"The cluster controller failed with the following error:\n{error}"
        if kv_initialized():
            kv_put(CLOUDTIK_CLUSTER_SCALING_ERROR, message, overwrite=True)

        redis_client = services.create_redis_client(
            self.redis_address, password=self.redis_password)

        from cloudtik.core._private.utils import publish_error
        publish_error(
            constants.ERROR_CLUSTER_CONTROLLER_DIED,
            message,
            redis_client=redis_client)

    def _signal_handler(self, sig, frame):
        self._handle_failure(f"Terminated with signal {sig}\n" +
                             "".join(traceback.format_stack(frame)))
        sys.exit(sig + 128)

    def run(self):
        # Register signal handlers for cluster scaler termination.
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        try:
            if kv_initialized():
                # Delete any previous autoscaling errors.
                kv_del(CLOUDTIK_CLUSTER_SCALING_ERROR)
            self._initialize_cluster_scaler()
            self._run()
        except Exception:
            self._handle_failure(traceback.format_exc())
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse the arguments of the Cluster Controller.")
    parser.add_argument(
        "--redis-address",
        required=True,
        type=str,
        help="the address to use for Redis")
    parser.add_argument(
        "--cluster-scaling-config",
        required=False,
        type=str,
        help="the path to the autoscaling config file")
    parser.add_argument(
        "--redis-password",
        required=False,
        type=str,
        default=None,
        help="the password to use for Redis")
    parser.add_argument(
        "--logging-level",
        required=False,
        type=str,
        default=constants.LOGGER_LEVEL_INFO,
        choices=constants.LOGGER_LEVEL_CHOICES,
        help=constants.LOGGER_LEVEL_HELP)
    parser.add_argument(
        "--logging-format",
        required=False,
        type=str,
        default=constants.LOGGER_FORMAT,
        help=constants.LOGGER_FORMAT_HELP)
    parser.add_argument(
        "--logging-filename",
        required=False,
        type=str,
        default=constants.LOG_FILE_NAME_CLUSTER_CONTROLLER,
        help="Specify the name of log file, "
        "log to stdout if set empty, default is "
        f"\"{constants.LOG_FILE_NAME_CLUSTER_CONTROLLER}\"")
    parser.add_argument(
        "--logs-dir",
        required=True,
        type=str,
        help="Specify the path of the temporary directory "
        "processes.")
    parser.add_argument(
        "--logging-rotate-bytes",
        required=False,
        type=int,
        default=constants.LOGGING_ROTATE_MAX_BYTES,
        help="Specify the max bytes for rotating "
        "log file, default is "
        f"{constants.LOGGING_ROTATE_MAX_BYTES} bytes.")
    parser.add_argument(
        "--logging-rotate-backup-count",
        required=False,
        type=int,
        default=constants.LOGGING_ROTATE_BACKUP_COUNT,
        help="Specify the backup count of rotated log file, default is "
        f"{constants.LOGGING_ROTATE_BACKUP_COUNT}.")
    parser.add_argument(
        "--controller-ip",
        required=False,
        type=str,
        default=None,
        help="The IP address of the machine hosting the controller process.")
    args = parser.parse_args()
    setup_component_logger(
        logging_level=args.logging_level,
        logging_format=args.logging_format,
        log_dir=args.logs_dir,
        filename=args.logging_filename,
        max_bytes=args.logging_rotate_bytes,
        backup_count=args.logging_rotate_backup_count)

    logger.info(f"Starting controller using CloudTik installation: {cloudtik.__file__}")
    logger.info(f"CloudTik version: {cloudtik.__version__}")
    logger.info(f"CloudTik commit: {cloudtik.__commit__}")
    logger.info(f"Controller started with command: {sys.argv}")

    if args.cluster_scaling_config:
        cluster_scaling_config = os.path.expanduser(args.cluster_scaling_config)
    else:
        cluster_scaling_config = None

    controller = ClusterController(
        args.redis_address,
        cluster_scaling_config,
        redis_password=args.redis_password,
        controller_ip=args.controller_ip)

    controller.run()
