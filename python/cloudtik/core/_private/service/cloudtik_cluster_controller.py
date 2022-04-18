"""Cluster control loop daemon."""

import argparse
from dataclasses import asdict
import logging.handlers
import os
import sys
import signal
import time
import traceback
import json
from multiprocessing.synchronize import Event
from typing import Optional

try:
    import prometheus_client
except ImportError:
    prometheus_client = None

import cloudtik
from cloudtik.core._private.cluster.cluster_scaler import ClusterScaler
from cloudtik.core._private.cluster.cluster_operator import teardown_cluster
from cloudtik.core._private.constants import CLOUDTIK_UPDATE_INTERVAL_S, \
    CLOUDTIK_METRIC_PORT, CLOUDTIK_RESOURCE_REQUEST_CHANNEL
from cloudtik.core._private.cluster.event_summarizer import EventSummarizer
from cloudtik.core._private.prometheus_metrics import ClusterPrometheusMetrics
from cloudtik.core._private.cluster.cluster_metrics import ClusterMetrics
from cloudtik.core._private.utils import CLOUDTIK_CLUSTER_SCALING_ERROR, \
    CLOUDTIK_CLUSTER_SCALING_STATUS
from cloudtik.core._private import constants, services
from cloudtik.core._private.logging_utils import setup_component_logger
from cloudtik.core._private.state.kv_store import kv_initialize, \
    kv_put, kv_initialized, kv_get, kv_del
from cloudtik.core._private.state.control_state import ControlState, ResourceInfoClient, StateClient
from cloudtik.core._private.services import validate_redis_address

logger = logging.getLogger(__name__)


MAX_FAILURES_FOR_LOGGING = 16


def parse_resource_demands(resource_load):
    """Handle the message.resource_load for the demand
    based cluster scaling.

    Args:
        resource_load (ResourceLoad): The resource demands or None.

    Returns:
        List[ResourceDict]: Waiting bundles (ready and feasible).
        List[ResourceDict]: Infeasible bundles.
    """
    waiting_bundles, infeasible_bundles = [], []

    # TODO (haifeng): implement this in the future for resource demands based scaling
    return waiting_bundles, infeasible_bundles


class ClusterController:
    """Cluster Controller for scaling  workers

    This process periodically collects stats from the control state and triggers
    cluster scaler updates.

    Attributes:
        redis: A connection to the Redis server.
    """

    def __init__(self,
                 address,
                 cluster_scaling_config,
                 redis_password=None,
                 prefix_cluster_info=False,
                 controller_ip=None,
                 stop_event: Optional[Event] = None):

        # Initialize the Redis clients.
        redis_address = address
        self.redis = services.create_redis_client(
            redis_address, password=redis_password)
        (ip, port) = address.split(":")

        if prometheus_client:
            controller_addr = f"{controller_ip}:{CLOUDTIK_METRIC_PORT}"
            # TODO (haifeng): handle metrics
            self.redis.set(constants.CLOUDTIK_METRIC_ADDRESS_KEY, controller_addr)

        control_state = ControlState()
        _, redis_ip_address, redis_port = validate_redis_address(redis_address)
        control_state.initialize_control_state(redis_ip_address, redis_port, redis_password)

        self.resource_info_client = ResourceInfoClient.create_from_control_state(control_state)

        head_node_ip = redis_address.split(":")[0]
        self.redis_address = redis_address
        self.redis_password = redis_password

        # initialize the global kv store client
        state_client = StateClient.create_from_redis(self.redis)
        kv_initialize(state_client)

        self.cluster_metrics = ClusterMetrics()
        self.last_avail_resources = None
        self.event_summarizer = EventSummarizer()
        self.prefix_cluster_info = prefix_cluster_info
        # Can be used to signal graceful exit from controller loop.
        self.stop_event = stop_event  # type: Optional[Event]
        self.cluster_scaling_config = cluster_scaling_config
        self.cluster_scaler = None
        self.cluster_metrics_failures = 0

        self.prometheus_metrics = ClusterPrometheusMetrics()
        if prometheus_client:
            try:
                logger.info(
                    "Starting metrics server on port {}".format(
                        CLOUDTIK_METRIC_PORT))
                prometheus_client.start_http_server(
                    port=CLOUDTIK_METRIC_PORT,
                    addr=controller_ip,
                    registry=self.prometheus_metrics.registry)
            except Exception:
                logger.exception(
                    "An exception occurred while starting the metrics server.")
        else:
            logger.warning("`prometheus_client` not found, so metrics will "
                           "not be exported.")

        logger.info("Controller: Started")

    def _initialize_cluster_scaler(self):
        cluster_scaling_config = self.cluster_scaling_config

        self.cluster_scaler = ClusterScaler(
            cluster_scaling_config,
            self.cluster_metrics,
            prefix_cluster_info=self.prefix_cluster_info,
            event_summarizer=self.event_summarizer,
            prometheus_metrics=self.prometheus_metrics)

    def update_cluster_metrics(self):
        try:
            self._update_cluster_metrics()
            # reset if there is a success
            self.cluster_metrics_failures = 0
        except Exception as e:
            if self.cluster_metrics_failures == 0 or self.cluster_metrics_failures == MAX_FAILURES_FOR_LOGGING:
                # detailed form
                error = traceback.format_exc()
                logger.exception(f"Load metrics update failed with the following error:\n{error}")
            elif self.cluster_metrics_failures < MAX_FAILURES_FOR_LOGGING:
                # short form
                logger.exception(f"Load metrics update failed with the following error:{str(e)}")

            if self.cluster_metrics_failures == MAX_FAILURES_FOR_LOGGING:
                logger.exception(f"The above error has been showed consecutively"
                                 f" for {self.cluster_metrics_failures} times. Stop showing.")

            self.cluster_metrics_failures += 1

    def _update_cluster_metrics(self):
        """Fetches resource usage data from control state and updates load metrics."""
        # TODO (haifeng): implement load metrics
        resources_usage_batch = self.resource_info_client.get_cluster_resource_usage(timeout=60)
        waiting_bundles, infeasible_bundles = parse_resource_demands(
            resources_usage_batch.resource_demands)

        cluster_full = False
        for resource_message in resources_usage_batch.batch:
            node_id = resource_message.get('node_id')
            last_heartbeat_time = resource_message.get('last_heartbeat_time')
            # Generate node type config based on reported node list.

            if (hasattr(resource_message, "cluster_full")
                    and resource_message.get('cluster_full')):
                # Aggregate this flag across all batches.
                cluster_full = True
            # FIXME: implement the dynamic adjustment
            resource_load = {}
            total_resources = {}
            available_resources = {}

            use_node_id_as_ip = (self.cluster_scaler is not None
                                 and self.cluster_scaler.config["provider"].get("use_node_id_as_ip", False))

            # "use_node_id_as_ip" is a hack meant to address situations in
            # which there's more than one service node residing at a given ip.

            # TODO: Stop using ips as node identifiers.
            # (1) generating node ids when launching nodes, and
            # (2) propagating the node id to the start command so the node will
            # report resource stats under its assigned node id.

            if use_node_id_as_ip:
                ip = node_id.hex()
            else:
                ip = resource_message["resource"].get("ip")
            self.cluster_metrics.update(ip, node_id, last_heartbeat_time, total_resources,
                                        available_resources, resource_load,
                                        waiting_bundles, infeasible_bundles,
                                        cluster_full)

    def update_resource_requests(self):
        """Fetches resource requests from the internal KV and updates load."""
        if not kv_initialized():
            return
        data = kv_get(CLOUDTIK_RESOURCE_REQUEST_CHANNEL)
        if data:
            try:
                resource_request = json.loads(data)
                self.cluster_metrics.set_resource_requests(resource_request)
            except Exception:
                logger.exception("Error parsing resource requests")

    def _run(self):
        """Run the controller loop."""
        while True:
            if self.stop_event and self.stop_event.is_set():
                break
            self.update_cluster_metrics()
            # self.update_resource_requests()
            # self.update_event_summary()
            status = {
                "cluster_metrics_report": asdict(self.cluster_metrics.summary()),
                "time": time.time(),
                "controller_pid": os.getpid()
            }

            # Process autoscaling actions
            if self.cluster_scaler:
                # Only used to update the load metrics for the scaler.
                self.cluster_scaler.update()
                status["cluster_scaler_report"] = asdict(self.cluster_scaler.summary())

                for msg in self.event_summarizer.summary():
                    # Need to prefix each line of the message for the lines to
                    # get pushed to the driver logs.
                    for line in msg.split("\n"):
                        logger.info("{}{}".format(
                            constants.LOG_PREFIX_EVENT_SUMMARY, line))
                self.event_summarizer.clear()

            as_json = json.dumps(status)
            if kv_initialized():
                kv_put(
                    CLOUDTIK_CLUSTER_SCALING_STATUS, as_json, overwrite=True)

            # Wait for a cluster scaler update interval before processing the next
            # round of messages.
            time.sleep(CLOUDTIK_UPDATE_INTERVAL_S)

    def update_event_summary(self):
        """Report the current size of the cluster.

        To avoid log spam, only cluster size changes (CPU or GPU count change)
        are reported to the event summarizer. The event summarizer will report
        only the latest cluster size per batch.
        """
        avail_resources = self.cluster_metrics.resources_avail_summary()
        if avail_resources != self.last_avail_resources:
            self.event_summarizer.add(
                "Resized to {}.",  # e.g., Resized to 100 CPUs, 4 GPUs.
                quantity=avail_resources,
                aggregate=lambda old, new: new)
            self.last_avail_resources = avail_resources

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
        description=("Parse Redis server for the "
                     "controller to connect to."))
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
        default=constants.LOGGER_LEVEL,
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
