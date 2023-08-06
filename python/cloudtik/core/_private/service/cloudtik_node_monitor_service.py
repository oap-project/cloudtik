"""Node control loop daemon."""

import argparse
import logging.handlers
import sys
import signal
import time
import traceback
import threading
from multiprocessing.synchronize import Event
from typing import Optional
import json
import psutil
import subprocess

import cloudtik
from cloudtik.core._private import constants
from cloudtik.core._private.logging_utils import setup_component_logger
from cloudtik.core._private.metrics.metrics_collector import MetricsCollector
from cloudtik.core._private.state.control_state import ControlState
from cloudtik.core._private.state.state_utils import NODE_STATE_NODE_IP, NODE_STATE_NODE_ID, NODE_STATE_NODE_KIND, \
    NODE_STATE_HEARTBEAT_TIME, NODE_STATE_NODE_TYPE
from cloudtik.core._private.utils import get_runtime_processes, make_node_id

logger = logging.getLogger(__name__)


class NodeMonitor:
    """Node Monitor for node heartbeats and node status updates
    """

    def __init__(self,
                 node_id,
                 node_ip,
                 node_kind,
                 node_type,
                 redis_address,
                 redis_password=None,
                 static_resource_list=None,
                 stop_event: Optional[Event] = None,
                 runtimes: str = None):
        if node_id is None:
            node_id = make_node_id(node_ip)
        self.node_id = node_id
        (redis_ip, redis_port) = redis_address.split(":")

        self.redis_address = redis_address
        self.redis_password = redis_password
        self.node_ip = node_ip
        self.node_kind = node_kind
        self.node_type = node_type
        self.static_resource_list = static_resource_list
        # node_info store the resource, process and other details of the current node
        self.old_processes = {}
        self.node_info = {
            NODE_STATE_NODE_ID: node_id,
            NODE_STATE_NODE_IP: node_ip,
            NODE_STATE_NODE_KIND: node_kind,
            "process": self.old_processes
        }
        if node_type:
            self.node_info[NODE_STATE_NODE_TYPE] = node_type

        self.node_metrics = {
            NODE_STATE_NODE_ID: node_id,
            NODE_STATE_NODE_IP: node_ip,
            NODE_STATE_NODE_KIND: node_kind,
            "metrics": {},
        }
        self.metrics_collector = None

        # Can be used to signal graceful exit from monitor loop.
        self.stop_event = stop_event  # type: Optional[Event]

        self.control_state = ControlState()
        self.control_state.initialize_control_state(redis_ip, redis_port, redis_password)
        self.node_table = self.control_state.get_node_table()
        self.node_metrics_table = self.control_state.get_node_metrics_table()

        self.processes_to_check = constants.CLOUDTIK_PROCESSES
        runtime_list = runtimes.split(",") if runtimes and len(runtimes) > 0 else None
        self.processes_to_check.extend(get_runtime_processes(runtime_list))
        self.node_info_lock = threading.Lock()

        logger.info("Monitor: Started")

    def _run(self):
        """Run the monitor loop."""
        self._run_heartbeat()
        self._update()

    def _update(self):
        while True:
            if self.stop_event and self.stop_event.is_set():
                break

            # Wait for update interval before processing the next
            # round of messages.
            try:
                self._update_process_status()
                self._update_metrics()
                self._send_node_metrics()
            except Exception as e:
                logger.exception("Error happened when checking processes: " + str(e))
            time.sleep(constants.CLOUDTIK_UPDATE_INTERVAL_S)

    def _handle_failure(self, error):
        logger.exception("Error in node monitor loop")
        logger.exception(f"The node monitor failed with the following error:\n{error}")

    def _signal_handler(self, sig, frame):
        self._handle_failure(f"Terminated with signal {sig}\n" +
                             "".join(traceback.format_stack(frame)))
        sys.exit(sig + 128)

    def _run_heartbeat(self):
        thread = threading.Thread(target=self._heartbeat)
        # ensure when node_monitor exits, the thread will stop automatically.
        thread.setDaemon(True)
        thread.start()

    def _heartbeat(self):
        while True:
            time.sleep(constants.CLOUDTIK_HEARTBEAT_PERIOD_SECONDS)
            now = time.time()
            with self.node_info_lock:
                node_info = self.node_info.copy()
            node_info.update({NODE_STATE_HEARTBEAT_TIME: now})
            node_info_as_json = json.dumps(node_info)
            try:
                self.node_table.put(self.node_id, node_info_as_json)
            except Exception as e:
                logger.exception("Failed sending heartbeat: " + str(e))
                logger.exception(traceback.format_exc())

    def _update_process_status(self):
        """check CloudTik runtime processes on the local machine."""
        process_infos = []
        for proc in psutil.process_iter(["name", "cmdline"]):
            try:
                process_infos.append((proc, proc.name(), proc.cmdline()))
            except psutil.Error:
                pass

        found_process = {}
        for keyword, filter_by_cmd, process_name, node_kind in self.processes_to_check:
            if (self.node_kind != node_kind) and ("node" != node_kind):
                continue

            if filter_by_cmd and len(keyword) > 15:
                # getting here is an internal bug, so we do not use cli_logger
                msg = ("The filter string should not be more than {} "
                       "characters. Actual length: {}. Filter: {}").format(
                    15, len(keyword), keyword)
                raise ValueError(msg)
            found_process[process_name] = "-"
            for candidate in process_infos:
                proc, proc_cmd, proc_args = candidate
                corpus = (proc_cmd
                          if filter_by_cmd else subprocess.list2cmdline(proc_args))
                if keyword in corpus:
                    found_process[process_name] = proc.status()

        if found_process != self.old_processes:
            logger.info("Cloudtik processes status changed, latest process information: {}".format(str(found_process)))
            # lock write
            with self.node_info_lock:
                self.node_info["process"] = found_process
            self.old_processes = found_process

    def _update_metrics(self):
        if self.metrics_collector is None:
            self.metrics_collector = MetricsCollector()

        metrics = self.metrics_collector.get_all_metrics()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Metrics collected for node: {}".format(metrics))
        self.node_metrics["metrics"] = metrics

    def _send_node_metrics(self):
        now = time.time()
        node_metrics = self.node_metrics
        node_metrics.update({"metrics_time": now})
        node_metrics_as_json = json.dumps(node_metrics)
        try:
            self.node_metrics_table.put(self.node_id, node_metrics_as_json)
        except Exception as e:
            logger.exception("Failed sending node metrics: " + str(e))
            logger.exception(traceback.format_exc())

    def run(self):
        # Register signal handlers for cluster scaler termination.
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        try:
            self._run()
        except Exception:
            self._handle_failure(traceback.format_exc())
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse the arguments of the Node Monitor")
    parser.add_argument(
        "--node-kind",
        required=True,
        type=str,
        help="the node kind of the current node: head or worker")
    parser.add_argument(
        "--node-type",
        required=False,
        type=str,
        default=None,
        help="the node type of the this node")
    parser.add_argument(
        "--redis-address",
        required=True,
        type=str,
        help="the address to use for Redis")
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
        default=constants.LOG_FILE_NAME_NODE_MONITOR,
        help="Specify the name of log file, "
        "log to stdout if set empty, default is "
        f"\"{constants.LOG_FILE_NAME_NODE_MONITOR}\"")
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
        "--node-id",
        required=False,
        type=str,
        default=None,
        help="The unique node id to use to for this node.")
    parser.add_argument(
        "--monitor-ip",
        required=False,
        type=str,
        default=None,
        help="The IP address of the machine hosting the monitor process.")
    parser.add_argument(
        "--static_resource_list",
        required=False,
        type=str,
        default="",
        help="The static resource list of this node.")
    parser.add_argument(
        "--runtimes",
        required=False,
        type=str,
        default=None,
        help="The runtimes enabled for this cluster.")
    args = parser.parse_args()
    setup_component_logger(
        logging_level=args.logging_level,
        logging_format=args.logging_format,
        log_dir=args.logs_dir,
        filename=args.logging_filename,
        max_bytes=args.logging_rotate_bytes,
        backup_count=args.logging_rotate_backup_count)

    logger.info(f"Starting Node Monitor using CloudTik installation: {cloudtik.__file__}")
    logger.info(f"CloudTik version: {cloudtik.__version__}")
    logger.info(f"CloudTik commit: {cloudtik.__commit__}")
    logger.info(f"Node Monitor started with command: {sys.argv}")

    node_monitor = NodeMonitor(
        args.node_id,
        args.monitor_ip,
        args.node_kind,
        args.node_type,
        args.redis_address,
        redis_password=args.redis_password,
        static_resource_list=args.static_resource_list,
        runtimes=args.runtimes,)

    node_monitor.run()
