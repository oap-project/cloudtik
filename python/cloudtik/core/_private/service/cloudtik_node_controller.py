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
from cloudtik.core._private import constants, services
from cloudtik.core._private.logging_utils import setup_component_logger
from cloudtik.core._private.state.control_state import ControlState
from cloudtik.runtime.spark.utils import get_spark_runtime_processes

logger = logging.getLogger(__name__)


class NodeController:
    """Node Controller for node management

    Attributes:
        redis: A connection to the Redis server.
    """

    def __init__(self,
                 node_type,
                 address,
                 redis_password=None,
                 controller_ip=None,
                 static_resource_list=None,
                 stop_event: Optional[Event] = None):

        # Initialize the Redis clients.
        redis_address = address
        self.redis = services.create_redis_client(
            redis_address, password=redis_password)
        (ip, port) = address.split(":")

        self.redis_address = redis_address
        self.redis_password = redis_password
        self.controller_ip = controller_ip
        self.node_type = node_type
        self.static_resource_list = static_resource_list
        # node_detail store the resource, process and other details of the current node
        self.node_detail = {}
        resource_dict, node_id = self._parse_resource_list()
        self.node_detail["resource"] = resource_dict
        self.node_detail["node_id"] = node_id
        self.node_detail["node_type"] = self.node_type
        self.node_id = node_id
        self.old_processes = []

        # Can be used to signal graceful exit from controller loop.
        self.stop_event = stop_event  # type: Optional[Event]

        self.control_state = ControlState()
        self.control_state.initialize_control_state(ip, port, redis_password)
        self.node_table = self.control_state.get_node_table()
        self.processes_to_check = constants.CLOUDTIK_PROCESSES
        self.processes_to_check.extend(get_spark_runtime_processes())
        logger.info("Controller: Started")

    def _run(self):
        """Run the controller loop."""
        self.create_heart_beat_thread()
        while True:
            if self.stop_event and self.stop_event.is_set():
                break

            # Wait for update interval before processing the next
            # round of messages.
            self._check_process()
            time.sleep(constants.CLOUDTIK_UPDATE_INTERVAL_S)

    def _handle_failure(self, error):
        logger.exception("Error in controller loop")

    def _signal_handler(self, sig, frame):
        self._handle_failure(f"Terminated with signal {sig}\n" +
                             "".join(traceback.format_stack(frame)))
        sys.exit(sig + 128)

    def create_heart_beat_thread(self):
        thread = threading.Thread(target=self.send_heart_beat)
        # ensure when node_controller exits, the thread will stop automatically.
        thread.setDaemon(True)
        thread.start()

    def send_heart_beat(self):
        while True:
            time.sleep(constants.CLOUDTIK_HEARTBEAT_PERIOD_SECONDS)
            now = time.time()
            node_info = self.node_detail.copy()
            node_info.update({"last_heartbeat_time": now})
            as_json = json.dumps(node_info)
            try:
                self.node_table.put(self.node_id, as_json)
            except Exception as e:
                logger.exception("Failed sending heartbeat: " + str(e))
                logger.exception(traceback.format_exc())

    def _parse_resource_list(self):
        node_resource_dict = {}
        resource_split = self.static_resource_list.split(",")
        for i in range(int(len(resource_split) / 2)):
            if "node" in resource_split[2 * i]:
                node_resource_dict["ip"] = resource_split[2 * i].split(":")[1]
                node_id = resource_split[2 * i].replace(":", "_")
            else:
                node_resource_dict[resource_split[2 * i]] = float(resource_split[2 * i + 1])
        return node_resource_dict, node_id

    def _check_process(self):
        """check CloudTik runtime processes on the local machine."""
        process_infos = []
        for proc in psutil.process_iter(["name", "cmdline"]):
            try:
                process_infos.append((proc, proc.name(), proc.cmdline()))
            except psutil.Error:
                pass

        found_process = {}
        for keyword, filter_by_cmd, process_name, node_type in self.processes_to_check:
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
                if keyword in corpus and (self.node_type == node_type or "node" == node_type):
                    found_process[process_name] = proc.status()

        if found_process != self.old_processes:
            logger.info("Cloudtik processes status changed, latest process information: {}".format(str(found_process)))
        self.node_detail["process"] = found_process
        self.old_processes = found_process

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
        description=("Parse Redis server for the "
                     "controller to connect to."))
    parser.add_argument(
        "--node-type",
        required=True,
        type=str,
        help="the node type of the current node")
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
        default=constants.LOG_FILE_NAME_NODE_CONTROLLER,
        help="Specify the name of log file, "
        "log to stdout if set empty, default is "
        f"\"{constants.LOG_FILE_NAME_NODE_CONTROLLER}\"")
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
    parser.add_argument(
        "--static_resource_list",
        required=False,
        type=str,
        default="",
        help="The static resource list of this node.")
    args = parser.parse_args()
    setup_component_logger(
        logging_level=args.logging_level,
        logging_format=args.logging_format,
        log_dir=args.logs_dir,
        filename=args.logging_filename,
        max_bytes=args.logging_rotate_bytes,
        backup_count=args.logging_rotate_backup_count)

    logger.info(f"Starting Node Controller using CloudTik installation: {cloudtik.__file__}")
    logger.info(f"CloudTik version: {cloudtik.__version__}")
    logger.info(f"CloudTik commit: {cloudtik.__commit__}")
    logger.info(f"Node Controller started with command: {sys.argv}")

    controller = NodeController(
        args.node_type,
        args.redis_address,
        redis_password=args.redis_password,
        controller_ip=args.controller_ip,
        static_resource_list=args.static_resource_list)

    controller.run()
