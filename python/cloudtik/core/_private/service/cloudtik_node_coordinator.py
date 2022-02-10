"""Node coordinating loop daemon."""

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

import cloudtik
from cloudtik.core._private import constants, services
from cloudtik.core._private.logging_utils import setup_component_logger
from cloudtik.core._private.state.control_state import ControlState

logger = logging.getLogger(__name__)


class NodeCoordinator:
    """Node Coordinator for node management

    Attributes:
        redis: A connection to the Redis server.
    """

    def __init__(self,
                 address,
                 redis_password=None,
                 coordinator_ip=None,
                 static_resource_list=None,
                 stop_event: Optional[Event] = None):

        # Initialize the Redis clients.
        redis_address = address
        self.redis = services.create_redis_client(
            redis_address, password=redis_password)
        (ip, port) = address.split(":")

        head_node_ip = redis_address.split(":")[0]
        self.redis_address = redis_address
        self.redis_password = redis_password
        self.coordinator_ip = coordinator_ip
        self.static_resource_list = static_resource_list
        self.node_resource_dict = self._parse_resource_list()

        # Can be used to signal graceful exit from coordinator loop.
        self.stop_event = stop_event  # type: Optional[Event]

        self.control_state = ControlState()
        self.control_state.initialize_control_state(ip, port, redis_password)
        self.node_table = self.control_state.get_node_table()
        logger.info("Coordinator: Started")

    def _run(self):
        """Run the coordinator loop."""
        self.create_heart_beat_thread()
        while True:
            if self.stop_event and self.stop_event.is_set():
                break

            # TODO (haifeng): implement the node coordinator functionality

            # Wait for update interval before processing the next
            # round of messages.
            time.sleep(constants.CLOUDTIK_UPDATE_INTERVAL_S)

    def _handle_failure(self, error):
        logger.exception("Error in coordinator loop")
        # TODO (haifeng): improve to publish the error through redis

    def _signal_handler(self, sig, frame):
        self._handle_failure(f"Terminated with signal {sig}\n" +
                             "".join(traceback.format_stack(frame)))
        sys.exit(sig + 128)


    def create_heart_beat_thread(self):
        thread = threading.Thread(target=self.send_heart_beat)
        # ensure when node_coordinator exits, the thread will stop automatically.
        thread.setDaemon(True)
        thread.start()

    def send_heart_beat(self):
        while True:
            time.sleep(constants.CLOUDTIK_HEARTBEAT_PERIOD_SECONDS)
            now = time.time()
            node_info = self.node_resource_dict.copy()
            node_info.update({"last_heartbeat_time": now})
            as_json = json.dumps(node_info)
            self.node_table.put("worker", as_json)
            pass

    def _parse_resource_list(self):
        node_resource_dict = {}
        resource_split = self.static_resource_list.split(",")
        for i in range(int(len(resource_split) / 2)):
            node_resource_dict[resource_split[2 * i]] = float(resource_split[2 * i + 1])
        return node_resource_dict

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
                     "coordinator to connect to."))
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
        default=constants.LOG_FILE_NAME_NODE_COORDINATOR,
        help="Specify the name of log file, "
        "log to stdout if set empty, default is "
        f"\"{constants.LOG_FILE_NAME_NODE_COORDINATOR}\"")
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
        "--coordinator-ip",
        required=False,
        type=str,
        default=None,
        help="The IP address of the machine hosting the coordinator process.")
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

    logger.info(f"Starting Node Coordinator using CloudTik installation: {cloudtik.__file__}")
    logger.info(f"CloudTik version: {cloudtik.__version__}")
    logger.info(f"CloudTik commit: {cloudtik.__commit__}")
    logger.info(f"Node Coordinator started with command: {sys.argv}")

    coordinator = NodeCoordinator(
        args.redis_address,
        redis_password=args.redis_password,
        coordinator_ip=args.coordinator_ip,
        static_resource_list=args.static_resource_list)

    coordinator.run()
