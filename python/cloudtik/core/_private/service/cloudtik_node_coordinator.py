"""Node coordinating loop daemon."""

import argparse
import logging.handlers
import sys
import signal
import time
import traceback
from multiprocessing.synchronize import Event
from typing import Optional

import cloudtik
from cloudtik.core._private import constants, services
from cloudtik.core._private.logging_utils import setup_component_logger

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

        # Can be used to signal graceful exit from coordinator loop.
        self.stop_event = stop_event  # type: Optional[Event]

        logger.info("Coordinator: Started")

    def _run(self):
        """Run the coordinator loop."""
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
        coordinator_ip=args.coordinator_ip)

    coordinator.run()
