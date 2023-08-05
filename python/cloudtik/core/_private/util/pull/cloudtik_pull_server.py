"""Pull server daemon."""

import argparse
import json
import logging.handlers
import signal
import sys
import time
import traceback
from multiprocessing.synchronize import Event
from typing import Optional

import cloudtik
from cloudtik.core._private import constants
from cloudtik.core._private.core_utils import load_class
from cloudtik.core._private.logging_utils import setup_component_logger
from cloudtik.core._private.util.pull.pull_job import ScriptPullJob
from cloudtik.core._private.util.pull.pull_server import PROCESS_PULL_SERVER

logger = logging.getLogger(__name__)

DEFAULT_PULL_INTERVAL = 10


def cmd_args_to_call_args(cmd_args):
    args = []
    kwargs = {}
    for arg in cmd_args:
        if arg.count('=') >= 1:
            key, value = arg.split('=', 1)
        else:
            key, value = None, arg
        try:
            value = json.loads(value)
        except ValueError:
            pass
        if key:
            kwargs[key] = value
        else:
            args.append(value)
    return args, kwargs


class PullServer:
    """Pull Server for user to run pulling tasks with a specific interval.
    The pulling tasks can be in the form of a PullJob class or python module,
    python script, or shell script to run.
    """

    def __init__(self,
                 identifier,
                 pull_class, pull_script,
                 pull_args=None, interval=None,
                 stop_event: Optional[Event] = None):
        self.identifier = identifier
        self.pull_class = pull_class
        self.pull_script = pull_script
        self.pull_args = pull_args
        self.interval = interval

        # Can be used to signal graceful exit from main loop.
        self.stop_event = stop_event  # type: Optional[Event]

        if not self.interval:
            self.interval = DEFAULT_PULL_INTERVAL

        self.pull_job = self._create_pull_job()

        logger.info("Pull Server: Started")

    def _create_pull_job(self):
        if self.pull_class:
            pull_job_cls = load_class(self.pull_class)
            args, kwargs = cmd_args_to_call_args(self.pull_args)
            return pull_job_cls(*args, **kwargs)
        else:
            return ScriptPullJob(self.pull_script, self.pull_args)

    def _run(self):
        """Run the main loop."""
        while True:
            if self.stop_event and self.stop_event.is_set():
                break

            try:
                self.pull_job.pull()
            except Exception as e:
                logger.exception("Error happened when pulling: " + str(e))
            time.sleep(self.interval)

    def _handle_failure(self, error):
        logger.exception("Error in pulling loop")
        logger.exception(f"The pulling failed with the following error:\n{error}")

    def _signal_handler(self, sig, frame):
        self._handle_failure(f"Terminated with signal {sig}\n" +
                             "".join(traceback.format_stack(frame)))
        sys.exit(sig + 128)

    def run(self):
        # Register signal handlers for pull server termination.
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        try:
            self._run()
        except Exception:
            self._handle_failure(traceback.format_exc())
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Parse the arguments of Pull Server")
    parser.add_argument(
        "--identifier",
        required=True,
        type=str,
        help="The identifier of this pull instance.")
    parser.add_argument(
        "--pull-class",
        required=False,
        type=str,
        help="The python module and class to run for pulling.")
    parser.add_argument(
        "--pull-script",
        required=False,
        type=str,
        help="The bash script or python script to run for pulling.")
    parser.add_argument(
        "--interval",
        required=False,
        type=int,
        help="The pull interval.")
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
        default=PROCESS_PULL_SERVER,
        help="Specify the name of log file, "
             "log to stdout if set empty, default is "
             f"\"{PROCESS_PULL_SERVER}\"")
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
    args, argv = parser.parse_known_args()
    setup_component_logger(
        logging_level=args.logging_level,
        logging_format=args.logging_format,
        log_dir=args.logs_dir,
        filename=args.logging_filename,
        max_bytes=args.logging_rotate_bytes,
        backup_count=args.logging_rotate_backup_count)

    logger.info(f"Starting Pull Server using CloudTik installation: {cloudtik.__file__}")
    logger.info(f"CloudTik version: {cloudtik.__version__}")
    logger.info(f"CloudTik commit: {cloudtik.__commit__}")
    logger.info(f"Pull Server started with command: {sys.argv}")

    pull_server = PullServer(
        args.identifier,
        args.pull_class,
        args.pull_script,
        pull_args=argv,
        interval=args.interval
    )

    pull_server.run()


if __name__ == "__main__":
    main()
