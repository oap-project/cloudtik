"""Web server that runs on on-premise/private clusters to simulate cloud operations and manage
different clusters for multiple users. It receives node provider function calls
through HTTP requests from remote OnPremiseNodeProvider and runs them
locally in CloudSimulatorScheduler. To start the webserver the user runs:
`python cloudtik_cloud_simulator.py --ips <comma separated ips> --port <PORT>`."""
import argparse
import datetime
import logging
import os
import threading
from http.server import SimpleHTTPRequestHandler, HTTPServer
import json

from cloudtik.core._private import constants
from cloudtik.core._private.core_utils import try_to_create_directory, try_to_symlink
from cloudtik.core._private.logging_utils import setup_component_logger
from cloudtik.core._private.utils import save_server_process, get_user_temp_dir, get_cloudtik_temp_dir
from cloudtik.providers._private.onpremise.config import DEFAULT_CLOUD_SIMULATOR_PORT, \
    _get_http_response_from_simulator, get_cloud_simulator_process_file, _discover_cloud_simulator
from cloudtik.providers._private.onpremise.cloud_simulator_scheduler \
    import CloudSimulatorScheduler, load_provider_config

logger = logging.getLogger(__name__)

LOG_FILE_NAME_CLOUD_SIMULATOR = f"cloudtik_cloud_simulator.log"


def runner_handler(node_provider):
    class Handler(SimpleHTTPRequestHandler):
        """A custom handler for Cloud Simulator.

        Handles all requests and responses coming into and from the
        remote CloudSimulatorScheduler.
        """

        def __init__(self, *args, directory=None, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)
            self._http_server = None

        def _do_header(self, response_code=200, headers=None):
            """Sends the header portion of the HTTP response.

            Args:
                response_code (int): Standard HTTP response code
                headers (list[tuples]): Standard HTTP response headers
            """
            if headers is None:
                headers = [("Content-type", "application/json")]

            self.send_response(response_code)
            for key, value in headers:
                self.send_header(key, value)
            self.end_headers()

        def do_HEAD(self):
            """HTTP HEAD handler method."""
            self._do_header()

        def do_GET(self):
            """Processes requests from remote CloudSimulatorScheduler."""
            if self.headers["content-length"]:
                raw_data = (self.rfile.read(
                    int(self.headers["content-length"]))).decode("utf-8")

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Cloud Simulator received request: " + str(raw_data))
                request = json.loads(raw_data)

                if "shutdown" == request["type"]:
                    logger.info("Cloud Simulator is going down...")

                    def kill_me_please(server):
                        server.shutdown()
                        logger.info("Shutdown Cloud Simulator successfully.")

                    shutdown_thread = threading.Thread(
                        target=kill_me_please,
                        args=(self._http_server,))
                    shutdown_thread.start()
                    response = None
                else:
                    response = getattr(node_provider,
                                       request["type"])(*request["args"])
                response_code = 200
                message = json.dumps(response)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Cloud Simulator response content: " + str(message))
                self._do_header(response_code=response_code)
                self.wfile.write(message.encode())

    return Handler


class CloudSimulator(threading.Thread):
    """Initializes HTTPServer and serves CloudSimulatorScheduler forever.

    It handles requests from the remote CloudSimulatorScheduler. The
    requests are forwarded to CloudSimulatorScheduler function calls.
    """

    def __init__(self, config, host, port):
        """Initialize HTTPServer and serve forever by invoking self.run()."""

        logger.info("Running Cloud Simulator on address " + host +
                    ":" + str(port))
        threading.Thread.__init__(self)
        self._port = port
        self._config = config
        address = (host, self._port)

        provider_config = load_provider_config(config)
        scheduler = CloudSimulatorScheduler(provider_config, cluster_name=None)
        request_handler = runner_handler(scheduler)
        self._server = HTTPServer(
            address,
            request_handler,
        )
        request_handler._http_server = self._server

        bind_address, bind_port = self._server.server_address
        server_process = {"pid": os.getpid(), "bind_address": bind_address, "port": bind_port}
        process_file = get_cloud_simulator_process_file()
        save_server_process(process_file, server_process)

        self.start()

    def run(self):
        self._server.serve_forever()

    def shutdown(self):
        """Shutdown the underlying server."""
        self._server.shutdown()
        self._server.server_close()

        process_file = get_cloud_simulator_process_file()
        save_server_process(process_file, {})


def start_server(
        config_file, bind_address, port, args):
    if bind_address is None:
        bind_address = "0.0.0.0"
    if port is None:
        port = DEFAULT_CLOUD_SIMULATOR_PORT

    if not args.logs_dir:
        temp_dir = get_cloudtik_temp_dir()
        args.logs_dir = os.path.join(temp_dir, "cloud-simulator")

    try_to_create_directory(args.logs_dir)

    # session
    # date including microsecond
    date_str = datetime.datetime.today().strftime(
        "%Y-%m-%d_%H-%M-%S_%f")
    session_name = f"session_{date_str}_{os.getpid()}"
    session_dir = os.path.join(args.logs_dir, session_name)
    session_symlink = os.path.join(args.logs_dir, "session_latest")

    # Send a warning message if the session exists.
    try_to_create_directory(session_dir)
    try_to_symlink(session_symlink, session_dir)

    # Create a directory to be used for process log files.
    logs_dir = os.path.join(session_dir, "logs")
    try_to_create_directory(logs_dir)

    setup_component_logger(
        logging_level=args.logging_level,
        logging_format=args.logging_format,
        log_dir=logs_dir,
        filename=args.logging_filename,
        max_bytes=args.logging_rotate_bytes,
        backup_count=args.logging_rotate_backup_count)

    print("Logging to: {}".format(logs_dir))
    CloudSimulator(
        config=config_file,
        host=bind_address,
        port=port,
    )


def _get_cloud_simulator_address(bind_address, port):
    if bind_address is None:
        cloud_simulator_address = _discover_cloud_simulator()
    else:
        if port is None:
            port = DEFAULT_CLOUD_SIMULATOR_PORT
        cloud_simulator_address = "{}:{}".format(bind_address, port)
    return cloud_simulator_address


def reload_config(
        config_file, bind_address, port):
    cloud_simulator_address = _get_cloud_simulator_address(
        bind_address, port)

    def _get_http_response(request):
        return _get_http_response_from_simulator(cloud_simulator_address, request)

    try:
        # make a HTTP request to reload the config
        request = {"type": "reload", "args": (config_file,)}
        _get_http_response(request)
        print("Configuration reloaded successfully.")
    except Exception as e:
        print("Failed to reload the configurations: {}".format(str(e)))


def shutdown_server(
        config_file, bind_address, port):
    cloud_simulator_address = _get_cloud_simulator_address(
        bind_address, port)

    def _get_http_response(request):
        return _get_http_response_from_simulator(cloud_simulator_address, request)

    try:
        request = {"type": "shutdown", "args": ()}
        _get_http_response(request)
        print("Shutdown successfully.")
    except Exception as e:
        print("Failed to shutdown: {}".format(str(e)))


def main():
    parser = argparse.ArgumentParser(
        description="Please provide a config file for nodes to start cloud simulator.")
    parser.add_argument(
        "config", help="A config file for nodes. The same format of on-premise provider section at top level.")
    parser.add_argument(
        "--bind-address",
        type=str,
        required=False,
        help="The address to bind. Bind to the address resolved from hostname if not specified.")
    parser.add_argument(
        "--port",
        type=int,
        required=False,
        help="The port on which the Cloud Simulator listens. Default: {}".format(DEFAULT_CLOUD_SIMULATOR_PORT))

    parser.add_argument(
        "--reload", default=False, action="store_true",
        help="Request the running cloud simulator service to reload the configuration for applying a change.")

    parser.add_argument(
        "--shutdown", default=False, action="store_true",
        help="Request the running cloud simulator service to shutdown itself.")
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
        default=LOG_FILE_NAME_CLOUD_SIMULATOR,
        help="Specify the name of log file, "
             "log to stdout if set empty, default is "
             f"\"{LOG_FILE_NAME_CLOUD_SIMULATOR}\"")
    parser.add_argument(
        "--logs-dir",
        required=False,
        type=str,
        help="Specify the path of the temporary directory used by cloud simulator "
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

    args = parser.parse_args()
    if args.reload and args.shutdown:
        print("Can only specify one of the two options: --reload or --shutdown.")
        return

    if args.reload:
        reload_config(
            args.config, args.bind_address, args.port)
    elif args.shutdown:
        shutdown_server(
            args.config, args.bind_address, args.port)
    else:
        start_server(
            args.config, args.bind_address, args.port, args)


if __name__ == "__main__":
    main()
