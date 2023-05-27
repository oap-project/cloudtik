"""Web server that runs on on-premise/private clusters to simulate cloud operations and manage
different clusters for multiple users. It receives node provider function calls
through HTTP requests from remote OnPremiseNodeProvider and runs them
locally in CloudSimulatorScheduler. To start the webserver the user runs:
`python cloudtik_cloud_simulator.py --ips <comma separated ips> --port <PORT>`."""
import argparse
import logging
import os
import threading
from http.server import SimpleHTTPRequestHandler, HTTPServer
import json
import socket

from cloudtik.core._private.utils import save_server_process
from cloudtik.providers._private.onpremise.config import DEFAULT_CLOUD_SIMULATOR_PORT, \
    _get_http_response_from_simulator, get_cloud_simulator_process_file
from cloudtik.providers._private.onpremise.cloud_simulator_scheduler \
    import CloudSimulatorScheduler, load_provider_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def runner_handler(node_provider):
    class Handler(SimpleHTTPRequestHandler):
        """A custom handler for Cloud Simulator.

        Handles all requests and responses coming into and from the
        remote CloudSimulatorScheduler.
        """

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
                logger.info("Cloud Simulator received request: " +
                            str(raw_data))
                request = json.loads(raw_data)
                response = getattr(node_provider,
                                   request["type"])(*request["args"])
                logger.info("Cloud Simulator response content: " +
                            str(raw_data))
                response_code = 200
                message = json.dumps(response)
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
        self._server = HTTPServer(
            address,
            runner_handler(CloudSimulatorScheduler(provider_config, cluster_name=None)),
        )

        server_process = {"pid": os.getpid(), "bind_address": host, "port": self._port}

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
        config_file, bind_address, port):
    if bind_address is None:
        bind_address = socket.gethostbyname(socket.gethostname())
    if port is None:
        port = DEFAULT_CLOUD_SIMULATOR_PORT
    CloudSimulator(
        config=config_file,
        host=bind_address,
        port=port,
    )


def reload_config(
        config_file, bind_address, port):
    if bind_address is None:
        bind_address = socket.gethostbyname(socket.gethostname())
    if port is None:
        port = DEFAULT_CLOUD_SIMULATOR_PORT

    cloud_simulator_address = "{}:{}".format(bind_address, port)

    def _get_http_response(request):
        return _get_http_response_from_simulator(cloud_simulator_address, request)

    try:
        # make a HTTP request to reload the config
        request = {"type": "reload", "args": (config_file,)}
        _get_http_response(request)
        print("Configuration reloaded successfully.")
    except Exception as e:
        print("Failed to reload the configurations: {}".format(str(e)))


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

    args = parser.parse_args()
    if args.reload:
        reload_config(
            args.config, args.bind_address, args.port)
    else:
        start_server(
            args.config, args.bind_address, args.port)


if __name__ == "__main__":
    main()
