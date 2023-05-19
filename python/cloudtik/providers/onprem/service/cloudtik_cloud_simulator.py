"""Web server that runs on on-premise/private clusters to simulate cloud operations and manage
different clusters for multiple users. It receives node provider function calls
through HTTP requests from remote OnPremNodeProvider and runs them
locally in CloudSimulatorNodeProvider. To start the webserver the user runs:
`python cloudtik_cloud_simulator.py --ips <comma separated ips> --port <PORT>`."""
import argparse
import logging
import threading
from http.server import SimpleHTTPRequestHandler, HTTPServer
import json
import socket

import yaml

from cloudtik.providers._private.onprem.config import DEFAULT_CLOUD_SIMULATOR_PORT
from cloudtik.providers.onprem.service.cloud_simulator_node_provider \
    import CloudSimulatorNodeProvider


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_provider_config(config_file):
    with open(config_file) as f:
        config_object = yaml.safe_load(f) or {}

    return config_object


def runner_handler(node_provider):
    class Handler(SimpleHTTPRequestHandler):
        """A custom handler for Cloud Simulator.

        Handles all requests and responses coming into and from the
        remote CloudSimulatorNodeProvider.
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
            """Processes requests from remote CloudSimulatorNodeProvider."""
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
    """Initializes HTTPServer and serves CloudSimulatorNodeProvider forever.

    It handles requests from the remote CloudSimulatorNodeProvider. The
    requests are forwarded to CloudSimulatorNodeProvider function calls.
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
            runner_handler(CloudSimulatorNodeProvider(provider_config, cluster_name=None)),
        )
        self.start()

    def run(self):
        self._server.serve_forever()

    def shutdown(self):
        """Shutdown the underlying server."""
        self._server.shutdown()
        self._server.server_close()


def main():
    parser = argparse.ArgumentParser(
        description="Please provide a config file and port.")
    parser.add_argument(
        "config", help="A config file for nodes. The same format of on-prem provider section at top level.")
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
    args = parser.parse_args()
    bind_address = args.bind_address
    port = args.port
    if bind_address is None:
        bind_address = socket.gethostbyname(socket.gethostname())
    if port is None:
        port = DEFAULT_CLOUD_SIMULATOR_PORT
    CloudSimulator(
        config=args.config,
        host=bind_address,
        port=port,
    )


if __name__ == "__main__":
    main()
