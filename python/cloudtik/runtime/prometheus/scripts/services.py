import argparse
import os

from cloudtik.core._private.runtime_utils import get_runtime_value
from cloudtik.runtime.prometheus.utils import start_pull_server, stop_pull_server, _get_home_dir


def _is_scrape_local_file():
    home_dir = _get_home_dir()
    config_file = os.path.join(
        home_dir, "conf", "scrape-config-local-file.yaml")
    return os.path.exists(config_file)


def start_service(head):
    if _is_scrape_local_file():
        # needed for only scrape local cluster with file
        start_pull_server(head)


def stop_service():
    if _is_scrape_local_file():
        stop_pull_server()


def main():
    parser = argparse.ArgumentParser(
        description="Start or stop runtime services")
    parser.add_argument(
        '--head', action='store_true', default=False,
        help='Start or stop services for head node.')
    # positional
    parser.add_argument(
        "command", type=str,
        help="The service command to execute: start or stop")
    parser.add_argument(
        "command_args",
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    high_availability = get_runtime_value("PROMETHEUS_HIGH_AVAILABILITY")
    if high_availability == "true" or args.head:
        if args.command == "start":
            start_service(args.head)
        elif args.command == "stop":
            stop_service()


if __name__ == "__main__":
    main()
