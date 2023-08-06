import argparse
import os

from cloudtik.runtime.prometheus.utils import start_pull_server, stop_pull_server, _get_home_dir


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

    high_availability = os.environ.get("PROMETHEUS_HIGH_AVAILABILITY")
    if high_availability == "true" or args.head:
        home_dir = _get_home_dir()
        config_file = os.path.join(
            home_dir, "conf", "scrape-config-local-file.yaml")
        if os.path.exists(config_file):
            # needed for only scrape local cluster with file
            if args.command == "start":
                start_pull_server(args.head)
            elif args.command == "stop":
                stop_pull_server()


if __name__ == "__main__":
    main()
