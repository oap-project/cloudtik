import argparse
import os

from cloudtik.runtime.grafana.utils import start_pull_server, stop_pull_server, \
    GRAFANA_DATA_SOURCES_SCOPE_WORKSPACE


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

    high_availability = os.environ.get("GRAFANA_HIGH_AVAILABILITY")
    if high_availability == "true" or args.head:
        data_sources_scope = os.environ.get("GRAFANA_DATA_SOURCES_SCOPE")
        if data_sources_scope == GRAFANA_DATA_SOURCES_SCOPE_WORKSPACE:
            # needed for only discover the data sources of workspace
            if args.command == "start":
                start_pull_server(args.head)
            elif args.command == "stop":
                stop_pull_server()


if __name__ == "__main__":
    main()
