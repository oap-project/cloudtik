import argparse
import os

from cloudtik.runtime.haproxy.utils import start_pull_server, stop_pull_server, \
    HAPROXY_CONFIG_MODE_DYNAMIC


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

    high_availability = os.environ.get("HAPROXY_HIGH_AVAILABILITY")
    if high_availability == "true" or args.head:
        config_mode = os.environ.get("HAPROXY_CONFIG_MODE")
        if config_mode == HAPROXY_CONFIG_MODE_DYNAMIC:
            # needed pull server only for dynamic backend
            if args.command == "start":
                start_pull_server(args.head)
            elif args.command == "stop":
                stop_pull_server()


if __name__ == "__main__":
    main()
