import argparse

from cloudtik.core._private.runtime_utils import get_runtime_value
from cloudtik.runtime.haproxy.utils \
    import start_pull_server, stop_pull_server, HAPROXY_CONFIG_MODE_DYNAMIC


def start_service(head):
    config_mode = get_runtime_value("HAPROXY_CONFIG_MODE")
    if config_mode == HAPROXY_CONFIG_MODE_DYNAMIC:
        # needed pull server only for dynamic backend
        start_pull_server(head)


def stop_service():
    config_mode = get_runtime_value("HAPROXY_CONFIG_MODE")
    if config_mode == HAPROXY_CONFIG_MODE_DYNAMIC:
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

    high_availability = get_runtime_value("HAPROXY_HIGH_AVAILABILITY")
    if high_availability == "true" or args.head:
        if args.command == "start":
            start_service(args.head)
        elif args.command == "stop":
            stop_service()


if __name__ == "__main__":
    main()
