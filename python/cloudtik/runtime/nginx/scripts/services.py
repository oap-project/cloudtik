import argparse

from cloudtik.core._private.runtime_utils import get_runtime_value, get_runtime_bool
from cloudtik.runtime.nginx.utils \
    import start_pull_server, stop_pull_server, NGINX_APP_MODE_API_GATEWAY, \
    NGINX_CONFIG_MODE_DNS, NGINX_APP_MODE_LOAD_BALANCER, NGINX_CONFIG_MODE_DYNAMIC


def _need_pull_server():
    app_mode = get_runtime_value("NGINX_APP_MODE")
    config_mode = get_runtime_value("NGINX_CONFIG_MODE")
    if (app_mode == NGINX_APP_MODE_API_GATEWAY
        and (config_mode == NGINX_CONFIG_MODE_DNS or
             config_mode == NGINX_CONFIG_MODE_DYNAMIC)) or (
            app_mode == NGINX_APP_MODE_LOAD_BALANCER and
            config_mode == NGINX_CONFIG_MODE_DYNAMIC
    ):
        return True
    else:
        return False


def start_service(head):
    if _need_pull_server():
        start_pull_server(head)


def stop_service():
    if _need_pull_server():
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

    high_availability = get_runtime_bool("NGINX_HIGH_AVAILABILITY")
    if high_availability or args.head:
        if args.command == "start":
            start_service(args.head)
        elif args.command == "stop":
            stop_service()


if __name__ == "__main__":
    main()
