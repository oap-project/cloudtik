import argparse

from cloudtik.core._private.constants import CLOUDTIK_RUNTIME_ENV_QUORUM_JOIN
from cloudtik.core._private.runtime_utils import subscribe_nodes_info, get_runtime_value
from cloudtik.core.tags import QUORUM_JOIN_STATUS_INIT
from cloudtik.runtime.zookeeper.utils import request_to_join_cluster


def start_service():
    quorum_join = get_runtime_value(CLOUDTIK_RUNTIME_ENV_QUORUM_JOIN)
    if quorum_join and quorum_join == QUORUM_JOIN_STATUS_INIT:
        # request to join a exiting cluster
        nodes_info = subscribe_nodes_info()
        request_to_join_cluster(nodes_info)


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

    if not args.head and args.command == "start":
        start_service()


if __name__ == "__main__":
    main()
