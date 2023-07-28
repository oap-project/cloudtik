import argparse

from cloudtik.core._private.runtime_utils import subscribe_nodes_info
from cloudtik.runtime.etcd.utils import configure_initial_cluster


def main():
    parser = argparse.ArgumentParser(
        description="Configuring runtime.")
    parser.add_argument('--head', action='store_true', default=False,
                        help='Configuring for head node.')
    args = parser.parse_args()

    # Bootstrap the initial cluster
    if not args.head:
        nodes_info = subscribe_nodes_info()
        configure_initial_cluster(nodes_info)


if __name__ == "__main__":
    main()
