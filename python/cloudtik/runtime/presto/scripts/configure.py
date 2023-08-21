import argparse

from cloudtik.core._private.runtime_utils import get_runtime_config_from_node
from cloudtik.runtime.presto.utils import configure_connectors


def main():
    parser = argparse.ArgumentParser(
        description="Configuring runtime.")
    parser.add_argument('--head', action='store_true', default=False,
                        help='Configuring for head node.')
    args = parser.parse_args()

    # Configure other connectors
    runtime_config = get_runtime_config_from_node(args.head)
    configure_connectors(runtime_config)


if __name__ == "__main__":
    main()
