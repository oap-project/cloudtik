import argparse

from cloudtik.core._private.runtime_utils import subscribe_runtime_config
from cloudtik.core._private.utils import load_head_cluster_config, RUNTIME_CONFIG_KEY
from cloudtik.runtime.trino.utils import configure_connectors


def main():
    parser = argparse.ArgumentParser(
        description="Configuring runtime.")
    parser.add_argument('--head', action='store_true', default=False,
                        help='Configuring for head node.')
    args = parser.parse_args()

    # Configure other connectors
    if args.head:
        runtime_config = load_head_cluster_config().get(
            RUNTIME_CONFIG_KEY)
    else:
        runtime_config = subscribe_runtime_config()
    configure_connectors(runtime_config)


if __name__ == "__main__":
    main()
