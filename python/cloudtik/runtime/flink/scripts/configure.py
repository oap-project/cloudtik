import argparse

from cloudtik.runtime.flink.utils import update_flink_configurations


def main():
    parser = argparse.ArgumentParser(
        description="Configuring runtime.")
    parser.add_argument('--head', action='store_true', default=False,
                        help='Configuring for head node.')
    args = parser.parse_args()
    if args.head:
        # Update flink configuration from cluster config file
        update_flink_configurations()


if __name__ == "__main__":
    main()
