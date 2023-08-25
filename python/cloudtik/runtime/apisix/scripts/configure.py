import argparse

from cloudtik.runtime.apisix.utils import update_configurations


def main():
    parser = argparse.ArgumentParser(
        description="Configuring runtime.")
    parser.add_argument('--head', action='store_true', default=False,
                        help='Configuring for head node.')
    args = parser.parse_args()

    # Update configuration from runtime config
    update_configurations(args.head)


if __name__ == "__main__":
    main()
