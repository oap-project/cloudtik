import argparse

from cloudtik.runtime.consul.utils import configure_join, configure_services


def main():
    parser = argparse.ArgumentParser(
        description="Configuring runtime.")
    parser.add_argument('--head', action='store_true', default=False,
                        help='Configuring for head node.')
    args = parser.parse_args()

    # configure join
    configure_join(args.head)

    # Configure the Consul services
    configure_services(args.head)


if __name__ == "__main__":
    main()
