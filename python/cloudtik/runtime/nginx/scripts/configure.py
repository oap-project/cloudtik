import argparse
import os

from cloudtik.runtime.nginx.utils import configure_backend


def main():
    parser = argparse.ArgumentParser(
        description="Configuring runtime.")
    parser.add_argument('--head', action='store_true', default=False,
                        help='Configuring for head node.')
    args = parser.parse_args()

    high_availability = os.environ.get("NGINX_HIGH_AVAILABILITY")
    if high_availability == "true" or args.head:
        configure_backend(args.head)


if __name__ == "__main__":
    main()
