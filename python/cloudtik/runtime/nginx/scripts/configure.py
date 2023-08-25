import argparse

from cloudtik.core._private.runtime_utils import get_runtime_bool
from cloudtik.runtime.nginx.utils import configure_backend


def main():
    parser = argparse.ArgumentParser(
        description="Configuring runtime.")
    parser.add_argument('--head', action='store_true', default=False,
                        help='Configuring for head node.')
    args = parser.parse_args()

    high_availability = get_runtime_bool("NGINX_HIGH_AVAILABILITY")
    if high_availability or args.head:
        configure_backend(args.head)


if __name__ == "__main__":
    main()
