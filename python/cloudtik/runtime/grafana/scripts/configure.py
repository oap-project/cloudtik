import argparse

from cloudtik.core._private.runtime_utils import get_runtime_bool
from cloudtik.runtime.grafana.utils import configure_data_sources


def main():
    parser = argparse.ArgumentParser(
        description="Configuring runtime.")
    parser.add_argument('--head', action='store_true', default=False,
                        help='Configuring for head node.')
    args = parser.parse_args()

    high_availability = get_runtime_bool("GRAFANA_HIGH_AVAILABILITY")
    if high_availability or args.head:
        configure_data_sources(args.head)


if __name__ == "__main__":
    main()
