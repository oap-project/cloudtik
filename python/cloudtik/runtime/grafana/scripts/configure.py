import argparse
import os

from cloudtik.runtime.grafana.utils import configure_data_sources


def main():
    parser = argparse.ArgumentParser(
        description="Configuring runtime.")
    parser.add_argument('--head', action='store_true', default=False,
                        help='Configuring for head node.')
    args = parser.parse_args()

    high_availability = os.environ.get("GRAFANA_HIGH_AVAILABILITY")
    if high_availability == "true" or args.head:
        configure_data_sources(args.head)


if __name__ == "__main__":
    main()
