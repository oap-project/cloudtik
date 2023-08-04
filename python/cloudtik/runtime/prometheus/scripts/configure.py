import argparse
import os

from cloudtik.runtime.prometheus.utils import configure_scrape


def main():
    parser = argparse.ArgumentParser(
        description="Configuring runtime.")
    parser.add_argument('--head', action='store_true', default=False,
                        help='Configuring for head node.')
    args = parser.parse_args()

    high_availability = os.environ.get("PROMETHEUS_HIGH_AVAILABILITY")
    if high_availability == "true" or args.head:
        configure_scrape(args.head)


if __name__ == "__main__":
    main()
