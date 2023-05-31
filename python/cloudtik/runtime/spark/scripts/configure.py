import argparse

from cloudtik.runtime.spark.utils import update_spark_configurations


def main():
    parser = argparse.ArgumentParser(
        description="Configuring runtime.")
    parser.add_argument('--head', action='store_true', default=False,
                        help='Configuring for head node.')
    args = parser.parse_args()
    if args.head:
        # Update spark configuration from cluster config file
        update_spark_configurations()


if __name__ == "__main__":
    main()
