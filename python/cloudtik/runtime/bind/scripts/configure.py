import argparse

from cloudtik.runtime.bind.utils import configure_upstream


def main():
    parser = argparse.ArgumentParser(
        description="Configuring runtime.")
    parser.add_argument('--head', action='store_true', default=False,
                        help='Configuring for head node.')
    args = parser.parse_args()

    configure_upstream(args.head)


if __name__ == "__main__":
    main()
