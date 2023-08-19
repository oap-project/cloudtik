import argparse

from cloudtik.core._private.util.resolv_conf import update_resolv_conf


def main():
    parser = argparse.ArgumentParser(
        description="Update the /etc/resolv.conf with a list of name servers.")
    parser.add_argument(
        "name_servers",
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    if args.name_servers:
        update_resolv_conf(name_servers=args.name_servers)


if __name__ == "__main__":
    main()
