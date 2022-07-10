import sys

""" This script is meant to be run from a pod in the same Kubernetes namespace
as your cluster.
"""


def main():
    print("Success!")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
