import argparse
import os
from shlex import quote

from cloudtik.core._private.utils import run_system_command, subscribe_runtime_config
from cloudtik.runtime.kafka.utils import update_configurations, _get_zookeeper_connect


def main():
    parser = argparse.ArgumentParser(
        description="Configuring runtime.")
    parser.add_argument('--head', action='store_true', default=False,
                        help='Configuring for head node.')
    args = parser.parse_args()

    this_dir = os.path.dirname(__file__)
    shell_path = os.path.join(this_dir, "configure-kafka.sh")
    cmds = [
        "bash",
        quote(shell_path),
    ]

    if args.head:
        cmds += ["--head"]

    # We either get the zookeeper_connect from kafka runtime config
    # or we get it from redis published zookeeper uri (or make it by nodes info?)
    if not args.head:
        runtime_config = subscribe_runtime_config()
        zookeeper_connect = _get_zookeeper_connect(runtime_config)
        if zookeeper_connect is None:
            raise RuntimeError("Not able to get zookeeper connect.")

        cmds += ["--zookeeper_connect={}".format(quote(zookeeper_connect))]

    final_cmd = " ".join(cmds)
    run_system_command(final_cmd)

    if not args.head:
        # Update kafka configuration from runtime config
        update_configurations()


if __name__ == "__main__":
    main()
