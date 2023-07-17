import logging
import os
from shlex import quote

from cloudtik.runtime.ai.runner.launcher import Launcher

logger = logging.getLogger(__name__)


def add_distributed_params(parser):
    group = parser.add_argument_group("Distributed Parameters")
    group.add_argument(
        "--nnodes",
        type=int, default=0,
        help="The number of nodes to use for distributed training")
    group.add_argument(
        "--nproc-per-node", "--nproc_per_node",
        action='store', type=int, default=0,
        help="The number of processes to launch on each node")
    group.add_argument(
        "--hosts",
        default="", type=str,
        help="List of hosts separated with comma for launching tasks. "
             "When hosts is specified, it implies distributed training. "
             "node address which should be either the IP address"
             "or the hostname with or without slots.")
    group.add_argument(
        "--hostfile",
        default="", type=str,
        help="Hostfile is necessary for multi-node multi-proc "
             "training. hostfile includes the node address list "
             "node address which should be either the IP address"
             "or the hostname with or without slots.")

    group.add_argument(
        "--master-addr", "--master_addr",
        action='store', default="127.0.0.1", type=str,
        help="Master node (rank 0)'s address, should be either "
             "the IP address or the hostname of node 0, for "
             "single node multi-proc training, the "
             "--master_addr can simply be 127.0.0.1")
    group.add_argument(
        "--master-port", "--master_port",
        action='store', default=29500, type=int,
        help="Master node (rank 0)'s free port that needs to "
             "be used for communication during distributed "
             "training")


class DistributedLauncher(Launcher):
    r"""
     Launcher for distributed training
     """

    def __init__(self, args, distributor):
        super().__init__(args, distributor)

    def setup(self):
        self.distributor.check_distributed_with_hosts()
        self.set_master()

    def get_master_addr(self, args):
        if self.distributor.distributed_with_hosts:
            if args.master_addr and args.master_addr != "127.0.0.1":
                return args.master_addr
            return self.distributor.hosts[0]["ip"]
        else:
            if args.master_addr:
                return args.master_addr
            return "127.0.0.1"

    def set_master(self):
        args = self.args
        args.master_addr = self.get_master_addr(args)

        # set distributed related environmental variables
        # This is only necessary for pytorch based distributed training
        # But other distributed launcher such as Horovod may also need this
        self.set_env("MASTER_ADDR", args.master_addr)
        self.set_env("MASTER_PORT", str(args.master_port))

    def get_command_to_run(self):
        args = self.args
        cmd = []
        self.with_python_command(cmd)
        cmd.extend(args.command)
        return self.get_command_str(cmd)

    def get_command_str(self, cmd):
        args = self.args
        cmd_s = ' '.join(quote(par) for par in cmd)
        if args.log_dir:
            log_name = args.log_file_prefix + ".log"
            log_file = os.path.join(args.log_dir, log_name)
            cmd_s = "{} 2>&1 | tee {}".format(cmd_s, log_file)
        logger.info(cmd_s)
        return cmd_s

    def run(self):
        command = self.get_command_to_run()
        logger.info(command)

    def finalize(self):
        pass
