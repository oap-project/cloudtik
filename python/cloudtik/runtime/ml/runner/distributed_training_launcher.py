import logging
import os
import sys

from cloudtik.runtime.ml.runner.launcher import Launcher

logger = logging.getLogger(__name__)


class DistributedTrainingLauncher(Launcher):
    r"""
     Launcher for distributed training
     """

    def __init__(self, args):
        super().__init__(args)
        self.hosts = None

    def launch(self):
        """
        Set ENVs and launch process for distributed training by calling run with command
        """
        self.verify_hosts()
        self.set_master()
        self.set_environment()
        self.run()

    def verify_hosts(self):
        args = self.args
        # There are 3 cases
        # 1. local single node training (args.nnodes <= 1, no args.hosts and no args.hostfile)
        # 2. remote single node training (args.nnodes <= 1, args.hosts or args.hostfile)
        # 3. remote multi-node training (args.nnodes == 0 or args.nnodes > 1, args.hosts or args.hostfile)
        if not args.hosts and not os.path.exists(args.hostfile):
            if args.nnodes is not None and args.nnodes > 1:
                raise ValueError("hosts or hostfile is necessary when you use multi-node distributed training,")
            # local single node training
            if not args.master_addr:
                args.master_addr = "127.0.0.1"
            args.nnodes = 1
        else:
            # either hosts or hostfile specified, remote training
            if args.hostfile:
                host_list = []
                with open(args.hostfile) as f:
                    for line in f:
                        line = line.strip().strip("\n")
                        host_list.append(line)
                if not host_list:
                    raise ValueError("No IP listed in hostfile.")
            else:
                # hosts specified
                host_list = args.hosts.split(',')

            self.hosts = host_list
            host_number = len(host_list)
            if not args.nnodes:
                args.nnodes = host_number
            elif args.nnodes > host_number:
                raise ValueError("nnodes {} cannot be greater than the number of hosts {}.".format(
                    args.nnodes, host_number
                ))
            args.master_addr = host_list[0]

    def set_master(self):
        args = self.args
        # set distributed related environmental variables
        self.set_env("MASTER_ADDR", args.master_addr)
        self.set_env("MASTER_PORT", str(args.master_port))

    def set_environment(self):
        args = self.args
        # Default we run single instance per node
        if not args.nproc_per_node:
            args.nproc_per_node = 1

    def get_command_to_run(self):
        args = self.args
        cmd = []
        with_python = not args.no_python
        if with_python:
            cmd.append(sys.executable)
            cmd.append("-u")
        if args.module:
            cmd.append("-m")
        cmd.append(args.program)
        cmd.extend(args.program_args)
        cmd_s = " ".join(cmd)
        if args.log_path:
            log_name = args.log_file_prefix + ".log"
            log_file = os.path.join(args.log_path, log_name)
            cmd_s = "{} 2>&1 | tee {}".format(cmd_s, log_file)
        logger.info(cmd_s)
        return cmd_s

    def run(self):
        command = self.get_command_to_run()
        logger.info(command)
