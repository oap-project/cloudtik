import logging
import os

from cloudtik.runtime.ai.runner.launcher import Launcher

logger = logging.getLogger(__name__)


class DistributedTrainingLauncher(Launcher):
    r"""
     Launcher for distributed training
     """

    def __init__(self, args, distributor):
        super().__init__(args, distributor)

    def launch(self):
        """
        Set ENVs and launch process for distributed training by calling run with command
        """
        self.set_environment()
        self.run()

    def set_environment(self):
        # we default to run single proc per node if not specified
        self.distributor.resolve()
        self.set_master()

    def get_master_addr(self, args):
        if self.distributor.distributed:
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

