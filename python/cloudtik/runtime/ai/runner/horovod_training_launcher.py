import logging
import sys

from cloudtik.runtime.ai.runner.distributed_training_launcher import DistributedTrainingLauncher
from cloudtik.runtime.ai.runner.util.utils import is_python_program

logger = logging.getLogger(__name__)


class HorovodTrainingLauncher(DistributedTrainingLauncher):
    r"""
     Launcher for distributed training with Horovod
     """

    def __init__(self, args, distributor):
        super().__init__(args, distributor)

    def get_command_to_run(self):
        args = self.args
        cmd = []
        self.with_python_command(cmd)
        cmd.extend(args.command)
        return cmd

    def run(self):
        args = self.args

        # Run with Horovod
        from horovod.runner import _HorovodArgs
        from horovod.runner.launch import _run

        hargs = _HorovodArgs()

        hargs.num_proc = self.distributor.num_proc
        hargs.hosts = self.distributor.hosts_slots_str

        hargs.mpi_args = args.mpi_args
        hargs.use_mpi = args.use_mpi
        hargs.use_gloo = args.use_gloo
        hargs.nics = args.nics
        hargs.output_filename = args.output_filename
        hargs.verbose = args.verbose

        if args.run_func:
            hargs.run_func = args.run_func
            hargs.executable = args.executable
        else:
            command = self.get_command_to_run()
            hargs.command = command

        return _run(hargs)
