import logging
import sys

from cloudtik.runtime.ai.runner.distributed_training_launcher import DistributedTrainingLauncher

logger = logging.getLogger(__name__)


class HorovodTrainingLauncher(DistributedTrainingLauncher):
    r"""
     Launcher for distributed training with Horovod
     """

    def __init__(self, args):
        super().__init__(args)

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
        return cmd

    def run(self):
        args = self.args

        # Run with Horovod
        from horovod.runner import _HorovodArgs
        from horovod.runner.launch import _run

        num_proc = args.nnodes * args.nproc_per_node

        hargs = _HorovodArgs()

        hargs.num_proc = num_proc
        if args.hosts:
            host_slots = ["{}:{}".format(host, args.nproc_per_node) for host in self.hosts]
            host_slots_list = ",".join(host_slots)
            hargs.hosts = host_slots_list
        else:
            hargs.hostfile = args.hostfile
        hargs.mpi_args = args.more_mpi_params
        hargs.use_mpi = args.use_mpi
        hargs.use_gloo = args.use_gloo
        hargs.verbose = args.verbose

        if args.run_func:
            hargs.run_func = args.run_func
            hargs.executable = args.executable
        else:
            command = self.get_command_to_run()
            hargs.command = command

        return _run(hargs)
