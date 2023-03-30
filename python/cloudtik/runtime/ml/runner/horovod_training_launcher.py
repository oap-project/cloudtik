import logging

from cloudtik.runtime.ml.runner.distributed_training_launcher import DistributedTrainingLauncher

logger = logging.getLogger(__name__)


class HorovodTrainingLauncher(DistributedTrainingLauncher):
    r"""
     Launcher for distributed training with Horovod
     """

    def __init__(self, args):
        super().__init__(args)

    def run(self, command):
        args = self.args

        # Run with Horovod
        from horovod.runner import _HorovodArgs
        from horovod.runner.launch import _run

        num_proc = args.nnodes * args.nproc_per_node
        mpi_args = args.more_mpi_params
        use_mpi = True

        hargs = _HorovodArgs()

        hargs.num_proc = num_proc
        if args.hosts:
            host_slots = ["{}:{}".format(host, args.nproc_per_node) for host in self.hosts]
            host_slots_list = ",".join(host_slots)
            hargs.hosts = host_slots_list
        else:
            hargs.hostfile = args.hostfile
        hargs.mpi_args = mpi_args
        hargs.use_mpi = use_mpi
        hargs.command = command

        return _run(hargs)
