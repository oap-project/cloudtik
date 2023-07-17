from cloudtik.runtime.ai.runner.cpu.cpu_launcher import add_cpu_launcher_params
from cloudtik.runtime.ai.runner.cpu.distributed_launcher import add_distributed_cpu_launcher_params
from cloudtik.runtime.ai.runner.cpu.local_launcher import add_local_cpu_launcher_params, add_auto_ipex_params
from cloudtik.runtime.ai.runner.distributed_launcher import add_distributed_params
from cloudtik.runtime.ai.runner.horovod.horovod_launcher import add_horovod_params
from cloudtik.runtime.ai.runner.mpi.mpi_launcher import add_mpi_params


def add_launcher_params(parser):
    add_cpu_launcher_params(parser)
    add_local_cpu_launcher_params(parser)
    add_auto_ipex_params(parser)

    add_distributed_params(parser)
    add_distributed_cpu_launcher_params(parser)

    add_mpi_params(parser)
    add_horovod_params(parser)

    return parser


def create_launcher(launcher_name, args, distributor):
    if launcher_name == "local":
        from cloudtik.runtime.ai.runner.cpu.local_launcher \
            import LocalCPULauncher
        launcher = LocalCPULauncher(args, distributor)
    elif launcher_name == "distributed":
        from cloudtik.runtime.ai.runner.cpu.distributed_launcher \
            import DistributedCPULauncher
        launcher = DistributedCPULauncher(args, distributor)
    elif launcher_name == "mpi":
        from cloudtik.runtime.ai.runner.mpi.mpi_launcher \
            import MPILauncher
        launcher = MPILauncher(args, distributor)
    elif launcher_name == "horovod":
        from cloudtik.runtime.ai.runner.horovod.horovod_launcher \
            import HorovodLauncher
        launcher = HorovodLauncher(args, distributor)
    elif launcher_name == "horovod.spark":
        from cloudtik.runtime.ai.runner.horovod.horovod_spark_launcher \
            import HorovodSparkLauncher
        launcher = HorovodSparkLauncher(args, distributor)
    elif launcher_name == "horovod.ray":
        from cloudtik.runtime.ai.runner.horovod.horovod_ray_launcher \
            import HorovodRayLauncher
        launcher = HorovodRayLauncher(args, distributor)
    else:
        raise ValueError("Launcher type with name {} is not supported.".format(launcher_name))
    return launcher
