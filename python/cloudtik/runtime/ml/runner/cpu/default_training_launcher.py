import copy
import logging
import os

from cloudtik.runtime.ml.runner import utils
from cloudtik.runtime.ml.runner.cpu.launcher import CPULauncher
from cloudtik.runtime.ml.runner.distributed_training_launcher import DistributedTrainingLauncher

logger = logging.getLogger(__name__)

# Threshold for large cluster MPI issues:
_LARGE_CLUSTER_THRESHOLD = 64


class DefaultTrainingLauncher(CPULauncher, DistributedTrainingLauncher):
    r"""
     Launcher for distributed training with MPI launcher
     """

    def __init__(self, args):
        super().__init__(args)

    def run(self):
        args = self.args
        command = self.get_command_to_run()

        # default to use OpenMPI to launch
        _OMPI_FLAGS = ['-mca pml ob1', '-mca btl ^openib']
        _NO_BINDING_ARGS = ['-bind-to none', '-map-by slot']

        num_proc = args.nnodes * args.nproc_per_node

        mpi_impl_flags = _OMPI_FLAGS
        if self.hosts and len(self.hosts) >= _LARGE_CLUSTER_THRESHOLD:
            mpi_impl_flags.append('-mca plm_rsh_no_tree_spawn true')
            mpi_impl_flags.append(
                '-mca plm_rsh_num_concurrent {}'.format(len(self.hosts)))

        # if user does not specify any hosts, mpirun by default uses local host.
        # There is no need to specify localhost.
        if self.hosts:
            host_slots = ["{}:{}".format(host, args.nproc_per_node) for host in self.hosts]
            host_slots_str = ",".join(host_slots)
            hosts_arg = '-{opt} {hosts}'.format(opt='H',
                                                hosts=host_slots_str)
        else:
            hosts_arg = ''
        binding_args = ' '.join(_NO_BINDING_ARGS)
        basic_args = '--allow-run-as-root --tag-output'
        env = os.environ.copy()
        env_list = ' '.join(
            '-x %s' % key for key in sorted(env.keys()) if utils.is_exportable(key))

        def get_cloudtik_rsh():
            cloudtik_ml_home = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            return os.path.join(cloudtik_ml_home, "scripts", "cloudtik-rsh.sh")

        extra_mpi_args = args.more_mpi_params
        if self.hosts and (not extra_mpi_args or "-mca plm_rsh_agent" not in extra_mpi_args):
            extra_mpi_args = (
                '{extra_mpi_args} -mca plm_rsh_agent "{rsh_agent}"'
                .format(extra_mpi_args=extra_mpi_args if extra_mpi_args else '',
                        rsh_agent=get_cloudtik_rsh()))

        # Pass all the env variables to the mpirun command.
        mpirun_command = (
            'mpirun {basic_args} '
            '-np {num_proc} '
            '{hosts_arg} '
            '{binding_args} '
            '{mpi_args} '
            '{env} {extra_mpi_args} {command}'  # expect a lot of environment variables
            .format(basic_args=basic_args,
                    num_proc=num_proc,
                    hosts_arg=hosts_arg,
                    binding_args=binding_args,
                    mpi_args=' '.join(mpi_impl_flags),
                    env=env_list,
                    extra_mpi_args=extra_mpi_args if extra_mpi_args else '',
                    command=command)
        )

        # we need the driver's PATH and PYTHONPATH in env to run mpirun,
        # env for mpirun is different to env encoded in mpirun_command
        for var in ['PATH', 'PYTHONPATH']:
            if var not in env and var in os.environ:
                # copy env so we do not leak env modifications
                env = copy.copy(env)
                # copy var over from os.environ
                env[var] = os.environ[var]

        logger.info("Final command run: {}".format(mpirun_command))

        # Execute the mpirun command.
        os.execve('/bin/sh', ['/bin/sh', '-c', mpirun_command], env)
