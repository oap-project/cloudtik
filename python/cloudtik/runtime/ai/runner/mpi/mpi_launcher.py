import copy
import logging
import os
import subprocess

from cloudtik.runtime.ai.runner import get_cloudtik_rsh
from cloudtik.runtime.ai.runner.util import utils
from cloudtik.runtime.ai.runner.distributed_launcher import DistributedLauncher

logger = logging.getLogger(__name__)

# Threshold for large cluster MPI issues:
_LARGE_CLUSTER_THRESHOLD = 64


def add_mpi_params(parser):
    group = parser.add_argument_group("MPI Parameters")

    # mpi control
    group.add_argument(
        "--mpi-args", "--mpi_args",
        action='store', dest='mpi_args', default="", type=str,
        help="User can pass more parameters for mpirun")


class MPILauncher(DistributedLauncher):
    r"""
     Launcher for distributed training with MPI launcher
     """

    def __init__(self, args, distributor):
        super().__init__(args, distributor)

    def run(self):
        command = self.get_command_to_run()
        if utils.is_impi_or_mpich():
            self._run_command_impi(command)
        else:
            self._run_command_openmpi(command)

    def _run_command_openmpi(self, command):
        args = self.args
        # default to use OpenMPI to launch
        _OMPI_FLAGS = ['-mca pml ob1', '-mca btl ^openib']
        _NO_BINDING_ARGS = ['-bind-to none', '-map-by slot']

        num_proc = self.distributor.num_proc

        mpi_impl_flags = _OMPI_FLAGS
        if self.distributor.distributed and len(self.distributor.hosts) >= _LARGE_CLUSTER_THRESHOLD:
            mpi_impl_flags.append('-mca plm_rsh_no_tree_spawn true')
            mpi_impl_flags.append(
                '-mca plm_rsh_num_concurrent {}'.format(len(self.distributor.hosts)))

        # if user does not specify any hosts, mpirun by default uses local host.
        # There is no need to specify localhost.
        if self.distributor.distributed:
            host_slots_str = self.distributor.hosts_slots_str
            hosts_arg = '-{opt} {hosts}'.format(opt='H',
                                                hosts=host_slots_str)
        else:
            hosts_arg = ''
        binding_args = ' '.join(_NO_BINDING_ARGS)
        basic_args = '--allow-run-as-root --tag-output'
        env_list = ""
        env = self.environ_set
        if env:
            # Shall we pass on all the local environment?
            # env = os.environ.copy()
            env_list = ' '.join(
                '-x %s' % key for key in sorted(env.keys()) if utils.is_exportable(key))

        extra_mpi_args = args.mpi_args
        if self.distributor.distributed and (not extra_mpi_args or "-mca plm_rsh_agent" not in extra_mpi_args):
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

    def _run_command_impi(self, command):
        args = self.args

        # make sure that for IMPI cases, all the nodes have the same slots
        self.distributor.validate_same_slots()

        num_proc = self.distributor.num_proc
        nproc_per_node = self.distributor.nproc_per_node

        cmd = ['mpirun']
        mpi_config = "-l -np {} -ppn {}".format(
            num_proc, nproc_per_node)
        if self.environ_set:
            genvs = [f"-genv {k}={v}" for k, v in self.environ_set.items()]
            mpi_config += " {}".format(' '.join(genvs))
        if args.mpi_args:
            mpi_config += " {}".format(args.mpi_args)

        if self.distributor.distributed:
            mpi_config += " -hosts {}".format(self.distributor.hosts_str)
            # Unified to pass by hosts instead of hostfile
            # mpi_config += " -hostfile {}".format(hostfile)

        # only add this for remote training
        if self.distributor.distributed:
            if "-launcher-exec" not in mpi_config:
                mpi_config += (
                    ' -launcher rsh -launcher-exec "{launcher_exec}"'.format(
                        launcher_exec=get_cloudtik_rsh()))

        cmd.extend(mpi_config.split())
        mpi_command = " ".join(cmd)

        # TODO: handle log to file
        final_command = "{mpi_command} {command}".format(
            mpi_command=mpi_command,
            command=command
        )
        logger.info("Final command run: {}".format(final_command))
        process = subprocess.Popen(final_command, env=os.environ, shell=True)
        process.wait()
