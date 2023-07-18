import copy
import logging
import os

from cloudtik.runtime.ai.runner import get_cloudtik_rsh
from cloudtik.runtime.ai.runner.mpi import mpi_utils
from cloudtik.runtime.ai.runner.distributed_launcher import DistributedLauncher
from cloudtik.runtime.ai.runner.util import env as env_utils, safe_shell_exec
from cloudtik.runtime.ai.runner.util.utils import _run_func

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


class MPILauncherArgs(object):
    def __init__(self):
        self.verbose = None
        self.mpi_args = None
        self.nics = None

        self.env = None
        self.stdout = None
        self.stderr = None


class MPILauncher(DistributedLauncher):
    r"""
     Launcher for distributed training with MPI launcher
     """

    def __init__(self, args, distributor):
        super().__init__(args, distributor)
        self.margs = MPILauncherArgs()
        self._init_launcher_args(self.margs)

    def run(self):
        args = self.args

        def run_command(command):
            self._run_command(command)

        if args.func:
            func = self.wrap_func()
            num_proc = self.distributor.num_proc
            return _run_func(
                func, num_proc, run_command,
                executable=args.executable,
                verbose=args.verbose)
        else:
            command = self.get_command_to_run()
            # TODO: handle NICs included for MPI
            run_command(command)
            return None

    def _run_command(self, command):
        margs = self.margs

        command_str = self.get_command_str(command)
        env = self._get_env(margs)
        run_kwargs = self._get_kwargs(
            margs, ["stdout", "stderr"])
        if mpi_utils.is_impi_or_mpich():
            self._run_command_impi(
                margs, command_str, env, **run_kwargs)
        else:
            self._run_command_openmpi(
                margs, command_str, env, **run_kwargs)

    def _run_command_openmpi(
            self, margs, command, env,
            stdout=None, stderr=None):
        # default to use OpenMPI to launch
        _OMPI_FLAGS = ['-mca pml ob1', '-mca btl ^openib']
        _NO_BINDING_ARGS = ['-bind-to none', '-map-by slot']

        num_proc = self.distributor.num_proc

        mpi_impl_flags = _OMPI_FLAGS
        if self.distributor.distributed_with_hosts and len(
                self.distributor.hosts) >= _LARGE_CLUSTER_THRESHOLD:
            mpi_impl_flags.append('-mca plm_rsh_no_tree_spawn true')
            mpi_impl_flags.append(
                '-mca plm_rsh_num_concurrent {}'.format(len(self.distributor.hosts)))

        # if user does not specify any hosts, mpirun by default uses local host.
        # There is no need to specify localhost.
        if self.distributor.distributed_with_hosts:
            host_slots_str = self.distributor.hosts_slots_str
            hosts_arg = '-{opt} {hosts}'.format(opt='H',
                                                hosts=host_slots_str)
        else:
            hosts_arg = ''
        binding_args = ' '.join(_NO_BINDING_ARGS)
        basic_args = '--allow-run-as-root --tag-output'
        env_list = ""
        if env:
            # Shall we pass on all the local environment?
            # env = os.environ.copy()
            env_list = ' '.join(
                '-x %s' % key for key in sorted(env.keys()) if env_utils.is_exportable(key))

        extra_mpi_args = margs.mpi_args
        if self.distributor.distributed_with_hosts and (
                not extra_mpi_args or "-mca plm_rsh_agent" not in extra_mpi_args):
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

        self.run_mpi_command(
            mpirun_command, env,
            stdout=stdout, stderr=stderr)

    def _run_command_impi(
            self, margs, command, env,
            stdout=None, stderr=None):
        # make sure that for IMPI cases, all the nodes have the same slots
        self.distributor.check_same_slots()

        num_proc = self.distributor.num_proc
        nproc_per_node = self.distributor.nproc_per_node

        cmd = ['mpirun']
        mpi_config = "-l -np {} -ppn {}".format(
            num_proc, nproc_per_node)
        if env:
            genvs = [f"-genv {k}={v}" for k, v in env.items()]
            mpi_config += " {}".format(' '.join(genvs))
        if margs.mpi_args:
            mpi_config += " {}".format(margs.mpi_args)

        if self.distributor.distributed_with_hosts:
            mpi_config += " -hosts {}".format(self.distributor.hosts_str)
            # Unified to pass by hosts instead of hostfile
            # mpi_config += " -hostfile {}".format(hostfile)

        # only add this for remote training
        if self.distributor.distributed_with_hosts:
            if "-launcher-exec" not in mpi_config:
                mpi_config += (
                    ' -launcher rsh -launcher-exec "{launcher_exec}"'.format(
                        launcher_exec=get_cloudtik_rsh()))

        cmd.extend(mpi_config.split())
        mpi_command = " ".join(cmd)

        mpirun_command = "{mpi_command} {command}".format(
            mpi_command=mpi_command,
            command=command
        )

        self.run_mpi_command(
            mpirun_command, env,
            stdout=stdout, stderr=stderr)

    def run_mpi_command(
            self, mpirun_command, env,
            stdout=None, stderr=None):
        # TODO: Do we need to add the os.environ when run?
        # os_env = os.environ.copy()

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
        if self.args.func:
            # function mode
            exit_code = safe_shell_exec.execute(
                mpirun_command, env=env, stdout=stdout, stderr=stderr)
            if exit_code != 0:
                raise RuntimeError(
                    "mpirun failed with exit code {exit_code}".format(exit_code=exit_code))
        else:
            # os.execve execute a new program, replacing the current process; they do not return.
            os.execve('/bin/sh', ['/bin/sh', '-c', mpirun_command], env)

            # do we really need execve or use Popen
            # process = subprocess.Popen(mpirun_command, env=os.environ, shell=True)
            # process.wait()
