import copy
import logging
import os
import sys

from cloudtik.runtime.ml.runner.cpu import utils
from cloudtik.runtime.ml.runner.cpu.launcher import Launcher

logger = logging.getLogger(__name__)

# Threshold for large cluster MPI issues:
_LARGE_CLUSTER_THRESHOLD = 64


class DistributedTrainingLauncher(Launcher):
    r"""
     Launcher for distributed training with MPI launcher
     """

    def __init__(self, args):
        super().__init__(args)
        self.hosts = None

    def launch(self):
        """
        Set ENVs and launch MPI process for distributed training.
        """
        self.verify_hosts()
        self.set_master()
        self.set_environment()
        command = self.get_command_to_run()
        self.run(command)

    def verify_hosts(self):
        args = self.args
        # There are 3 cases
        # 1. local single node training (args.nnodes <= 1, no args.hosts and no args.hostfile)
        # 2. remote single node training (args.nnodes <= 1, args.hosts or args.hostfile)
        # 3. remote multi-node training (args.nnodes == 0 or args.nnodes > 1, args.hosts or args.hostfile)
        if not args.hosts and not os.path.exists(args.hostfile):
            if args.nnodes > 1:
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
            if args.nnodes == 0:
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
        if args.nproc_per_node == 0:
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

    def run(self, command):
        args = self.args

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
