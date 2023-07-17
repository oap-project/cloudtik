import os
from shlex import quote

CLOUDTIK_COMMAND_PREFIX = 'cloudtik head exec'


def get_cloudtik_exec(local_command, host):
    final_command = quote(local_command)
    return f'{CLOUDTIK_COMMAND_PREFIX} {final_command} --node-ip={host}'


def get_cloudtik_rsh():
    runtime_home = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(runtime_home, "scripts", "cloudtik-rsh.sh")


class _LaunchArgs(object):
    def __init__(self):
        self.command = None
        self.func = None
        self.func_args = ()
        self.func_kwargs = {}
        self.executable = None

        self.num_proc = 0

        # Distributed Launcher
        # If nnodes and nproc_per_node is specified
        # hosts/hostfile can be host address without slots
        # Or you can specify hosts with slots and specify the num_proc
        # all these are handled by Distributor
        self.nnodes = 0
        self.nproc_per_node = 0
        self.hosts = None
        self.hostfile = None

        # Common arguments
        self.launcher = None
        self.launcher_kwargs = {}

        # Shared arguments for some launchers
        self.module = False
        self.no_python = False
        self.log_dir = None
        self.log_file_prefix = None
        self.verbose = None
        self.validate_ld_preload = True

        # Pytorch DDP
        self.master_addr = "127.0.0.1"
        self.master_port = 29500

        # MPI
        self.mpi_args = None

        # CPU Launcher
        self.ncores_per_proc = 0
        self.use_logical_cores = False
        self.use_e_cores = False
        self.memory_allocator = "auto"
        self.omp_runtime = "auto"

        # Local CPU Launcher
        self.process_idx = ""
        self.nodes_list = ""
        self.cores_list = ""
        self.task_manager = "auto"
        self.skip_cross_node_cores = False
        self.latency_mode = False
        self.throughput_mode = False
        self.benchmark = False

        # Distributed CPU Launcher
        self.ccl_worker_count = 4
        self.logical_cores_for_ccl = False

        # Horovod Launcher
        self.use_gloo = None
        self.use_mpi = None
        self.nics = None
        self.output_filename = None


def run(
        func,
        args=(),
        kwargs={},
        num_proc=None,
        nnodes=None,
        nproc_per_node=None,
        hosts=None,
        hostfile=None,
        executable=None,
        launcher=None,
        **launcher_kwargs,
       ):
    """
    Launch a job to run the specified process function and get the return value.

    :param func: The function to be run in job processes. The function return value will
                 be collected as the corresponding process return value.
                 This function must be compatible with pickle.
    :param args: Arguments to pass to `func`.
    :param kwargs: Keyword arguments to pass to `func`.
    :param num_proc: The number of processes for running.
    :param nnodes: The number of nodes. if not specified, use the number of nodes in the hosts
    :param nproc_per_node: The number of process per node.
    :param hosts: List of host names and the number of available slots
                  for running processes on each, of the form: <hostname>:<slots>
                  (e.g.: host1:2,host2:4,host3:1 indicating 2 processes can run on host1,
                  4 on host2, and 1 on host3). If not specified, defaults to using localhost:<num_proc>
    :param hostfile: Path to a host file containing the list of host names and the number of
                     available slots. Each line of the file must be of the form:
                     <hostname> slots=<slots>
    :param executable: Optional executable to run when launching the workers. Defaults to `sys.executable`.
    :param launcher: The launcher to use: local, distributed, mpi, horovod, horovod.spark, horovod.ray.
    :param launcher_kwargs: The additional keyword arguments for launcher.
    :return: Return a list which contains values return by all processes.
             The index of the list corresponds to the rank of each process.
             Returns only the first min_num_proc results, if set.
    """
    from cloudtik.runtime.ai.runner.run import _run

    if hosts is not None and hostfile is not None:
        raise ValueError('Argument hosts and hostfile only allow one provided.')

    if not launcher:
        # default Horovod for running a function if not specified
        launcher = "horovod"

    largs = _LaunchArgs()

    largs.func = func
    largs.func_args = args
    largs.func_kwargs = kwargs
    largs.executable = executable
    largs.num_proc = num_proc
    largs.nnodes = nnodes
    largs.nproc_per_node = nproc_per_node
    largs.hosts = hosts
    largs.hostfile = hostfile
    largs.launcher = launcher
    largs.launcher_kwargs = launcher_kwargs

    # set for launcher args
    for key, value in launcher_kwargs.items():
        setattr(largs, key, value)

    return _run(largs)


def run_command(
        command,
        num_proc=None,
        nnodes=None,
        nproc_per_node=None,
        hosts=None,
        hostfile=None,
        launcher=None,
        **launcher_kwargs,
       ):
    """
    Launch command to run the specified process function and get the return value.

    :param command: The command with arguments to be run in job processes.
    :param num_proc: The number of processes for running.
    :param nnodes: The number of nodes. if not specified, use the number of nodes in the hosts
    :param nproc_per_node: The number of process per node.
    :param hosts: List of host names and the number of available slots
                  for running processes on each, of the form: <hostname>:<slots>
                  (e.g.: host1:2,host2:4,host3:1 indicating 2 processes can run on host1,
                  4 on host2, and 1 on host3). If not specified, defaults to using localhost:<num_proc>
    :param hostfile: Path to a host file containing the list of host names and the number of
                     available slots. Each line of the file must be of the form:
                     <hostname> slots=<slots>
    :param launcher: The launcher to use: local, distributed, mpi, horovod.
    :param launcher_kwargs: The additional keyword arguments for launcher.
    :return: None
    """
    from cloudtik.runtime.ai.runner.run import _run

    if hosts is not None and hostfile is not None:
        raise ValueError('Argument hosts and hostfile only allow one provided.')

    largs = _LaunchArgs()

    largs.command = command
    largs.num_proc = num_proc
    largs.nnodes = nnodes
    largs.nproc_per_node = nproc_per_node
    largs.hosts = hosts
    largs.hostfile = hostfile
    largs.launcher = launcher
    largs.launcher_kwargs = launcher_kwargs

    # set for launcher args
    for key, value in launcher_kwargs.items():
        setattr(largs, key, value)

    return _run(largs)
