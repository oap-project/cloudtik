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
        self.run_func = None
        self.executable = None

        # nodes and processes
        # If nnodes and nproc_per_node is specified
        # hosts/hostfile can be host address only without slots
        # Or you can specify hosts with slots and specify the num_proc
        # all these are handled by Distributor

        self.num_proc = None
        self.nnodes = None
        self.nproc_per_node = None

        # host arguments
        self.hosts = None
        self.hostfile = None

        self.distributed = False
        self.launcher = None

        # control flags
        self.module = False
        self.no_python = False
        self.log_path = None
        self.log_file_prefix = None
        self.verbose = None

        # Pytorch DPP
        self.master_addr = "127.0.0.1"
        self.master_port = 29500

        # library arguments
        # MPI
        self.mpi_args = None
        self.tcp_flag = None
        self.binding_args = None
        self.num_nccl_streams = None
        self.thread_affinity = None

        # CPU, CCL, IOMP, Allocator options
        self.use_logical_core = False
        self.ccl_worker_count = 4
        self.disable_iomp = False
        self.enable_tcmalloc = False
        self.enable_jemalloc = False
        self.use_default_allocator = False

        # Horovod controller arguments
        self.use_gloo = None
        self.use_mpi = None
        self.nics = None
        self.output_filename = None


def run(
        func,
        args=(),
        kwargs=None,
        num_proc=None,
        nnodes=None,
        nproc_per_node=None,
        hosts=None,
        hostfile=None,
        executable=None,
        verbose=None,
        mpi_args=None,
        use_gloo=None,
        use_mpi=None,
        network_interfaces=None,
        output_filename=None,
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
    :param verbose: If this flag is set, extra messages will be printed.
    :param mpi_args: Extra arguments for the MPI controller. This is only used when use_mpi is True.
    :param use_gloo: Run using the Gloo
    :param use_mpi: Run using the MPI
    :param network_interfaces: List of network interfaces to use for communication. If not specified,
                               Horovod will find the common NICs among all the workers.
                               Example: ["eth0", "eth1"].
    :param output_filename: For Gloo, writes stdout / stderr of all processes to a filename of the form
                            <output_filename>/rank.<rank>/<stdout | stderr>. The <rank> will be padded with 0
                            characters to ensure lexicographical order.
                            For MPI, delegates its behavior to mpirun.
    :return: Return a list which contains values return by all processes.
             The index of the list corresponds to the rank of each process.
             Returns only the first min_num_proc results, if set.
    """
    from cloudtik.runtime.ai.runner.launch import _run

    if kwargs is None:
        kwargs = {}

    def wrapped_func():
        return func(*args, **kwargs)

    if hosts is not None and hostfile is not None:
        raise ValueError('Argument hosts and hostfile only allow one provided.')

    if use_gloo and use_mpi:
        raise ValueError('Argument use_gloo and use_mpi only allow one set True.')

    largs = _LaunchArgs()

    largs.run_func = wrapped_func
    largs.executable = executable
    largs.num_proc = num_proc
    largs.nnodes = nnodes
    largs.nproc_per_node = nproc_per_node
    largs.hosts = hosts
    largs.hostfile = hostfile
    largs.launcher = "horovod"
    largs.verbose = verbose
    largs.mpi_args = mpi_args
    largs.use_gloo = use_gloo
    largs.use_mpi = use_mpi
    largs.nics = set(network_interfaces) if network_interfaces else None
    largs.output_filename = output_filename

    return _run(largs)


def run_command(
        command,
        num_proc=None,
        nnodes=None,
        nproc_per_node=None,
        hosts=None,
        hostfile=None,
        launcher=None,
        verbose=None,
        mpi_args=None,
        use_gloo=None,
        use_mpi=None,
        network_interfaces=None,
        output_filename=None,
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
    :param launcher: The launcher to use. Valid launchers are "default", "optimized", "horovod"
    :param verbose: If this flag is set, extra messages will be printed.
    :param mpi_args: Extra arguments for the MPI controller. This is only used when use_mpi is True.
    :param use_gloo: Run using the Gloo
    :param use_mpi: Run using the MPI
    :param network_interfaces: List of network interfaces to use for communication. If not specified,
                               Horovod will find the common NICs among all the workers.
                               Example: ["eth0", "eth1"].
    :param output_filename: For Gloo, writes stdout / stderr of all processes to a filename of the form
                        <output_filename>/rank.<rank>/<stdout | stderr>. The <rank> will be padded with 0
                        characters to ensure lexicographical order.
                        For MPI, delegates its behavior to mpirun.
    :return: None
    """
    from cloudtik.runtime.ai.runner.launch import _run

    if hosts is not None and hostfile is not None:
        raise ValueError('Argument hosts and hostfile only allow one provided.')

    if use_gloo and use_mpi:
        raise ValueError('Argument use_gloo and use_mpi only allow one set True.')

    largs = _LaunchArgs()

    largs.command = command
    largs.num_proc = num_proc
    largs.nnodes = nnodes
    largs.nproc_per_node = nproc_per_node
    largs.hosts = hosts
    largs.hostfile = hostfile
    largs.launcher = launcher
    largs.verbose = verbose
    largs.mpi_args = mpi_args
    largs.use_gloo = use_gloo
    largs.use_mpi = use_mpi
    largs.nics = set(network_interfaces) if network_interfaces else None
    largs.output_filename = output_filename

    return _run(largs)
