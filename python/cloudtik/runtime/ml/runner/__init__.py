class _LaunchArgs(object):
    def __init__(self):
        self.nnodes = None
        self.nproc_per_node = None
        self.program = None
        self.program_args = None
        self.run_func = None
        self.executable = None

        # host arguments
        self.hosts = None
        self.hostfile = None

        self.master_addr = "127.0.0.1"
        self.master_port = 29500

        self.nics = None
        self.verbose = None
        self.output_filename = None

        # control flags
        self.multi_instance = False
        self.distributed = False
        self.launcher = None
        self.module = False
        self.no_python = False
        self.log_path = None
        self.log_file_prefix = None

        # library arguments
        # MPI
        self.more_mpi_args = None
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


def run(
        func,
        args=(),
        kwargs=None,
        nnodes=None,
        nproc_per_node=None,
        hosts=None,
        hostfile=None,
        output_filename=None,
        verbose=None,
        use_gloo=None,
        use_mpi=None,
        mpi_args=None,
        network_interfaces=None,
        executable=None,
       ):
    """
    Launch a job to run the specified process function and get the return value.

    :param func: The function to be run in job processes. The function return value will
                 be collected as the corresponding process return value.
                 This function must be compatible with pickle.
    :param args: Arguments to pass to `func`.
    :param kwargs: Keyword arguments to pass to `func`.
    :param nnodes: The number of nodes. if not specified, use the number of nodes in the hosts
    :param nproc_per_node: The number of process per node.
    :param hosts: List of host names and the number of available slots
                  for running processes on each, of the form: <hostname>:<slots>
                  (e.g.: host1:2,host2:4,host3:1 indicating 2 processes can run on host1,
                  4 on host2, and 1 on host3). If not specified, defaults to using localhost:<num_proc>
    :param hostfile: Path to a host file containing the list of host names and the number of
                     available slots. Each line of the file must be of the form:
                     <hostname> slots=<slots>
    :param output_filename: For Gloo, writes stdout / stderr of all processes to a filename of the form
                            <output_filename>/rank.<rank>/<stdout | stderr>. The <rank> will be padded with 0
                            characters to ensure lexicographical order.
                            For MPI, delegates its behavior to mpirun.
    :param verbose: If this flag is set, extra messages will be printed.
    :param use_gloo: Run using the Gloo
    :param use_mpi: Run using the MPI
    :param mpi_args: Extra arguments for the MPI controller. This is only used when use_mpi is True.
    :param network_interfaces: List of network interfaces to use for communication. If not specified,
                               Horovod will find the common NICs among all the workers.
                               Example: ["eth0", "eth1"].
    :param executable: Optional executable to run when launching the workers. Defaults to `sys.executable`.
    :return: Return a list which contains values return by all processes.
             The index of the list corresponds to the rank of each process.
             Returns only the first min_num_proc results, if set.
    """
    from .launch import _run

    if kwargs is None:
        kwargs = {}

    def wrapped_func():
        return func(*args, **kwargs)

    if hosts is not None and hostfile is not None:
        raise ValueError('Argument hosts and hostfile only allow one provided.')

    if use_gloo and use_mpi:
        raise ValueError('Argument use_gloo and use_mpi only allow one set True.')

    largs = _LaunchArgs()

    largs.nnodes = nnodes
    largs.nproc_per_node = nproc_per_node
    largs.hosts = hosts
    largs.hostfile = hostfile
    largs.launcher = "horovod"
    largs.more_mpi_args = mpi_args
    largs.output_filename = output_filename
    largs.verbose = verbose
    largs.use_gloo = use_gloo
    largs.use_mpi = use_mpi
    largs.nics = set(network_interfaces) if network_interfaces else None
    largs.run_func = wrapped_func
    largs.executable = executable

    return _run(largs)


def run_command(
        program,
        program_args=None,
        nnodes=None,
        nproc_per_node=None,
        hosts=None,
        hostfile=None,
        launcher=None,
        output_filename=None,
        verbose=None,
        use_gloo=None,
        use_mpi=None,
        mpi_args=None,
        network_interfaces=None
       ):
    """
    Launch a job to run the specified process function and get the return value.

    :param program: The program to be run in job processes.
    :param program_args: The list of program arguments
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
    :param output_filename: For Gloo, writes stdout / stderr of all processes to a filename of the form
                            <output_filename>/rank.<rank>/<stdout | stderr>. The <rank> will be padded with 0
                            characters to ensure lexicographical order.
                            For MPI, delegates its behavior to mpirun.
    :param verbose: If this flag is set, extra messages will be printed.
    :param use_gloo: Run using the Gloo
    :param use_mpi: Run using the MPI
    :param mpi_args: Extra arguments for the MPI controller. This is only used when use_mpi is True.
    :param network_interfaces: List of network interfaces to use for communication. If not specified,
                               Horovod will find the common NICs among all the workers.
                               Example: ["eth0", "eth1"].
    :return: None
    """
    from .launch import _run

    if hosts is not None and hostfile is not None:
        raise ValueError('Argument hosts and hostfile only allow one provided.')

    if use_gloo and use_mpi:
        raise ValueError('Argument use_gloo and use_mpi only allow one set True.')

    largs = _LaunchArgs()

    largs.nnodes = nnodes
    largs.nproc_per_node = nproc_per_node
    largs.hosts = hosts
    largs.hostfile = hostfile
    largs.launcher = launcher
    largs.more_mpi_args = mpi_args
    largs.output_filename = output_filename
    largs.verbose = verbose
    largs.use_gloo = use_gloo
    largs.use_mpi = use_mpi
    largs.nics = set(network_interfaces) if network_interfaces else None
    largs.program = program
    largs.program_args = program_args

    return _run(largs)
