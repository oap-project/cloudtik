import glob
import logging
import os
import platform
from argparse import ArgumentParser, REMAINDER
from argparse import RawTextHelpFormatter
from datetime import datetime

logger = logging.getLogger(__name__)

r"""
This is a script for launching PyTorch training and inference on Intel Xeon CPU with optimal configurations.
Now, single instance inference/training, multi-instance inference/training and distributed training
with oneCCL backend is enabled.

To get the peak performance on Intel Xeon CPU, the script optimizes the configuration of thread and memory
management. For thread management, the script configures thread affinity and the preload of Intel OMP library.
For memory management, it configures NUMA binding and preload optimized memory allocation library (e.g. tcmalloc, jemalloc).

**How to use this module:**

*** Single instance inference/training ***

1. Run single-instance inference or training on a single node with all CPU nodes.

::

   >>> cloudtik-ml-run --throughput_mode script.py args

2. Run single-instance inference or training on a single CPU node.

::

   >>> cloudtik-ml-run --node_id 1 script.py args

*** Multi-instance inference ***

1. Multi-instance
   By default, one instance per node. if you want to set the instance numbers and core per instance,
   --ninstances and  --ncore_per_instance should be set.


   >>> cloudtik-ml-run -- python_script args

   eg: on CLX8280 with 14 instance, 4 cores per instance
::

   >>> cloudtik-ml-run  --ninstances 14 --ncore_per_instance 4 python_script args

2. Run single-instance inference among multiple instances.
   By default, runs all ninstances. If you want to independently run a single instance among ninstances, specify instance_idx.

   eg: run 0th instance among SKX with 2 instance (i.e., numactl -C 0-27)
::

   >>> cloudtik-ml-run  --ninstances 2 --instance_idx 0 python_script args

   eg: run 1st instance among SKX with 2 instance (i.e., numactl -C 28-55)
::

   >>> cloudtik-ml-run  --ninstances 2 --instance_idx 1 python_script args

   eg: run 0th instance among SKX with 2 instance, 2 cores per instance, first four cores (i.e., numactl -C 0-1)
::

   >>> cloudtik-ml-run  --core_list "0, 1, 2, 3" --ninstances 2 --ncore_per_instance 2 --instance_idx 0 python_script args

*** Distributed Training ***

spawns up multiple distributed training processes on each of the training nodes. For intel_extension_for_pytorch, oneCCL
is used as the communication backend and MPI used to launch multi-proc. To get the better
performance, you should specify the different cores for oneCCL communication and computation
process separately. This tool can automatically set these ENVs(such as I_MPI_PIN_DOMIN) and launch
multi-proc for you.

The utility can be used for single-node distributed training, in which one or
more processes per node will be spawned.  It can also be used in
multi-node distributed training, by spawning up multiple processes on each node
for well-improved multi-node distributed training performance as well.


1. Single-Node multi-process distributed training

::

    >>> cloudtik-ml-run --distributed  python_script  --arg1 --arg2 --arg3 and all other
                arguments of your training script

2. Multi-Node multi-process distributed training: (e.g. two nodes)


rank 0: *(IP: 192.168.10.10, and has a free port: 295000)*

::

    >>> cloudtik-ml-run --distributed --nproc_per_node=xxx
               --nnodes=2 --hostfile hostfile python_sript --arg1 --arg2 --arg3
               and all other arguments of your training script)


3. To look up what optional arguments this module offers:

::

    >>> cloudtik-ml-run --help

*** Memory allocator  ***

"--enable_tcmalloc" and "--enable_jemalloc" can be used to enable different memory allocator.

"""


def add_cpu_option_params(parser):
    group = parser.add_argument_group("Parameters for CPU options")
    group.add_argument("--use_logical_core", action='store_true', default=False,
                       help="Whether only use physical cores")


def add_distributed_training_params(parser):
    group = parser.add_argument_group("Distributed Training Parameters With oneCCL backend")
    group.add_argument("--nnodes", metavar='\b', type=int, default=0,
                       help="The number of nodes to use for distributed "
                       "training")
    group.add_argument("--nproc_per_node", metavar='\b', type=int, default=0,
                       help="The number of processes to launch on each node")
    # ccl control
    group.add_argument("--ccl_worker_count", metavar='\b', default=4, type=int,
                       help="Core numbers per rank used for ccl communication")
    # mpi control
    group.add_argument("--master_addr", metavar='\b', default="127.0.0.1", type=str,
                       help="Master node (rank 0)'s address, should be either "
                            "the IP address or the hostname of node 0, for "
                            "single node multi-proc training, the "
                            "--master_addr can simply be 127.0.0.1")
    group.add_argument("--master_port", metavar='\b', default=29500, type=int,
                       help="Master node (rank 0)'s free port that needs to "
                            "be used for communication during distributed "
                            "training")
    group.add_argument("--hostfile", metavar='\b', default="", type=str,
                       help="Hostfile is necessary for multi-node multi-proc "
                            "training. hostfile includes the node address list "
                            "node address which should be either the IP address"
                            "or the hostname.")
    group.add_argument("--hosts", metavar='\b', default="", type=str,
                       help="List of hosts separated with comma for launching tasks. "
                            "When hosts is specified, it implies distributed training. "
                            "node address which should be either the IP address"
                            "or the hostname.")
    group.add_argument("--more_mpi_params", metavar='\b', default="", type=str,
                       help="User can pass more parameters for mpiexec.hydra "
                            "except for -np -ppn -hostfile and -genv I_MPI_PIN_DOMAIN")


def add_memory_allocator_params(parser):
    group = parser.add_argument_group("Memory Allocator Parameters")
    # allocator control
    group.add_argument("--enable_tcmalloc", action='store_true', default=False,
                       help="Enable tcmalloc allocator")
    group.add_argument("--enable_jemalloc", action='store_true', default=False,
                       help="Enable jemalloc allocator")
    group.add_argument("--use_default_allocator", action='store_true', default=False,
                       help="Use default memory allocator")


def add_multi_instance_params(parser):
    group = parser.add_argument_group("Multi-instance Parameters")
    # multi-instance control
    group.add_argument("--ncore_per_instance", metavar='\b', default=-1, type=int,
                       help="Cores per instance")
    group.add_argument("--skip_cross_node_cores", action='store_true', default=False,
                       help="If specified --ncore_per_instance, skips cross-node cores.")
    group.add_argument("--ninstances", metavar='\b', default=-1, type=int,
                       help="For multi-instance, you should give the cores number you used for per instance.")
    group.add_argument("--instance_idx", metavar='\b', default="-1", type=int,
                       help="Specify instance index to assign ncores_per_instance for instance_idx; "
                            "otherwise ncore_per_instance will be assigned sequentially to ninstances.")
    group.add_argument("--latency_mode", action='store_true', default=False,
                       help="By detault 4 core per instance and use all physical cores")
    group.add_argument("--throughput_mode", action='store_true', default=False,
                       help="By default one instance per node and use all physical cores")
    group.add_argument("--node_id", metavar='\b', default=-1, type=int,
                       help="node id for multi-instance, by default all nodes will be used")
    group.add_argument("--disable_numactl", action='store_true', default=False,
                       help="Disable numactl")
    group.add_argument("--disable_taskset", action='store_true', default=False,
                       help="Disable taskset")
    group.add_argument("--core_list", metavar='\b', default=None, type=str,
                       help="Specify the core list as 'core_id, core_id, ....', otherwise, all the cores will be used.")
    group.add_argument("--benchmark", action='store_true', default=False,
                       help="Enable benchmark config. JeMalloc's MALLOC_CONF has been tuned for low latency. "
                            "Recommend to use this for benchmarking purpose; for other use cases, "
                            "this MALLOC_CONF may cause Out-of-Memory crash.")


def add_kmp_iomp_params(parser):
    group = parser.add_argument_group("IOMP Parameters")
    group.add_argument("--disable_iomp", action='store_true', default=False,
                       help="By default, we use Intel OpenMP and libiomp5.so will be add to LD_PRELOAD")


def add_auto_ipex_params(parser, auto_ipex_default_enabled=False):
    group = parser.add_argument_group("Code_Free Parameters")
    group.add_argument("--auto_ipex", action='store_true', default=auto_ipex_default_enabled,
                       help="Auto enabled the ipex optimization feature")
    group.add_argument("--dtype", metavar='\b', default="float32", type=str,
                       choices=['float32', 'bfloat16'],
                       help="The data type to run inference. float32 or bfloat16 is allowed.")
    group.add_argument("--auto_ipex_verbose", action='store_true', default=False,
                       help="This flag is only used for debug and UT of auto ipex.")
    group.add_argument("--disable_ipex_graph_mode", action='store_true', default=False,
                       help="Enable the Graph Mode for ipex.optimize")


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(
        description="This is a script for launching PyTorch training and inference on Intel Xeon CPU "
                    "with optimal configurations. Now, single instance inference/training, multi-instance "
                    "inference/training and distributed training with oneCCL backend is enabled. "
                    "To get the peak performance on Intel Xeon CPU, the script optimizes the configuration "
                    "of thread and memory management. For thread management, the script configures thread "
                    "affinity and the preload of Intel OMP library. For memory management, it configures "
                    "NUMA binding and preload optimized memory allocation library (e.g. tcmalloc, jemalloc) "
                    "\n################################# Basic usage ############################# \n"
                    "\n 1. single instance\n"
                    "\n   >>> cloudtik-ml-run python_script args \n"
                    "\n2. multi-instance \n"
                    "\n    >>> cloudtik-ml-run --ninstances xxx --ncore_per_instance xx python_script args\n"
                    "\n3. Single-Node multi-process distributed training\n"
                    "\n    >>> cloudtik-ml-run --distributed  python_script args\n"
                    "\n4. Multi-Node multi-process distributed training: (e.g. two nodes)\n"
                    "\n   rank 0: *(IP: 192.168.10.10, and has a free port: 295000)*\n"
                    "\n   >>> cloudtik-ml-run --distributed --nproc_per_node=2\n"
                    "\n       --nnodes=2 --hostfile hostfile python_script args\n"
                    "\n############################################################################# \n",
                    formatter_class=RawTextHelpFormatter)

    parser.add_argument("--multi_instance", action='store_true', default=False,
                        help="Enable multi-instance, by default one instance per node")

    parser.add_argument('--distributed', action='store_true', default=False,
                        help='Enable distributed training.')
    parser.add_argument("--launcher", metavar='\b', default="", type=str,
                        help="The launcher to use: default, optimized, horovod")
    parser.add_argument("-m", "--module", default=False, action="store_true",
                        help="Changes each process to interpret the launch script "
                             "as a python module, executing with the same behavior as"
                             "'python -m'.")

    parser.add_argument("--no_python", default=False, action="store_true",
                        help="Do not prepend the --program script with \"python\" - just exec "
                             "it directly. Useful when the script is not a Python script.")

    parser.add_argument("--log_path", metavar='\b', default="", type=str,
                        help="The log file directory. Default path is '', which means disable logging to files.")
    parser.add_argument("--log_file_prefix", metavar='\b', default="run", type=str,
                        help="log file prefix")

    add_cpu_option_params(parser)
    add_memory_allocator_params(parser)
    add_kmp_iomp_params(parser)

    add_distributed_training_params(parser)
    add_multi_instance_params(parser)

    add_auto_ipex_params(parser)

    # positional
    parser.add_argument("program", type=str,
                        help="The full path to the program/script to be launched. "
                             "followed by all the arguments for the script")

    # rest from the training program
    parser.add_argument('program_args', nargs=REMAINDER)
    args = parser.parse_args()
    args.run_func = None
    args.executable = None
    return args


def _verify_ld_preload():
    if "LD_PRELOAD" in os.environ:
        lst_valid = []
        tmp_ldpreload = os.environ["LD_PRELOAD"]
        for item in tmp_ldpreload.split(":"):
            if item != "":
                matches = glob.glob(item)
                if len(matches) > 0:
                    lst_valid.append(item)
                else:
                    logger.warning("{} doesn't exist. Removing it from LD_PRELOAD.".format(item))
        if len(lst_valid) > 0:
            os.environ["LD_PRELOAD"] = ":".join(lst_valid)
        else:
            os.environ["LD_PRELOAD"] = ""


def _setup_logger(args):
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=format_str)

    root_logger = logging.getLogger("")
    root_logger.setLevel(logging.INFO)

    if args.log_path:
        path = os.path.dirname(args.log_path if args.log_path.endswith('/') else args.log_path + '/')
        if not os.path.exists(path):
            os.makedirs(path)
        args.log_path = path
        args.log_file_prefix = '{}_{}'.format(args.log_file_prefix, datetime.now().strftime("%Y%m%d%H%M%S"))

        fileHandler = logging.FileHandler("{0}/{1}_instances.log".format(args.log_path, args.log_file_prefix))
        logFormatter = logging.Formatter(format_str)
        fileHandler.setFormatter(logFormatter)

        # add the handle to root logger
        root_logger.addHandler(fileHandler)


def _run(args):
    if args.distributed and args.multi_instance:
        raise RuntimeError("Either args.distributed or args.multi_instance should be set")

    if args.nnodes > 1 or args.hosts or args.hostfile:
        args.distributed = True

    if not args.distributed:
        if args.latency_mode and args.throughput_mode:
            raise RuntimeError("Either args.latency_mode or args.throughput_mode should be set")

    if not args.no_python and not args.program.endswith(".py"):
        raise RuntimeError("For non Python script, you should use '--no_python' parameter.")

    env_before = set(os.environ.keys())

    # Verify LD_PRELOAD
    _verify_ld_preload()

    if args.distributed:
        if args.launcher == "default":
            from cloudtik.runtime.ml.runner.cpu.default_training_launcher \
                import DefaultTrainingLauncher
            launcher = DefaultTrainingLauncher(args)
        elif args.launcher == "horovod":
            from cloudtik.runtime.ml.runner.horovod_training_launcher \
                import HorovodTrainingLauncher
            launcher = HorovodTrainingLauncher(args)
        else:
            from cloudtik.runtime.ml.runner.cpu.optimized_distributed_training_launcher \
                import OptimizedDistributedTrainingLauncher
            launcher = OptimizedDistributedTrainingLauncher(args)
    else:
        from cloudtik.runtime.ml.runner.cpu.multi_instance_launcher \
            import MultiInstanceLauncher
        launcher = MultiInstanceLauncher(args)

    launcher.launch()

    for x in sorted(set(os.environ.keys()) - env_before):
        logger.debug('{0}={1}'.format(x, os.environ[x]))


def main():
    if platform.system() == "Windows":
        raise RuntimeError("Windows platform is not supported!!!")

    args = parse_args()
    _setup_logger(args)

    _run(args)


if __name__ == "__main__":
    main()
