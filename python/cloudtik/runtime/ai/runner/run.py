import argparse
import glob
import logging
import os
import platform
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
from datetime import datetime

from cloudtik.runtime.ai.runner.cpu.cpu_launcher import add_cpu_launcher_params
from cloudtik.runtime.ai.runner.cpu.local_launcher import add_local_cpu_launcher_params, add_auto_ipex_params
from cloudtik.runtime.ai.runner.cpu.training_launcher import add_cpu_training_launcher_params
from cloudtik.runtime.ai.runner.distributed_launcher import add_distributed_params
from cloudtik.runtime.ai.runner.horovod.horovod_launcher import add_horovod_params
from cloudtik.runtime.ai.runner.mpi.mpi_launcher import add_mpi_params
from cloudtik.runtime.ai.runner.util.distributor import Distributor

logger = logging.getLogger(__name__)

r"""
This is a launch program for running local or distributed training and inference program.

This launch program can wrapper different launch methods and provide a abstracted view of launching
python program.

For the launching on CPU clusters, the script optimizes the configuration of thread and memory
management. For thread management, the script configures thread affinity and the preload of Intel OMP library.
For memory management, it configures NUMA binding and preload optimized memory allocation library (e.g. tcmalloc, jemalloc).

**How to use this module:**

*** Local single-process inference/training ***

1. Run single-process inference or training on a single node with all CPU nodes.

::

   >>> cloudtik-run --throughput_mode script.py args

2. Run single-process inference or training on a single CPU node.

::

   >>> cloudtik-run --node_id 1 script.py args

*** Local multi-process inference ***

1. Multi-process
   By default, one instance per node. if you want to set the instance numbers and core per instance,
   --ninstances and --ncores-per-instance should be set.


   >>> cloudtik-run -- python_script args

   eg: on CLX8280 with 14 instance, 4 cores per instance
::

   >>> cloudtik-run --ninstances 14 --ncores-per-instance 4 python_script args

2. Run single-process inference among multiple instances.
   By default, runs all ninstances. If you want to independently run a single instance among ninstances, specify instance_idx.

   eg: run 0th instance among SKX with 2 instance (i.e., numactl -C 0-27)
::

   >>> cloudtik-run --ninstances 2 --instance-idx 0 python_script args

   eg: run 1st instance among SKX with 2 instance (i.e., numactl -C 28-55)
::

   >>> cloudtik-run --ninstances 2 --instance-idx 1 python_script args

   eg: run 0th instance among SKX with 2 instance, 2 cores per instance, first four cores (i.e., numactl -C 0-1)
::

   >>> cloudtik-run --cores-list "0, 1, 2, 3" --ninstances 2 --ncores-per-instance 2 --instance-idx 0 python_script args

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

    >>> cloudtik-run --distributed  python_script  --arg1 --arg2 --arg3 and all other
                arguments of your training script

2. Multi-Node multi-process distributed training: (e.g. two nodes)

::

    >>> cloudtik-run --nproc-per-node=2
               --nnodes=2 --hosts ip1,ip2 python_sript --arg1 --arg2 --arg3
               and all other arguments of your training script)
"""


def add_common_arguments(parser):
    parser.add_argument('--distributed',
                        action='store_true', default=False,
                        help='Enable distributed training.')
    parser.add_argument("--launcher",
                        default="", type=str,
                        help="The launcher to use: default, optimized, horovod")
    parser.add_argument("-m", "--module",
                        default=False, action="store_true",
                        help="Changes each process to interpret the launch script "
                             "as a python module, executing with the same behavior as"
                             "'python -m'.")

    parser.add_argument("--no-python", "--no_python",
                        default=False, action="store_true",
                        help="Do not prepend the --program script with \"python\" - just exec "
                             "it directly. Useful when the script is not a Python script.")

    parser.add_argument("--log-dir", "--log_dir",
                        default="", type=str,
                        help="The log file directory. Default path is '', which means disable logging to files.")
    parser.add_argument("--log-file-prefix", "--log_file_prefix",
                        default="run", type=str,
                        help="log file prefix")

    parser.add_argument("--verbose",
                        default=False, action='store_true',
                        help='If this flag is set, extra messages will be printed.')
    parser.add_argument("--validate-ld-preload", "--validate_ld_preload",
                        default=False, action='store_true',
                        help='If this flag is set, libraries set in LD_PRELOAD will be validated.')

    parser.add_argument('command', nargs=argparse.REMAINDER,
                        help='Command to be executed.')


def create_parser():
    """
    Helper function getting the parser of the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(
        description="This is a program for launching local or distributed training and inference."
                    "\n################################# Basic usage ############################# \n"
                    "\n1. Local single-process training or inference\n"
                    "\n   >>> cloudtik-run python_script args \n"
                    "\n2. Local multi-process inference \n"
                    "\n    >>> cloudtik-run --ninstances 2 --ncores-per-instance 8 python_script args\n"
                    "\n3. Single-Node multi-process distributed training\n"
                    "\n    >>> cloudtik-run --distributed  python_script args\n"
                    "\n4. Multi-Node multi-process distributed training: (e.g. two nodes)\n"
                    "\n   >>> cloudtik-run --nproc-per-node=2\n"
                    "\n       --nnodes=2 --hosts ip1,ip2 python_script args\n"
                    "\n############################################################################# \n",
                    formatter_class=RawTextHelpFormatter)

    add_common_arguments(parser)

    add_cpu_launcher_params(parser)
    add_local_cpu_launcher_params(parser)
    add_auto_ipex_params(parser)
    add_cpu_training_launcher_params(parser)

    add_distributed_params(parser)
    add_mpi_params(parser)
    add_horovod_params(parser)

    return parser


def parse_args(parser):
    """
    Helper function parsing the command line options
    @retval arguments
    """
    args = parser.parse_args()
    args.run_func = None
    args.executable = None
    return args


def _validate_ld_preload():
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

    if args.log_dir:
        path = os.path.dirname(args.log_dir if args.log_dir.endswith('/') else args.log_dir + '/')
        if not os.path.exists(path):
            os.makedirs(path)
        args.log_dir = path
        args.log_file_prefix = '{}_{}'.format(args.log_file_prefix, datetime.now().strftime("%Y%m%d%H%M%S"))

        fileHandler = logging.FileHandler("{0}/{1}_instances.log".format(args.log_dir, args.log_file_prefix))
        logFormatter = logging.Formatter(format_str)
        fileHandler.setFormatter(logFormatter)

        # add the handle to root logger
        root_logger.addHandler(fileHandler)


def _run(args):
    # check either command or func be specified
    if not args.command and not args.run_func:
        raise ValueError("Must specify either command or function to launch.")

    distributor = Distributor(
        args.num_proc,
        args.nnodes,
        args.nproc_per_node,
        args.hosts,
        args.hostfile,
    )

    if distributor.distributed:
        args.distributed = True

    if not args.distributed:
        if args.latency_mode and args.throughput_mode:
            raise RuntimeError("Either args.latency_mode or args.throughput_mode should be set")

    env_before = set(os.environ.keys())

    if args.validate_ld_preload:
        _validate_ld_preload()

    if args.distributed:
        if args.launcher == "mpi":
            from cloudtik.runtime.ai.runner.mpi.mpi_launcher \
                import MPILauncher
            launcher = MPILauncher(args, distributor)
        elif args.launcher == "horovod":
            from cloudtik.runtime.ai.runner.horovod.horovod_launcher \
                import HorovodLauncher
            launcher = HorovodLauncher(args, distributor)
        else:
            from cloudtik.runtime.ai.runner.cpu.training_launcher \
                import CPUTrainingLauncher
            launcher = CPUTrainingLauncher(args, distributor)
    else:
        from cloudtik.runtime.ai.runner.cpu.local_launcher \
            import LocalCPULauncher
        launcher = LocalCPULauncher(args, distributor)

    launcher.launch()

    for x in sorted(set(os.environ.keys()) - env_before):
        logger.debug('{0}={1}'.format(x, os.environ[x]))


def main():
    if platform.system() == "Windows":
        raise RuntimeError("Windows platform is not supported!!!")

    parser = create_parser()
    args = parse_args(parser)
    _setup_logger(args)

    _run(args)


if __name__ == "__main__":
    main()
