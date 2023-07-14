import logging
import os
import subprocess
from os.path import expanduser

from cloudtik.runtime.ai.runner.cpu.utils import CPUinfo

logger = logging.getLogger(__name__)


def add_cpu_option_params(parser):
    group = parser.add_argument_group("Parameters for CPU options")
    group.add_argument("--use-logical-core", "--use_logical_core",
                       action='store_true', default=False,
                       help="Whether only use physical cores")

    # ccl control
    group.add_argument("--ccl-worker-count", "--ccl_worker_count",
                       action='store', dest='ccl_worker_count', default=4, type=int,
                       help="Core numbers per rank used for ccl communication")


def add_memory_allocator_params(parser):
    group = parser.add_argument_group("Memory Allocator Parameters")
    # allocator control
    group.add_argument("--enable-tcmalloc", "--enable_tcmalloc",
                       action='store_true', default=False,
                       help="Enable tcmalloc allocator")
    group.add_argument("--enable-jemalloc", "--enable_jemalloc",
                       action='store_true', default=False,
                       help="Enable jemalloc allocator")
    group.add_argument("--use-default-allocator", "--use_default_allocator",
                       action='store_true', default=False,
                       help="Use default memory allocator")

    group.add_argument("--disable-iomp", "--disable_iomp",
                       action='store_true', default=False,
                       help="By default, we use Intel OpenMP and libiomp5.so will be add to LD_PRELOAD")


def add_local_launcher_params(parser):
    group = parser.add_argument_group("Local Instance Launching Parameters")
    # instances control
    group.add_argument("--ninstances",
                       default=-1, type=int,
                       help="The number of instances to run local. "
                            "You should give the cores number you used for per instance.")
    group.add_argument("--ncore-per-instance", "--ncore_per_instance",
                       default=-1, type=int,
                       help="Cores per instance")
    group.add_argument("--skip-cross-node-cores", "--skip_cross_node_cores",
                       action='store_true', default=False,
                       help="If specified --ncore_per_instance, skips cross-node cores.")
    group.add_argument("--instance-idx", "--instance_idx",
                       default="-1", type=int,
                       help="Specify instance index to assign ncores_per_instance for instance_idx; "
                            "otherwise ncore_per_instance will be assigned sequentially to ninstances.")
    group.add_argument("--latency-mode", "--latency_mode",
                       action='store_true', default=False,
                       help="By default 4 core per instance and use all physical cores")
    group.add_argument("--throughput-mode", "--throughput_mode",
                       action='store_true', default=False,
                       help="By default one instance per node and use all physical cores")
    group.add_argument("--node-id", "--node_id",
                       default=-1, type=int,
                       help="node id for the current instance, by default all nodes will be used")
    group.add_argument("--disable-numactl", "--disable_numactl",
                       action='store_true', default=False,
                       help="Disable numactl")
    group.add_argument("--disable-taskset", "--disable_taskset",
                       action='store_true', default=False,
                       help="Disable taskset")
    group.add_argument("--core-list", "--core_list",
                       default=None, type=str,
                       help="Specify the core list as 'core_id, core_id, ....', otherwise, all the cores will be used.")
    group.add_argument("--benchmark",
                       action='store_true', default=False,
                       help="Enable benchmark config. JeMalloc's MALLOC_CONF has been tuned for low latency. "
                            "Recommend to use this for benchmarking purpose; for other use cases, "
                            "this MALLOC_CONF may cause Out-of-Memory crash.")


def add_auto_ipex_params(parser, auto_ipex_default_enabled=False):
    group = parser.add_argument_group("Code_Free Parameters")
    group.add_argument("--auto-ipex", "--auto_ipex",
                       action='store_true', default=auto_ipex_default_enabled,
                       help="Auto enabled the ipex optimization feature")
    group.add_argument("--dtype",
                       default="float32", type=str,
                       choices=['float32', 'bfloat16'],
                       help="The data type to run inference. float32 or bfloat16 is allowed.")
    group.add_argument("--auto-ipex-verbose", "--auto_ipex_verbose",
                       action='store_true', default=False,
                       help="This flag is only used for debug and UT of auto ipex.")
    group.add_argument("--disable-ipex-graph-mode", "--disable_ipex_graph_mode",
                       action='store_true', default=False,
                       help="Enable the Graph Mode for ipex.optimize")


class CPULauncher:
    r"""
     Base class for launcher
    """
    def __init__(self):
        self.cpuinfo = CPUinfo()

    def is_numactl_available(self):
        numactl_available = False
        try:
            cmd = ["numactl", "-C", "0", "-m", "0", "ls"]
            r = subprocess.run(cmd, env=os.environ, stdout=subprocess.DEVNULL)
            if r.returncode == 0:
                numactl_available = True
        except Exception as e:
            logger.warning("Core binding with numactl is not available: {}".format(
                str(e)))
        return numactl_available

    def set_memory_allocator(self, enable_tcmalloc=True, enable_jemalloc=False, use_default_allocator=False, benchmark=False):
        """
        Enable TCMalloc/JeMalloc with LD_PRELOAD and set configuration for JeMalloc.
        By default, PTMalloc will be used for PyTorch, but TCMalloc and JeMalloc can get better
        memory resue and reduce page fault to improve performance.
        """
        if enable_tcmalloc and enable_jemalloc:
            raise RuntimeError("Unable to enable TCMalloc and JEMalloc at the same time")

        if enable_tcmalloc:
            find_tc = self.add_lib_preload(lib_type="tcmalloc")
            if not find_tc:
                logger.warning("Unable to find the {} library file lib{}.so in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib"
                               " or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or "
                               "{}/.local/lib/ so the LD_PRELOAD environment variable will not be set."
                               "you can use 'conda install -c conda-forge gperftools' to install tcmalloc"
                               .format("TCmalloc", "tcmalloc", expanduser("~")))
            else:
                logger.info("Use TCMalloc memory allocator")

        elif enable_jemalloc:
            find_je = self.add_lib_preload(lib_type="jemalloc")
            if not find_je:
                logger.warning("Unable to find the {} library file lib{}.so in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib"
                               " or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or "
                               "{}/.local/lib/ so the LD_PRELOAD environment variable will not be set."
                               "you can use 'conda install -c conda-forge jemalloc' to install jemalloc"
                               .format("JeMalloc", "jemalloc", expanduser("~")))
            else:
                logger.info("Use JeMalloc memory allocator")
                if benchmark:
                    self.set_env('MALLOC_CONF',
                                 "oversize_threshold:1,background_thread:false,metadata_thp:always,dirty_decay_ms:-1,muzzy_decay_ms:-1")
                else:
                    self.set_env('MALLOC_CONF',
                                 "oversize_threshold:1,background_thread:true,metadata_thp:auto")

        elif use_default_allocator:
            pass

        else:
            find_tc = self.add_lib_preload(lib_type="tcmalloc")
            if find_tc:
                logger.info("Use TCMalloc memory allocator")
                return
            find_je = self.add_lib_preload(lib_type="jemalloc")
            if find_je:
                logger.info("Use JeMalloc memory allocator")
                return
            logger.warning("Neither TCMalloc nor JeMalloc is found in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib"
                           " or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or "
                           "{}/.local/lib/ so the LD_PRELOAD environment variable will not be set. "
                           "This may drop the performance"
                           .format(expanduser("~")))

    # set_kmp_affinity is used to control whether to set KMP_AFFINITY or not.
    # In scenario that use all cores on all nodes, including logical cores,
    # setting KMP_AFFINITY disables logical cores. In this case, KMP_AFFINITY should not be set.
    def set_multi_thread_and_allocator(
            self, ncore_per_instance, disable_iomp=False, set_kmp_affinity=True,
            enable_tcmalloc=True, enable_jemalloc=False, use_default_allocator=False, benchmark=False):
        """
        Set multi-thread configuration and enable Intel openMP and TCMalloc/JeMalloc.
        By default, GNU openMP and PTMalloc are used in PyTorch. but Intel openMP and TCMalloc/JeMalloc are better alternatives
        to get performance benefit.
        """
        self.set_memory_allocator(enable_tcmalloc, enable_jemalloc, use_default_allocator, benchmark)
        self.set_env("OMP_NUM_THREADS", str(ncore_per_instance))
        if not disable_iomp:
            find_iomp = self.add_lib_preload(lib_type="iomp5")
            if not find_iomp:
                logger.warning("Unable to find the {} library file lib{}.so in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib"
                               " or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or "
                               "{}/.local/lib/ so the LD_PRELOAD environment variable will not be set."
                               "you can use 'conda install intel-openm' to install intel openMP"
                               .format("iomp", "iomp5", expanduser("~")))
            else:
                logger.info("Using Intel OpenMP")
                if set_kmp_affinity:
                    self.set_env("KMP_AFFINITY", "granularity=fine,compact,1,0")
                self.set_env("KMP_BLOCKTIME", "1")
        self.log_env("LD_PRELOAD")
