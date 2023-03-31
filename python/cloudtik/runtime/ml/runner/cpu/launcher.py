import logging
import os
import subprocess
from os.path import expanduser

from cloudtik.runtime.ml.runner.cpu.utils import CPUinfo
from cloudtik.runtime.ml.runner.launcher import Launcher

logger = logging.getLogger(__name__)


class CPULauncher(Launcher):
    r"""
     Base class for launcher
    """
    def __init__(self, args):
        super().__init__(args)
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
        '''
        Enable TCMalloc/JeMalloc with LD_PRELOAD and set configuration for JeMalloc.
        By default, PTMalloc will be used for PyTorch, but TCMalloc and JeMalloc can get better
        memory resue and reduce page fault to improve performance.
        '''
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
        '''
        Set multi-thread configuration and enable Intel openMP and TCMalloc/JeMalloc.
        By default, GNU openMP and PTMalloc are used in PyTorch. but Intel openMP and TCMalloc/JeMalloc are better alternatives
        to get performance benifit.
        '''
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
