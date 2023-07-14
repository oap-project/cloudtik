import logging

from cloudtik.runtime.ai.runner.cpu.launcher import CPULauncher
from cloudtik.runtime.ai.runner.mpi.mpi_training_launcher import MPITrainingLauncher
from cloudtik.runtime.ai.runner.cpu.utils import CPUinfo

logger = logging.getLogger(__name__)


class OptimizedTrainingLauncher(MPITrainingLauncher, CPULauncher):
    r"""
     Launcher for distributed training with MPI launcher
     """
    def __init__(self, args, distributor):
        MPITrainingLauncher.__init__(self, args, distributor)

    def get_mpi_pin_domain(self, nproc_per_node, ccl_worker_count, total_cores, flatten_node_cores):
        """
        I_MPI_PIN_DOMAIN specify the cores used for every MPI process.
        The first ccl_worker_count cores of every rank for ccl communication
        and the other cores will be used to do computation.
        For example: on CascadeLake 8280 CPU, 2 ranks on one node. ccl_worker_count=4
        CCL_WORKER_COUNT=4
        CCL_WORKER_AFFINITY="0,1,2,3,28,29,30,31"
        I_MPI_PIN_DOMAIN=[0xffffff0,0xffffff0000000]
        """
        ppn = nproc_per_node
        cores_per_rank = total_cores // ppn

        pin_domain = "["
        for proc in range(ppn):
            domain_binary = 0
            begin = proc * cores_per_rank + ccl_worker_count
            end = proc * cores_per_rank + cores_per_rank - 1
            for i in range(begin, end + 1):
                # CloudTik: patch start
                # domain_binary |= (1 << i)
                domain_binary |= (1 << flatten_node_cores[i])
                # CloudTik: patch end
            pin_domain += hex(domain_binary) + ","
        pin_domain += "]"
        return pin_domain

    def get_ccl_worker_affinity(self, nproc_per_node, ccl_worker_count, total_cores, flatten_node_cores):
        """
        Computation and communication use different cores when using oneCCL
        backend for distributed training. we use first ccl_worker_count cores of
        every rank for ccl communication
        """
        ppn = nproc_per_node
        cores_per_rank = total_cores // ppn
        affinity = ''
        for proc in range(ppn):
            for ccl_worker in range(ccl_worker_count):
                # CloudTik: patch start
                affinity += str(flatten_node_cores[proc * cores_per_rank + ccl_worker]) + ","
                # affinity += str(proc * cores_per_rank + ccl_worker) + ","
                # CloudTik: patch end
        affinity = affinity[:-1]
        return affinity

    def set_environment(self):
        args = self.args
        if not self.distributor.distributed:
            # for local single node
            cpuinfo = self.cpuinfo
        else:
            # use any worker address for getting cpu info
            worker_addr = self.get_master_addr(args)
            cpuinfo = CPUinfo(host_ip=worker_addr)

        self.distributor.resolve(cpuinfo.sockets())

        # call super set_environment to make sure other things are set
        # since the distributor already resolved, it will not resolve twice
        super().set_environment()

        node_cores = cpuinfo.node_physical_cores
        total_cores_per_node = cpuinfo.physical_cores()
        if args.use_logical_core:
            node_cores = cpuinfo.node_logical_cores
            total_cores_per_node = cpuinfo.logical_cores()

        nproc_per_node = self.distributor.nproc_per_node

        flatten_node_cores = []
        for node_numa_cores in node_cores:
            flatten_node_cores.extend(node_numa_cores)

        mpi_pin_domain = self.get_mpi_pin_domain(
            nproc_per_node, args.ccl_worker_count, total_cores_per_node, flatten_node_cores)
        self.set_env("I_MPI_PIN_DOMAIN", mpi_pin_domain)

        ppn = nproc_per_node
        cores_per_rank = total_cores_per_node // ppn

        omp_num_threads = cores_per_rank - args.ccl_worker_count
        self.set_multi_thread_and_allocator(omp_num_threads,
                                            args.disable_iomp,
                                            True,
                                            args.enable_tcmalloc,
                                            args.enable_jemalloc,
                                            args.use_default_allocator)

        self.set_env("CCL_WORKER_COUNT", str(args.ccl_worker_count))
        ccl_affinity = self.get_ccl_worker_affinity(
            nproc_per_node, args.ccl_worker_count, total_cores_per_node, flatten_node_cores)
        self.set_env("CCL_WORKER_AFFINITY", ccl_affinity)
