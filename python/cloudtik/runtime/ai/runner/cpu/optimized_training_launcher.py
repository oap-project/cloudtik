import logging
import os

from cloudtik.runtime.ai.runner.cpu.cpu_info import CPUPoolList
from cloudtik.runtime.ai.runner.cpu.launcher import CPULauncher
from cloudtik.runtime.ai.runner.mpi.mpi_training_launcher import MPITrainingLauncher

logger = logging.getLogger(__name__)


LD_PRELOAD_MARKER = "LD_PRELOAD_UNSET"


class OptimizedTrainingLauncher(MPITrainingLauncher, CPULauncher):
    r"""
     Launcher for distributed training with MPI launcher
     """
    def __init__(self, args, distributor):
        MPITrainingLauncher.__init__(self, args, distributor)
        CPULauncher.__init__(self)

        self.ld_preload_backup = None

    def add_env(self, env_name, env_value):
        self.set_env(env_name, env_value)

    def get_pin_domain_affinity(
            self, cpu_pools, ccl_worker_count, logical_cores_for_ccl=False
    ):
        """
        I_MPI_PIN_DOMAIN specify the cores used for every MPI process.
        1) use physical core for oneccl
           The first ccl_worker_count cores of every rank for ccl communication
           and the other cores will be used to do computation.
           For example: on CascadeLake 8280 CPU, 2 ranks on one node. ccl_worker_count=4
           CCL_WORKER_COUNT=4
           CCL_WORKER_AFFINITY="0,1,2,3,28,29,30,31"
           I_MPI_PIN_DOMAIN=[0xffffff0,0xffffff00000000]
        2) use logical core oneccl
           The first ccl_worker_count logical cores which is correponding to the
           first ccl_worker_count physical cores are used as the ccl cores.
           For example: on CascadeLake 8280 CPU, 2 ranks on one node. ccl_worker_count=4
           CCL_WORKER_COUNT=4
           CCL_WORKER_AFFINITY="56,57,58,59,84,85,86,87"
           I_MPI_PIN_DOMAIN=[0xfffffff,0xfffffff0000000]
        """
        domain_binaries = []
        affinity = []
        for pool in cpu_pools:
            if (
                    logical_cores_for_ccl
                    and len([c for c in pool if not c.is_physical_core]) < ccl_worker_count
            ):
                self.verbose(
                    "warning",
                    "Argument --logical-cores-for-ccl is set but no enough logical cores are available. Disable this argument.",
                )
                logical_cores_for_ccl = False
                break
        for pool in cpu_pools:
            domain_binary = 0
            cores = []
            if logical_cores_for_ccl:
                affinity.extend(
                    [str(c.cpu) for c in pool if not c.is_physical_core][
                    :ccl_worker_count
                    ]
                )
                cores = [str(c.cpu) for c in pool if c.is_physical_core]
            else:
                physical_cores = [str(c.cpu) for c in pool if c.is_physical_core]
                assert ccl_worker_count < len(
                    physical_cores
                ), f"ccl_worker_count ({ccl_worker_count}) cannot exceed number of available cores ({len(physical_cores)})."
                affinity.extend(physical_cores[:ccl_worker_count])
                cores = physical_cores[ccl_worker_count:]
            for c in cores:
                domain_binary |= 1 << int(c)
            domain_binaries.append(hex(domain_binary))
        return {
            "pin_domain": f'[{",".join(domain_binaries)}]',
            "affinity": ",".join(affinity),
        }

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
        """
        Set ENVs for launching MPI process for distributed training.
        """
        args = self.args
        if not self.distributor.distributed:
            # for local single node
            cpuinfo = self.cpuinfo
        else:
            # use any worker address for getting cpu info
            worker_addr = self.get_master_addr(args)
            cpuinfo = CPUPoolList(host_ip=worker_addr)

        self.distributor.resolve(cpuinfo.num_sockets())

        # call super set_environment to make sure other things are set
        # since the distributor already resolved, it will not resolve twice
        super().set_environment()
        self.set_mpi_environment(cpuinfo)

    def set_mpi_environment(self, cpuinfo):
        args = self.args
        assert not (
                args.logical_cores_for_ccl and args.use_logical_cores
        ), "Can't use --logical-cores-for-ccl and --use-logical-cores at the same time."

        nodes_list = self.parse_list_argument(args.nodes_list)
        if args.nprocs_per_node == 0:
            args.nprocs_per_node = (
                len(set([c.node for c in cpuinfo.pool_all]))
                if len(nodes_list) == 0
                else len(nodes_list)
            )
        ncores_per_instance = args.ncores_per_instance
        if ncores_per_instance > 0:
            if (
                    not args.logical_cores_for_ccl
                    or len([c for c in cpuinfo.pool_all if not c.is_physical_core])
                    < args.nprocs_per_node * args.ccl_worker_count
            ):
                ncores_per_instance += args.ccl_worker_count
            ncores_per_instance = len(
                [c for c in cpuinfo.pool_all if c.core < ncores_per_instance]
            )
        cpuinfo.gen_pools_ondemand(
            ninstances=args.nprocs_per_node,
            ncores_per_instance=ncores_per_instance,
            use_logical_cores=True,
            use_e_cores=args.use_e_cores,
            nodes_list=nodes_list,
        )

        self.set_memory_allocator(args.memory_allocator, False, ["jemalloc"])
        self.set_omp_runtime(args.omp_runtime, True)
        omp_num_threads = len(
            [c for c in cpuinfo.pools_ondemand[0] if c.is_physical_core]
        )
        if not args.logical_cores_for_ccl:
            omp_num_threads -= args.ccl_worker_count
        self.add_env("OMP_NUM_THREADS", str(omp_num_threads))

        pin_domain_affinity = self.get_pin_domain_affinity(
            cpuinfo.pools_ondemand,
            args.ccl_worker_count,
            args.logical_cores_for_ccl,
        )
        self.add_env("I_MPI_PIN_DOMAIN", pin_domain_affinity["pin_domain"])
        self.add_env("CCL_WORKER_COUNT", str(args.ccl_worker_count))
        self.add_env("CCL_WORKER_AFFINITY", pin_domain_affinity["affinity"])

        self.ld_preload_backup = (
            os.environ["LD_PRELOAD"]
            if "LD_PRELOAD" in os.environ
            else LD_PRELOAD_MARKER
        )
        if len(self.ld_preload) > 0:
            os.environ["LD_PRELOAD"] = ":".join(self.ld_preload)
            self.verbose("info", f'LD_PRELOAD={os.environ["LD_PRELOAD"]}')
        else:
            if "LD_PRELOAD" in os.environ:
                del os.environ["LD_PRELOAD"]
        for k, v in self.environ_set.items():
            self.verbose("info", f"env: {k}={v}")

    def finalize(self):
        if self.ld_preload_backup == LD_PRELOAD_MARKER:
            if "LD_PRELOAD" in os.environ:
                del os.environ["LD_PRELOAD"]
        else:
            os.environ["LD_PRELOAD"] = self.ld_preload_backup
