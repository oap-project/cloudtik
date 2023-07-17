import logging
import os

from cloudtik.runtime.ai.runner.cpu.cpu_pool import CPUPoolScheduler
from cloudtik.runtime.ai.runner.cpu.cpu_launcher import CPULauncher
from cloudtik.runtime.ai.runner.mpi.mpi_launcher import MPILauncher

logger = logging.getLogger(__name__)


LD_PRELOAD_MARKER = "LD_PRELOAD_UNSET"


def add_distributed_cpu_launcher_params(parser):
    group = parser.add_argument_group("Parameters for distributed CPU launcher")

    # ccl control
    group.add_argument(
        "--ccl-worker-count", "--ccl_worker_count",
        action='store', dest='ccl_worker_count', default=4, type=int,
        help="Core numbers per rank used for ccl communication")

    group.add_argument(
        "--logical-cores-for-ccl", "--logical_cores_for_ccl",
        action="store_true", default=False,
        help="Use logical cores for the ccl worker.",
    )


class DistributedCPULauncher(MPILauncher, CPULauncher):
    r"""
     Launcher for distributed training with MPI launcher
     """
    def __init__(self, args, distributor):
        MPILauncher.__init__(self, args, distributor)
        CPULauncher.__init__(self)

        self.scheduler = None
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
                    "Argument --logical-cores-for-ccl is set but no enough logical cores are available. "
                    "Disable this argument.",
                )
                logical_cores_for_ccl = False
                break
        for pool in cpu_pools:
            domain_binary = 0
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

    def resolve(self):
        args = self.args
        if not self.distributor.distributed_with_hosts:
            # for local single node
            self.scheduler = CPUPoolScheduler(logger=logger)
        else:
            # use any worker address for getting cpu info
            worker_addr = self.get_master_addr(args)
            self.scheduler = CPUPoolScheduler(logger=logger, host_ip=worker_addr)

        nproc_per_node = args.nproc_per_node
        if not nproc_per_node:
            nodes_list = self.parse_list_argument(args.nodes_list)
            nproc_per_node = (
                self.scheduler.num_sockets() if len(nodes_list) == 0 else len(nodes_list)
            )
        self.distributor.resolve(nproc_per_node, nnodes=1)

        # call super set_environment to make sure other things are set
        # since the distributor already resolved, it will not resolve twice
        super().resolve()

    def setup(self):
        """
        Set ENVs for launching MPI process for distributed training.
        """
        super().setup()

        nproc_per_node = self.distributor.nproc_per_node
        nodes_list = self.parse_list_argument(self.args.nodes_list)
        self.setup_mpi(
            self.scheduler, nodes_list, nproc_per_node)

    def setup_mpi(
            self, scheduler, nodes_list, nproc_per_node):
        args = self.args
        ncores_per_proc = args.ncores_per_proc
        if ncores_per_proc > 0:
            if (
                    not args.logical_cores_for_ccl
                    or len([c for c in scheduler.pool if not c.is_physical_core])
                    < nproc_per_node * args.ccl_worker_count
            ):
                ncores_per_proc += args.ccl_worker_count
            ncores_per_proc = len(
                [c for c in scheduler.pool if c.core < ncores_per_proc]
            )
        cpu_schedule = scheduler.schedule(
            num_proc=nproc_per_node,
            ncores_per_proc=ncores_per_proc,
            use_logical_cores=True,
            use_e_cores=args.use_e_cores,
            nodes_list=nodes_list,
        )

        self.set_memory_allocator(args.memory_allocator, False, ["jemalloc"])
        self.set_omp_runtime(args.omp_runtime, True)
        omp_num_threads = len(
            [c for c in cpu_schedule[0] if c.is_physical_core]
        )
        if not args.logical_cores_for_ccl:
            omp_num_threads -= args.ccl_worker_count
        self.add_env("OMP_NUM_THREADS", str(omp_num_threads))

        pin_domain_affinity = self.get_pin_domain_affinity(
            cpu_schedule,
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
