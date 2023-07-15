import logging
import os
import subprocess

from cloudtik.runtime.ai.runner.launcher import Launcher
from cloudtik.runtime.ai.runner.cpu.cpu_launcher import CPULauncher
from cloudtik.runtime.ai.runner.util.utils import is_python_program

logger = logging.getLogger(__name__)

TASK_MANAGERS = ["auto", "none", "numactl", "taskset"]


def add_local_cpu_launcher_params(parser):
    group = parser.add_argument_group("Local CPU Launching Parameters")
    group.add_argument(
        "--process-idx", "--process_idx",
        default="", type=str,
        help="Specify process index or a list of process indices to run. "
             "If it is set to -1 or empty, will run all the processes.")
    group.add_argument(
        "--nodes-list", "--nodes_list",
        default="", type=str,
        help='Specify nodes list for multiple processes to run on, in format of list of single node ids '
             'node_id,node_id,..." or list of node ranges "node_id-node_id,...". By default all nodes will be used.',
    )
    group.add_argument(
        "--cores-list", "--cores_list",
        default="", type=str,
        help='Specify cores list for multiple processes to run on, in format of list of single core ids '
             'core_id,core_id,..." or list of core ranges "core_id-core_id,...". '
             'By default all cores will be used.',
    )
    group.add_argument(
        "--task-manager", "--task_manager",
        default="auto", type=str, choices=TASK_MANAGERS,
        help=f"Choose which task manager to run the workloads with. Supported choices are {TASK_MANAGERS}.", )
    group.add_argument(
        "--skip-cross-node-cores", "--skip_cross_node_cores",
        action='store_true', default=False,
        help="If specified --ncores-per-proc, skips cross-node cores.")
    group.add_argument(
        "--latency-mode", "--latency_mode",
        action='store_true', default=False,
        help="By default 4 core per procedss and use all physical cores")
    group.add_argument(
        "--throughput-mode", "--throughput_mode",
        action='store_true', default=False,
        help="By default one procedss per node and use all physical cores")
    group.add_argument(
        "--benchmark",
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


class LocalCPULauncher(Launcher, CPULauncher):
    r"""
     Launcher for one or more procedss on local machine
     """

    def __init__(self, args, distributor):
        Launcher.__init__(self, args, distributor)
        CPULauncher.__init__(self)
        self.program = None

    def add_env(self, env_name, env_value):
        self.set_env(env_name, env_value)

    def is_command_available(self, cmd):
        is_available = False
        try:
            cmd_s = ["which", cmd]
            r = subprocess.run(
                cmd_s,
                env=os.environ,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if r.returncode == 0:
                is_available = True
        except FileNotFoundError as e:
            pass
        return is_available

    def set_task_manager(self, task_manager="auto", skip_list=None):
        """
        Set task manager
        """
        if skip_list is None:
            skip_list = []
        tm_bin_name = {
            "numactl": ["numactl", ""],
            "taskset": ["taskset", ""],
        }
        tm_local = self.set_lib_bin_from_list(
            task_manager,
            tm_bin_name,
            "task manager",
            TASK_MANAGERS,
            self.is_command_available,
            skip_list,
        )
        return tm_local

    def run_process(
        self, args, omp_runtime, task_mgr, environ, cpu_pools, index
    ):
        assert index > -1 and index <= len(
            cpu_pools
        ), "Designated procedss index for constructing execution commands is out of range."
        cmd = []
        environ_local = environ
        pool = cpu_pools[index]
        pool_str = pool.get_pool_str()
        cores_list_local = pool_str["cores"]
        nodes_list_local = pool_str["nodes"]
        if task_mgr != TASK_MANAGERS[1]:
            params = ""
            if task_mgr == "numactl":
                params = f"-C {cores_list_local} "
                params += f"-m {nodes_list_local}"
            elif task_mgr == "taskset":
                params = f"-c {cores_list_local}"
            cmd.append(task_mgr)
            cmd.extend(params.split())
        else:
            k = ""
            v = ""
            if omp_runtime == "default":
                k = "GOMP_CPU_AFFINITY"
                v = cores_list_local
            elif omp_runtime == "intel":
                k = "KMP_AFFINITY"
                v = f"granularity=fine,proclist=[{cores_list_local}],explicit"
            if k != "":
                self.verbose("info", "==========")
                self.verbose("info", f"env: {k}={v}")
                environ_local[k] = v

        self.with_python_command(cmd)
        if self.program:
            cmd.append(self.program)
            cmd.extend(args.command[1:])
        else:
            cmd.extend(args.command)

        cmd_s = " ".join(cmd)
        if args.log_dir:
            log_name = f'{args.log_file_prefix}_instance_{index}_cores_{cores_list_local.replace(",", "_")}.log'
            log_file = os.path.join(args.log_dir, log_name)
            cmd_s = f"{cmd_s} 2>&1 | tee {log_file}"
        self.verbose("info", f"cmd: {cmd_s}")
        if len(set([c.node for c in pool])) > 1:
            self.verbose(
                "warning",
                f"Cross NUMA nodes execution detected: cores [{cores_list_local}] are on different NUMA nodes [{nodes_list_local}]",
            )
        process = subprocess.Popen(cmd_s, env=environ_local, shell=True)
        return {"process": process, "cmd": cmd_s}

    def launch(self):
        args = self.args

        if args.latency_mode and args.throughput_mode:
            raise RuntimeError(
                "Argument latency_mode and throughput_mode cannot be set at the same time."
            )
        if args.latency_mode:
            if (
                args.num_proc
                or args.ncores_per_proc > 0
                or len(args.nodes_list) > 0
                or args.use_logical_cores
            ):
                self.verbose(
                    "warning",
                    "--latency-mode is exclusive to --num-proc, --ncores-per-proc, --nodes-list and \
                        --use-logical-cores. They won't take effect even if they are set explicitly.",
                )
            args.ncores_per_proc = 4
            args.num_proc = 0
            args.use_logical_cores = False
        elif args.throughput_mode:
            if (
                args.num_proc
                or args.ncores_per_proc > 0
                or len(args.nodes_list) > 0
                or args.use_logical_cores
            ):
                self.verbose(
                    "warning",
                    "--throughput-mode is exclusive to --num-proc, --ncores-per-proc, --nodes-list and \
                        --use-logical-cores. They won't take effect even if they are set explicitly.",
                )
            args.num_proc = len(set([c.node for c in self.cpuinfo.pool]))
            args.ncores_per_proc = 0
            args.use_logical_cores = False

        cores_list = self.parse_list_argument(args.cores_list)
        nodes_list = self.parse_list_argument(args.nodes_list)

        cpu_schedule = self.cpuinfo.schedule(
            num_proc=args.num_proc,
            ncores_per_proc=args.ncores_per_proc,
            use_logical_cores=args.use_logical_cores,
            use_e_cores=args.use_e_cores,
            skip_cross_node_cores=args.skip_cross_node_cores,
            nodes_list=nodes_list,
            cores_list=cores_list,
        )
        args.num_proc = len(cpu_schedule)
        args.ncores_per_proc = len(cpu_schedule[0])

        is_iomp_set = False
        for item in self.ld_preload:
            if item.endswith("libiomp5.so"):
                is_iomp_set = True
                break
        is_kmp_affinity_set = True if "KMP_AFFINITY" in os.environ else False
        set_kmp_affinity = True
        # When using all cores on all nodes, including logical cores, setting KMP_AFFINITY disables logical cores. \
        #   Thus, KMP_AFFINITY should not be set.
        if args.use_logical_cores and len(
            set([c for p in cpu_schedule for c in p])
        ) == len(self.cpuinfo.pool):
            assert (
                not is_kmp_affinity_set
            ), 'Environment variable "KMP_AFFINITY" is detected. Please unset it when using all cores.'
            set_kmp_affinity = False

        self.set_memory_allocator(args.memory_allocator, args.benchmark)
        omp_runtime = self.set_omp_runtime(args.omp_runtime, set_kmp_affinity)
        self.add_env("OMP_NUM_THREADS", str(args.ncores_per_proc))

        skip_list = []
        if is_iomp_set and is_kmp_affinity_set:
            skip_list.append("numactl")
        task_mgr = self.set_task_manager(
            args.task_manager, skip_list=skip_list
        )

        # Set environment variables for multi-process execution
        self.verbose(
            "info", "env: Untouched preset environment variables are not displayed."
        )
        environ_local = {}
        for k, v in os.environ.items():
            if k == "LD_PRELOAD":
                continue
            environ_local[k] = v
        if len(self.ld_preload) > 0:
            environ_local["LD_PRELOAD"] = ":".join(self.ld_preload)
            self.verbose("info", f'env: LD_PRELOAD={environ_local["LD_PRELOAD"]}')
        for k, v in self.environ_set.items():
            if task_mgr == TASK_MANAGERS[1]:
                if omp_runtime == "default" and k == "GOMP_CPU_AFFINITY":
                    continue
                if omp_runtime == "intel" and k == "KMP_AFFINITY":
                    continue
            self.verbose("info", f"env: {k}={v}")
            environ_local[k] = v

        if args.auto_ipex and is_python_program(args.command):
            import intel_extension_for_pytorch.cpu.auto_ipex as auto_ipex
            program = args.command[0]
            self.program = auto_ipex.apply_monkey_patch(
                program,
                args.dtype,
                args.auto_ipex_verbose,
                args.disable_ipex_graph_mode,
            )

        processes_available = list(range(args.num_proc))
        process_idx = self.parse_list_argument(args.process_idx)
        if -1 in process_idx:
            process_idx.clear()
        if len(process_idx) == 0:
            process_idx.extend(processes_available)
        process_idx.sort()
        process_idx = list(set(process_idx))
        assert set(process_idx).issubset(
            set(processes_available)
        ), "Designated nodes list contains invalid nodes."
        processes = []
        for i in process_idx:
            process = self.run_process(
                args=args,
                omp_runtime=omp_runtime,
                task_mgr=task_mgr,
                environ=environ_local,
                cpu_pools=cpu_schedule,
                index=i,
            )
            processes.append(process)
        try:
            for process in processes:
                p = process["process"]
                p.wait()
                if p.returncode != 0:
                    raise subprocess.CalledProcessError(
                        returncode=p.returncode, cmd=process["cmd"]
                    )
        finally:
            if args.auto_ipex:
                # Clean the temp file
                if self.program and os.path.exists(
                        self.program) and self.program.endswith("_auto_ipex"):
                    os.remove(self.program)
