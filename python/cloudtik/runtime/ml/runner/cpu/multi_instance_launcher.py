import logging
import os
import subprocess
import sys

from cloudtik.runtime.ml.runner.cpu.launcher import Launcher

logger = logging.getLogger(__name__)


class MultiInstanceLauncher(Launcher):
    r"""
     Launcher for single instance and multi-instance
     """
    def launch(self, args):
        processes = []
        cores = []
        set_kmp_affinity = True
        enable_taskset = False
        if args.core_list:  # user specify what cores will be used by params
            cores = [int(x) for x in args.core_list.split(",")]
            if args.ncore_per_instance == -1:
                logger.error("please specify the '--ncore_per_instance' if you have pass the --core_list params")
                exit(-1)
            elif args.ninstances > 1 and args.ncore_per_instance * args.ninstances < len(cores):
                logger.warning(
                    "only first {} cores will be used, but you specify {} cores in core_list".format(
                        args.ncore_per_instance * args.ninstances, len(cores)))
            else:
                args.ninstances = len(cores) // args.ncore_per_instance

        else:
            if args.use_logical_core:
                if args.node_id != -1:
                    cores = self.cpuinfo.get_node_logical_cores(args.node_id)
                else:
                    cores = self.cpuinfo.get_all_logical_cores()
                    # When using all cores on all nodes, including logical cores
                    # setting KMP_AFFINITY disables logical cores. Thus, KMP_AFFINITY should not be set.
                    set_kmp_affinity = False
            else:
                if args.node_id != -1:
                    cores = self.cpuinfo.get_node_physical_cores(args.node_id)
                else:
                    cores = self.cpuinfo.get_all_physical_cores()
            if not args.multi_instance and args.ninstances == -1 and args.ncore_per_instance == -1:
                args.ninstances = 1
                args.ncore_per_instance = len(cores)
            elif args.multi_instance and args.ninstances == -1 and args.ncore_per_instance == -1:
                args.throughput_mode = True
            elif args.ncore_per_instance == -1 and args.ninstances != -1:
                if args.ninstances > len(cores):
                    logger.error(
                        "there are {} total cores but you specify {} ninstances; "
                        "please make sure ninstances <= total_cores)".format(len(cores), args.ninstances))
                    exit(-1)
                else:
                    args.ncore_per_instance = len(cores) // args.ninstances
            elif args.ncore_per_instance != -1 and args.ninstances == -1:
                if not args.skip_cross_node_cores:
                    args.ninstances = len(cores) // args.ncore_per_instance
                else:
                    ncore_per_node = len(self.cpuinfo.node_physical_cores[0])
                    num_leftover_cores = ncore_per_node % args.ncore_per_instance
                    if args.ncore_per_instance > ncore_per_node:
                        # too many ncore_per_instance to skip cross-node cores
                        logger.warning("there are {} core(s) per socket, "
                                       "but you specify {} ncore_per_instance and skip_cross_node_cores. "
                                       "Please make sure --ncore_per_instance < core(s) per socket".format(
                            ncore_per_node, args.ncore_per_instance))
                        exit(-1)
                    elif num_leftover_cores == 0:
                        # aren't any cross-node cores
                        logger.info('--skip_cross_node_cores is set, but there are no cross-node cores.')
                        args.ninstances = len(cores) // args.ncore_per_instance
                    else:
                        # skip cross-node cores
                        if args.ninstances != -1:
                            logger.warning(
                                '--skip_cross_node_cores is exclusive to --ninstances. '
                                '--ninstances won\'t take effect even if it is set explicitly.')

                        i = 1
                        leftover_cores = set()
                        while ncore_per_node*i <= len(cores):
                            leftover_cores.update(cores[ncore_per_node*i-num_leftover_cores : ncore_per_node*i])
                            i += 1
                        cores = list(set(cores) - leftover_cores)
                        assert len(cores) % args.ncore_per_instance == 0
                        args.ninstances = len(cores) // args.ncore_per_instance
            else:
                if args.ninstances * args.ncore_per_instance > len(cores):
                    logger.error("Please make sure ninstances * ncore_per_instance <= total_cores")
                    exit(-1)
            if args.latency_mode:
                logger.warning('--latency_mode is exclusive to --ninstances, --ncore_per_instance, --node_id '
                               'and --use_logical_core. They won\'t take effect even if they are set explicitly.')
                args.ncore_per_instance = 4
                cores = self.cpuinfo.get_all_physical_cores()
                args.ninstances = len(cores) // args.ncore_per_instance

            if args.throughput_mode:
                logger.warning('--throughput_mode is exclusive to --ninstances, --ncore_per_instance, --node_id '
                               'and --use_logical_core. They won\'t take effect even if they are set explicitly.')
                args.ninstances = self.cpuinfo.node_nums()
                cores = self.cpuinfo.get_all_physical_cores()
                args.ncore_per_instance = len(cores) // args.ninstances

        if args.ninstances > 1 and args.instance_idx != -1:
            logger.info("assigning {} cores for instance {}".format(args.ncore_per_instance, args.instance_idx))

        if not args.disable_numactl:
            numactl_available = self.is_numactl_available()
            if not numactl_available:
                if not args.disable_taskset:
                    logger.warning(
                        "Core binding with numactl is not available. Disabling numactl and using taskset instead. "
                        "This may affect performance in multi-socket system; "
                        "please use numactl if memory binding is needed.")
                    args.disable_numactl = True
                    enable_taskset = True
                else:
                    logger.warning(
                        "Core binding with numactl is not available, and --disable_taskset is set. "
                        "Please unset --disable_taskset to use taskset insetad of numactl.")
                    exit(-1)

        if not args.disable_taskset:
            enable_taskset = True

        self.set_multi_thread_and_allocator(args.ncore_per_instance,
                                            args.disable_iomp,
                                            set_kmp_affinity,
                                            args.enable_tcmalloc,
                                            args.enable_jemalloc,
                                            args.use_default_allocator,
                                            args.benchmark)
        os.environ["LAUNCH_CMD"] = "#"

        if args.auto_ipex:
            import intel_extension_for_pytorch.cpu.auto_ipex as auto_ipex
            args.program = auto_ipex.apply_monkey_patch(
                args.program, args.dtype, args.auto_ipex_verbose, args.disable_ipex_graph_mode)

        for i in range(args.ninstances):
            cmd = []
            cur_process_cores = ""
            if not args.disable_numactl or enable_taskset:
                if not args.disable_numactl:
                    cmd = ["numactl"]
                elif enable_taskset:
                    cmd = ["taskset"]

                cores = sorted(cores)
                if args.instance_idx == -1:  # sequentially assign ncores_per_instance to ninstances
                    core_list = cores[i * args.ncore_per_instance: (
                        i + 1) * args.ncore_per_instance]
                else:  # assign ncores_per_instance from instance_idx
                    core_list = cores[args.instance_idx * args.ncore_per_instance: (
                        args.instance_idx + 1) * args.ncore_per_instance]

                core_ranges = []
                for core in core_list:
                    if len(core_ranges) == 0:
                        range_elem = {'start': core, 'end': core}
                        core_ranges.append(range_elem)
                    else:
                        if core - core_ranges[-1]['end'] == 1:
                            core_ranges[-1]['end'] = core
                        else:
                            range_elem = {'start': core, 'end': core}
                            core_ranges.append(range_elem)
                for r in core_ranges:
                    cur_process_cores = cur_process_cores + "{}-{},".format(r['start'], r['end'])
                cur_process_cores = cur_process_cores[:-1]

                if not args.disable_numactl:
                    numa_params = "-C {} ".format(cur_process_cores)
                    numa_params += "-m {}".format(",".join(
                        [str(numa_id) for numa_id in self.cpuinfo.numa_aware_check(core_list)]))
                    cmd.extend(numa_params.split())
                elif enable_taskset:
                    taskset_params = "-c {}".format(cur_process_cores)
                    cmd.extend(taskset_params.split())

            with_python = not args.no_python
            if with_python:
                cmd.append(sys.executable)
                cmd.append("-u")
            if args.module:
                cmd.append("-m")
            cmd.append(args.program)
            log_name = args.log_file_prefix + "_instance_{}_cores_".format(
                i) + cur_process_cores.replace(',', '_') + ".log"
            log_name = os.path.join(args.log_path, log_name)
            cmd.extend(args.program_args)
            os.environ["LAUNCH_CMD"] += " ".join(cmd) + ",#"
            cmd_s = " ".join(cmd)
            if args.log_path:
                cmd_s = "{} 2>&1 | tee {}".format(cmd_s, log_name)
            logger.info(cmd_s)
            if not args.disable_numactl:
                process = subprocess.Popen(cmd_s, env=os.environ, shell=True)
            elif enable_taskset:
                process = subprocess.Popen(cmd, env=os.environ)
            processes.append(process)

            if args.instance_idx != -1: # launches single instance, instance_idx, only
                break

        os.environ["LAUNCH_CMD"] = os.environ["LAUNCH_CMD"][:-2]
        try:
            for process in processes:
                process.wait()
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd_s)
        finally:
            if args.auto_ipex:
                # Clean the temp file
                if os.path.exists(args.program) and args.program.endswith("_auto_ipex"):
                    os.remove(args.program)
