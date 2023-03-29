import logging
import os
import subprocess
import sys

import psutil

from cloudtik.runtime.ml.runner.cpu.launcher import Launcher
from cloudtik.runtime.ml.runner.cpu.utils import CPUinfo

logger = logging.getLogger(__name__)


class DistributedTrainingLauncher(Launcher):
    r"""
     Launcher for distributed training with MPI launcher
     """
    def get_mpi_pin_domain(self, nproc_per_node, ccl_worker_count, total_cores, flatten_node_cores):
        '''
        I_MPI_PIN_DOMAIN specify the cores used for every MPI process.
        The first ccl_worker_count cores of every rank for ccl communication
        and the other cores will be used to do computation.
        For example: on CascadeLake 8280 CPU, 2 ranks on one node. ccl_worker_count=4
        CCL_WORKER_COUNT=4
        CCL_WORKER_AFFINITY="0,1,2,3,28,29,30,31"
        I_MPI_PIN_DOMAIN=[0xffffff0,0xffffff0000000]
        '''
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
        '''
        Computation and communication use different cores when using oneCCL
        backend for distributed training. we use first ccl_worker_count cores of
        every rank for ccl communication
        '''
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

    def launch(self, args):
        '''
        Set ENVs and launch MPI process for distributed training.
        '''
        if not args.hosts and args.nnodes > 1 and not os.path.exists(args.hostfile):
            raise ValueError("hostfile is necessary when you use multi-node distributed training,"
                             "Please create hostfile which include the ip list you used for distributed running")
        elif not args.hosts and args.nnodes > 1:
            ip_list = []
            with open(args.hostfile) as f:
                for line in f:
                    line = line.strip().strip("\n")
                    ip_list.append(line)
            if len(ip_list) < args.nnodes:
                logger.error("The number of IP {} should greater than nnodes parameters {}".format(
                    len(ip_list), args.nnodes))
                exit(-1)
            master_check = False
            dic = psutil.net_if_addrs()
            for adapter in dic:
                snicList = dic[adapter]
                for snic in snicList:
                    if snic.address == ip_list[0]:
                        master_check = True
            if not master_check:
                logger.error(
                    "MASTER_ADDR is incorrect. "
                    "Please make sure the first line {} in your hostfile is ip address of the current node".format(
                        ip_list[0]))
                exit(-1)

            logger.info("Begin to validate the ip connect")
            args.master_addr = ip_list[0]
            for ip in ip_list[1:]:
                completed_process = subprocess.run("ssh -o PasswordAuthentication=no {} ':'".format(ip), shell=True)
                if completed_process.returncode != 0:
                    logger.error(
                        "Passwordless SSH login to {} failed, please make sure you have setup SSH public key right")
                    exit(-1)
                else:
                    logger.info("connection from master node {} to slave node {} is OK".format(args.master_addr, ip))
        else:
            # CloudTik: patch start
            if args.hosts:
                host_list = args.hosts.split(',')
                args.nnodes = len(host_list)
                args.master_addr = host_list[0]
            # CloudTik: patch end

        if not args.hosts:
            node_cores = self.cpuinfo.node_physical_cores
            total_cores_per_node = self.cpuinfo.physical_core_nums()
            if args.use_logical_core:
                node_cores = self.cpuinfo.node_logical_cores
                total_cores_per_node = self.cpuinfo.logical_core_nums()
            if args.nproc_per_node == 0:
                args.nproc_per_node = self.cpuinfo.node_nums()
        else:
            # CloudTik: patch start
            total_cores_per_node = args.cores_per_node
            host_list = args.hosts.split(',')
            remote_cpuinfo = CPUinfo(host_ip=host_list[-1])
            node_cores = remote_cpuinfo.node_physical_cores
            if args.use_logical_core:
                node_cores = remote_cpuinfo.node_logical_cores
            if not total_cores_per_node:
                total_cores_per_node = remote_cpuinfo.physical_core_nums()
                if args.use_logical_core:
                    total_cores_per_node = remote_cpuinfo.logical_core_nums()
            if args.nproc_per_node == 0:
                args.nproc_per_node = remote_cpuinfo.node_nums()
        flatten_node_cores = []
        for node_numa_cores in node_cores:
            flatten_node_cores.extend(node_numa_cores)
            # CloudTik: patch end

        # set distributed related environmental variables
        self.set_env("MASTER_ADDR", args.master_addr)
        self.set_env("MASTER_PORT", str(args.master_port))
        # CloudTik: patch start
        mpi_pin_domain = self.get_mpi_pin_domain(
            args.nproc_per_node, args.ccl_worker_count, total_cores_per_node, flatten_node_cores)
        # CloudTik: patch end
        self.set_env("I_MPI_PIN_DOMAIN", mpi_pin_domain)

        ppn = args.nproc_per_node
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
            args.nproc_per_node, args.ccl_worker_count, total_cores_per_node, flatten_node_cores)
        self.set_env("CCL_WORKER_AFFINITY", ccl_affinity)

        os.environ["LAUNCH_CMD"] = "#"
        cmd = ['mpiexec.hydra']
        mpi_config = "-l -np {} -ppn {} -genv I_MPI_PIN_DOMAIN={} -genv OMP_NUM_THREADS={} ".format(
            args.nnodes * args.nproc_per_node, args.nproc_per_node, mpi_pin_domain, omp_num_threads)
        mpi_config += args.more_mpi_params

        # CloudTik: patch start
        if args.hosts:
            mpi_config += " -hosts {}".format(args.hosts)
        else:
            if args.nnodes > 1:
                mpi_config += " -hostfile {}".format(args.hostfile)

        def get_cloudtik_rsh():
            cloudtik_home = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'cloudtik')
            return os.path.join(cloudtik_home, "runtime/ml/scripts", "cloudtik-rsh.sh")

        if "-launcher-exec" not in mpi_config:
            mpi_config += (
                ' -launcher rsh -launcher-exec "{launcher_exec}"'.format(
                    launcher_exec=get_cloudtik_rsh()))
        # CloudTik: patch end

        cmd.extend(mpi_config.split())
        with_python = not args.no_python
        if with_python:
            cmd.append(sys.executable)
            cmd.append("-u")
        if args.module:
            cmd.append("-m")
        cmd.append(args.program)
        cmd.extend(args.program_args)
        log_name = args.log_file_prefix + ".log"
        log_name = os.path.join(args.log_path, log_name)
        cmd_s = " ".join(cmd)
        if args.log_path:
            cmd_s = "{} 2>&1 | tee {}".format(cmd_s, log_name)
        logger.info(cmd_s)
        process = subprocess.Popen(cmd_s, env=os.environ, shell=True)
        process.wait()
        os.environ["LAUNCH_CMD"] += " ".join(cmd) + ",#"
        os.environ["LAUNCH_CMD"] = os.environ["LAUNCH_CMD"][:-2]
