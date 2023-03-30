import logging
import os
import platform
import re
import subprocess

import numpy as np

logger = logging.getLogger(__name__)

# List of regular expressions to ignore environment variables by.
IGNORE_REGEXES = {'BASH_FUNC_.*', 'OLDPWD', '.*_SECRET_KEY'}


def is_exportable(v):
    return not any(re.match(r, v) for r in IGNORE_REGEXES)


class CPUinfo:
    """
    Get CPU information, such as cores list and NUMA information.
    If host_ip is not None, we will use `cloudtik head exec --node-ip ip` command to get lscpu info.
    """
    def __init__(self, host_ip=None):
        self.cpuinfo = []
        self.nodes = 0
        self.node_physical_cores = []  # node_id is index
        self.node_logical_cores = []   # node_id is index
        self.physical_core_node_map = {}  # physical core to numa node id
        self.logical_core_node_map = {}   # logical core to numa node id
        if platform.system() == "Windows":
            raise RuntimeError("Windows platform is not supported!!!")
        elif platform.system() == "Linux":
            # CloudTik: patch start
            if host_ip is None:
                args = ["lscpu", "--parse=CPU,Core,Socket,Node"]
            else:
                args = ["cloudtik", "head", "exec", "--node-ip", host_ip, "lscpu --parse=CPU,Core,Socket,Node"]
            # CloudTik: patch end
            env_lang = os.getenv('LANG', 'UNSET')
            os.environ['LANG'] = 'C'
            lscpu_info = subprocess.check_output(args, env=os.environ, universal_newlines=True).split("\n")
            if env_lang == 'UNSET':
                del os.environ['LANG']
            else:
                os.environ['LANG'] = env_lang

            # Get information about  cpu, core, socket and node
            for line in lscpu_info:
                pattern = r"^([\d]+,[\d]+,[\d]+,[\d]?)"
                regex_out = re.search(pattern, line)
                if regex_out:
                    self.cpuinfo.append(regex_out.group(1).strip().split(","))
            assert len(self.cpuinfo) > 0, "cpuinfo is empty"
            self.get_socket_info()

    def get_socket_info(self):
        idx_active = 3
        if self.cpuinfo[0][idx_active] == '':
            idx_active = 2
        self.nodes = int(max([line[idx_active] for line in self.cpuinfo])) + 1
        self.node_physical_cores = []  # node_id is index
        self.node_logical_cores = []   # node_id is index
        self.physical_core_node_map = {}  # physical core to numa node id
        self.logical_core_node_map = {}   # logical core to numa node id

        for node_id in range(self.nodes):
            cur_node_physical_core = []
            cur_node_logical_core = []
            # CloudTik: patch start
            cur_node_physical_core_cpu_id = []
            # CloudTik: patch end
            for line in self.cpuinfo:
                nid = line[idx_active] if line[idx_active] != '' else '0'
                if node_id == int(nid):
                    if int(line[1]) not in cur_node_physical_core:
                        cur_node_physical_core.append(int(line[1]))
                        # CloudTik: patch start
                        cur_node_physical_core_cpu_id.append(int(line[0]))
                        # CloudTik: patch end
                        self.physical_core_node_map[int(line[1])] = int(node_id)
                    cur_node_logical_core.append(int(line[0]))
                    self.logical_core_node_map[int(line[0])] = int(node_id)
            # CloudTik: patch start
            # self.node_physical_cores.append(cur_node_physical_core)
            self.node_physical_cores.append(cur_node_physical_core_cpu_id)
            # CloudTik: patch end
            self.node_logical_cores.append(cur_node_logical_core)

    def node_nums(self):
        return self.nodes

    def physical_core_nums(self):
        return len(self.node_physical_cores) * len(self.node_physical_cores[0])

    def logical_core_nums(self):
        return len(self.node_logical_cores) * len(self.node_logical_cores[0])

    def get_node_physical_cores(self, node_id):
        if node_id < 0 or node_id > self.nodes - 1:
            logger.error("Invalid node id")
        return self.node_physical_cores[node_id]

    def get_node_logical_cores(self, node_id):
        if node_id < 0 or node_id > self.nodes - 1:
            logger.error("Invalid node id")
        return self.node_logical_cores[node_id]

    def get_all_physical_cores(self):
        return np.array(self.node_physical_cores).flatten().tolist()

    def get_all_logical_cores(self):
        return np.array(self.node_logical_cores).flatten().tolist()

    def numa_aware_check(self, core_list):
        """
        Check whether all cores in core_list are in the same NUMA node. cross NUMA will reduce performance.
        We strongly advise to not use cores on different nodes.
        """
        cores_numa_map = self.logical_core_node_map
        numa_ids = []
        for core in core_list:
            numa_id = cores_numa_map[core]
            if not numa_id in numa_ids:
                numa_ids.append(numa_id)
        if len(numa_ids) > 1:
            logger.warning("Numa Aware: cores:{} on different NUMA nodes:{}".format(str(core_list), str(numa_ids)))
        if len(numa_ids) == 0:
            logger.error("invalid number of NUMA nodes; please make sure numa_ids >= 1")
            exit(-1)
        return numa_ids
