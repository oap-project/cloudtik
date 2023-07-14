import itertools
import os
import platform
import re
import subprocess

# lscpu Examples
# # The following is the parsable format, which can be fed to other
# # programs. Each different item in every column has an unique ID
# # starting from zero.
# CPU NODE SOCKET CORE L1d:L1i:L2:L3 ONLINE    MAXMHZ   MINMHZ      MHZ
#   0    0      0    0 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   1    0      0    0 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   2    0      0    1 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   3    0      0    1 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   4    1      1    2 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   5    1      1    2 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   6    1      1    3 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   7    1      1    3 0:0:0:0          yes 5000.0000 800.0000 2400.000

# CPU SOCKET CORE L1d:L1i:L2:L3 ONLINE    MAXMHZ   MINMHZ      MHZ
#   0      0    0 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   1      0    0 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   2      0    1 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   3      0    1 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   4      0    2 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   5      0    2 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   6      0    3 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   7      0    3 0:0:0:0          yes 5000.0000 800.0000 2400.000

# CPU NODE SOCKET CORE L1d:L1i:L2:L3 ONLINE    MAXMHZ   MINMHZ      MHZ
#   0    0      0    0 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   1    0      0    1 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   2    1      1    2 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   3    1      1    3 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   4    0      0    0 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   5    0      0    1 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   6    1      1    2 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   7    1      1    3 0:0:0:0          yes 5000.0000 800.0000 2400.000

# CPU NODE SOCKET CORE L1d:L1i:L2:L3 ONLINE    MAXMHZ   MINMHZ      MHZ
#   0    0      0    0 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   1    0      0    0 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   2    0      0    1 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   3    0      0    1 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   4    0      0    2 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   5    0      0    2 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   6    0      0    3 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   7    0      0    3 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   8    0      0    4 0:0:0:0          yes 3800.0000 800.0000 2400.000
#   9    0      0    5 0:0:0:0          yes 3800.0000 800.0000 2400.000
#  10    0      0    6 0:0:0:0          yes 3800.0000 800.0000 2400.000
#  11    0      0    7 0:0:0:0          yes 3800.0000 800.0000 2400.000


class CoreInfo:
    """
    Class to store core-specific information, including:
    - [int] CPU index
    - [int] Core index
    - [int] Numa node index
    - [int] Socket index
    - [bool] is a physical core or not
    - [float] maxmhz
    - [bool] is a performance core
    """

    def __init__(self, lscpu_txt="", headers=None):
        if headers is None:
            headers = {}
        self.cpu = -1
        self.core = -1
        self.socket = -1
        self.node = -1
        self.is_physical_core = True
        self.maxmhz = 0
        self.is_p_core = True
        if lscpu_txt != "" and len(headers) > 0:
            self.parse_raw(lscpu_txt, headers)

    def parse_raw(self, cols, headers):
        self.cpu = int(cols[headers["cpu"]])
        self.core = int(cols[headers["core"]])
        if "node" in headers:
            self.node = int(cols[headers["node"]])
            self.socket = int(cols[headers["socket"]])
        else:
            self.node = int(cols[headers["socket"]])
            self.socket = int(cols[headers["socket"]])
        if "maxmhz" in headers:
            self.maxmhz = float(cols[headers["maxmhz"]])

    def __str__(self):
        return f"{self.cpu}\t{self.core}\t{self.socket}\t{self.node}\t{self.is_physical_core}\t{self.maxmhz}\t{self.is_p_core}"


class CPUPool(list):
    """
    List of CoreInfo objects
    """

    def __init__(self):
        super(CPUPool, self).__init__()

    def get_ranges(self, l):
        for a, b in itertools.groupby(enumerate(l), lambda pair: pair[1] - pair[0]):
            b = list(b)
            yield b[0][1], b[-1][1]

    def get_pool_txt(self, return_mode="auto"):
        cpu_ids = [c.cpu for c in self]
        cpu_ranges = list(self.get_ranges(cpu_ids))
        cpu_ids_txt = ",".join([str(c) for c in cpu_ids])
        cpu_ranges_txt = ",".join([f"{r[0]}-{r[1]}" for r in cpu_ranges])
        node_ids_txt = ",".join(
            [str(n) for n in sorted(list(set([c.node for c in self])))]
        )
        ret = {"cores": "", "nodes": node_ids_txt}
        if return_mode.lower() == "list":
            ret["cores"] = cpu_ids_txt
        elif return_mode.lower() == "range":
            ret["cores"] = cpu_ranges_txt
        else:
            if len(cpu_ids) <= len(cpu_ranges):
                ret["cores"] = cpu_ids_txt
            else:
                ret["cores"] = cpu_ranges_txt
        return ret


class CPUPoolList:
    """
    Get a CPU pool with all available CPUs and CPU pools filtered with designated criterias.
    """

    def __init__(self, logger=None, lscpu_txt="", host_ip=None):
        self.pool_all = CPUPool()
        self.pools_ondemand = []

        self.logger = logger
        if platform.system() == "Windows":
            raise RuntimeError("Windows platform is not supported!!!")
        elif platform.system() == "Linux":
            """
            Retrieve CPU information from lscpu.
            """
            if lscpu_txt.strip() == "":
                if host_ip is None:
                    args = ["lscpu", "--all", "--extended"]
                else:
                    args = ["cloudtik", "head", "exec", "--node-ip", host_ip, "lscpu --all --extended"]
                env_lang = os.getenv("LANG", "UNSET")
                os.environ["LANG"] = "C"
                lscpu_info = subprocess.check_output(
                    args, env=os.environ, universal_newlines=True
                )
                if env_lang == "UNSET":
                    del os.environ["LANG"]
                else:
                    os.environ["LANG"] = env_lang
            else:
                lscpu_info = lscpu_txt

            """
            Filter out lines that are really useful.
            """
            lscpu_info = lscpu_info.strip().split("\n")
            headers = {}
            num_cols = 0
            for line in lscpu_info:
                line = re.sub(" +", " ", line.lower().strip())
                if "cpu" in line and "socket" in line and "core" in line:
                    t = line.split(" ")
                    num_cols = len(t)
                    for i in range(num_cols):
                        if t[i] in ["cpu", "core", "socket", "node", "maxmhz"]:
                            headers[t[i]] = i
                else:
                    t = line.split(" ")
                    if (
                        len(t) == num_cols
                        and t[headers["cpu"]].isdigit()
                        and t[headers["core"]].isdigit()
                        and t[headers["socket"]].isdigit()
                    ):
                        self.pool_all.append(CoreInfo(t, headers))
            assert len(self.pool_all) > 0, "cpuinfo is empty"

        # Determine logical cores
        core_cur = -1
        self.pool_all.sort(key=lambda x: (x.core, x.cpu))
        for c in self.pool_all:
            if core_cur != c.core:
                core_cur = c.core
            else:
                c.is_physical_core = False
        self.pool_all.sort(key=lambda x: x.cpu)

        # Determine e cores
        maxmhzs = list(set([c.maxmhz for c in self.pool_all]))
        maxmhzs.sort()
        mmaxmhzs = max(maxmhzs)
        if mmaxmhzs > 0:
            maxmhzs_norm = [f / mmaxmhzs for f in maxmhzs]
            separator_idx = -1
            for i in range(1, len(maxmhzs_norm)):
                if maxmhzs_norm[i] - maxmhzs_norm[i - 1] >= 0.15:
                    separator_idx = i
                    break
            if separator_idx > -1:
                e_core_mhzs = maxmhzs[:separator_idx]
                for c in self.pool_all:
                    if c.maxmhz in e_core_mhzs:
                        c.is_p_core = False

    def verbose(self, level, msg):
        if self.logger:
            logging_fn = {
                "warning": self.logger.warning,
                "info": self.logger.info,
            }
            assert (
                level in logging_fn.keys()
            ), f"Unrecognized logging level {level} is detected. Available levels are {logging_fn.keys()}."
            logging_fn[level](msg)
        else:
            print(msg)

    """
    Get CPU pools from all available CPU cores with designated criterias.
    - ninstances [int]: Number of instances. Should be a non negative integer, 0 by default. \
        When it is 0, it will be set according to usage scenarios automatically in the function.
    - ncores_per_instance [int]: Number of cores per instance. Should be a non negative integer, 0 by default. \
        When it is 0, it will be set according to usage scenarios automatically in the function.
    - use_logical_cores [bool]: Use logical cores on the workloads or not, False by default. When set to False, \
        only physical cores are used.
    - use_e_cores [bool]: Use Efficient-Cores, False by default. When set to False, only Performance-Cores are used.
    - skip_cross_node_cores [bool]: Allow instances to be executed on cores across NUMA nodes, False by default.
    - nodes_list [list]: A list containing all node ids that the execution is expected to be running on.
    - cores_list [list]: A list containing all cpu ids that the execution is expected to be running on.
    - return_mode [str]: A string that defines how result values are formed, could be either of 'auto', \
        'list' and 'range'. When set to 'list', a string with comma-separated cpu ids, '0,1,2,3,...', is returned. \
        When set to 'range', a string with comma-separated cpu id ranges, '0-2,6-8,...', is returned. \
        When set to 'auto', a 'list' or a 'range' whoever has less number of elements that are separated by \
        comma is returned. I.e. for a list '0,1,2,6,7,8' and a range '0-2,6-8', both reflect the same cpu \
        configuration, the range '0-2,6-8' is returned.
    """

    def gen_pools_ondemand(
        self,
        ninstances=0,
        ncores_per_instance=0,
        use_logical_cores=False,
        use_e_cores=False,
        skip_cross_node_cores=False,
        nodes_list=None,
        cores_list=None,
        return_mode="auto",
    ):
        if nodes_list is None:
            nodes_list = []
        if cores_list is None:
            cores_list = []

        # Generate an aggregated CPU pool
        if len(cores_list) > 0:
            cores_available = [c.cpu for c in self.pool_all]
            assert set(cores_list).issubset(
                set(cores_available)
            ), f"Designated cores list {cores_list} contains invalid cores."
            if use_logical_cores:
                self.verbose(
                    "warning",
                    "Argument --use-logical-cores won't take effect when --cores-list is set.",
                )
            if use_e_cores:
                self.verbose(
                    "warning",
                    "Argument --use-e-cores won't take effect when --cores-list is set.",
                )
            pool = [c for c in self.pool_all if c.cpu in cores_list]
            nodes = list(set([c.node for c in pool]))
            ncores_per_node = -1
            for n in nodes:
                ncores_local = len([c for c in pool if c.node == n])
                if ncores_per_node == -1:
                    ncores_per_node = ncores_local
                else:
                    if ncores_per_node != ncores_local and skip_cross_node_cores:
                        skip_cross_node_cores = False
                        self.verbose(
                            "warning",
                            "Argument --skip-cross-node-cores cannot take effect on the designated cores. Disabled.",
                        )
                        break
        else:
            if len(nodes_list) > 0:
                nodes_available = set([c.node for c in self.pool_all])
                assert set(nodes_list).issubset(
                    nodes_available
                ), f"Designated nodes list {nodes_list} contains invalid nodes out from {nodes_available}."
                pool = [c for c in self.pool_all if c.node in nodes_list]
            else:
                pool = self.pool_all
            if not use_logical_cores:
                pool = [c for c in pool if c.is_physical_core]
            if not use_e_cores:
                pool = [c for c in pool if c.is_p_core]
                e_cores = [c.cpu for c in pool if not c.is_p_core]
                if len(e_cores) > 0:
                    self.verbose(
                        "warning",
                        f"Efficient-Cores are detected ({e_cores}). Disabled for performance consideration. \
                            You can enable them with argument --use-e-cores.",
                    )

        # Determine ninstances and ncores_per_instance for grouping
        assert (
            ncores_per_instance >= 0
        ), "Argument --ncores-per-instance cannot be a negative value."
        assert ninstances >= 0, "Argument --ninstances cannot be a negative value."
        nodes = set([c.node for c in pool])
        if ncores_per_instance + ninstances == 0:
            # Both ncores_per_instance and ninstances are 0
            ninstances = 1
        if ncores_per_instance * ninstances == 0:
            # Either ncores_per_instance or ninstances is 0
            if skip_cross_node_cores:
                ncores_per_node = len(pool) // len(nodes)
                nresidual = 0
                if ncores_per_instance == 0:
                    nins_per_node = ninstances // len(nodes)
                    if ninstances % len(nodes) > 0:
                        nins_per_node += 1
                    ncores_per_instance = ncores_per_node // nins_per_node
                    nresidual = ncores_per_node % nins_per_node
                if ninstances == 0:
                    ninstances = ncores_per_node // ncores_per_instance * len(nodes)
                    nresidual = ncores_per_node % ncores_per_instance
                if nresidual > 0:
                    cores_remove = []
                    for n in nodes:
                        cores = [c for c in pool if c.node == n]
                        for i in range(nresidual):
                            cores_remove.append(cores[-1 * (i + 1)])
                    for c in cores_remove:
                        pool.remove(c)
            else:
                if ninstances == 0:
                    ninstances = len(pool) // ncores_per_instance
                if ncores_per_instance == 0:
                    ncores_per_instance = len(pool) // ninstances
        else:
            # Neither ncores_per_instance nor ninstances is 0
            if skip_cross_node_cores:
                self.verbose(
                    "warning",
                    "Argument --skip-cross-node-cores won't take effect when both --ninstances and \
                        --ncores-per-instance are explicitly set.",
                )
        assert (
            ninstances * ncores_per_instance > 0
            and ninstances * ncores_per_instance <= len(pool)
        ), "Requested number of cores exceeds what is available."

        # Split the aggregated pool into individual pools
        self.pools_ondemand.clear()
        pool.sort(key=lambda x: (x.core, 1 - int(x.is_physical_core)))
        for i in range(ninstances):
            # Generate individual raw pool
            pool_local = CPUPool()
            for j in range(ncores_per_instance):
                pool_local.append(pool[i * ncores_per_instance + j])
            pool_local.sort(key=lambda x: x.cpu)
            self.pools_ondemand.append(pool_local)

    def num_sockets(self):
        return len(set([c.node for c in self.pool_all]))
