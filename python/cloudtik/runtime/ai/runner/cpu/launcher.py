import glob
import logging
import os
from os.path import expanduser

from cloudtik.runtime.ai.runner.cpu.cpu_info import CPUPoolList

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
    def __init__(self, lscpu_txt=""):
        self.cpuinfo = CPUPoolList(logger, lscpu_txt)
        self.library_paths = self._get_library_paths()
        self.ma_supported = ["auto", "default", "tcmalloc", "jemalloc"]
        self.omp_supported = ["auto", "default", "intel"]
        self.ld_preload = (
            os.environ["LD_PRELOAD"].split(":") if "LD_PRELOAD" in os.environ else []
        )

    def add_env(self, env_name, env_value):
        raise NotImplementedError("The final launcher must implement this method.")

    @staticmethod
    def _get_library_paths():
        library_paths = []
        if "CONDA_PREFIX" in os.environ:
            library_paths.append(f'{os.environ["CONDA_PREFIX"]}/lib/')
        if "VIRTUAL_ENV" in os.environ:
            library_paths.append(f'{os.environ["VIRTUAL_ENV"]}/lib/')
        library_paths.extend(
            [
                f'{expanduser("~")}/.local/lib/',
                "/usr/local/lib/",
                "/usr/local/lib64/",
                "/usr/lib/",
                "/usr/lib64/",
                "/usr/lib/x86_64-linux-gnu/",
            ]
        )
        return library_paths

    def verbose(self, level, msg):
        logging_fn = {
            "warning": logger.warning,
            "info": logger.info,
        }
        assert (
            level in logging_fn.keys()
        ), f"Unrecognized logging level {level} is detected. Available levels are {logging_fn.keys()}."
        logging_fn[level](msg)

    def add_lib_preload(self, lib_type):
        """
        Enable TCMalloc/JeMalloc/intel OpenMP
        """
        lib_found = False
        lib_set = False
        for item in self.ld_preload:
            if item.endswith(f"lib{lib_type}.so"):
                lib_set = True
                break
        if not lib_set:
            for lib_path in self.library_paths:
                if lib_path.endswith("/"):
                    lib_path = lib_path[:-1]
                library_file = f"{lib_path}/lib{lib_type}.so"
                matches = glob.glob(library_file)
                if len(matches) > 0:
                    self.ld_preload.append(matches[0])
                    lib_found = True
                    break
        return lib_set or lib_found

    def set_lib_bin_from_list(
            self,
            name_input,
            name_map,
            category,
            supported,
            fn,
            skip_list=None,
            extra_warning_msg_with_default_choice="",
    ):
        """
        Function to set libraries or commands that are predefined in support lists.
        The support list is formed in format ['auto', default choice, alternative A, alternative B, ...].
        The list can only contain 'auto' and the default choice.
        Arguments:
            name_input: name of the lib/bin that user inputs.
            name_map: a dictionary. {'key': ['alias name', 'package installation command']} Its key is name of the
            lib/bin, its value is a list of string with 2 elements. First string of the list is alias name of the
            lib/bin that is searched in the system. For instance, when key is 'intel' for OpenMP runtime, the function
            will invoke fn (describe below) to search a library file 'libiomp5.so'.
            The fn function passed forms the library file name with its identifier 'iomp5'. Thus, the first string of
            this list for key 'intel' should be 'iomp5'. This value depends on how fn function searches for the lib/bin
            file.
            The second string should be a installation command guides users to install this package. When it is empty,
            the installation guide will not be prompted. category: category of this lib/bin. 'memory allocator',
            'multi-task manager', etc.
            supported: predefined support list
            fn: a function how the lib/bin files will be searched. Return True to indicate a successful searching,
            otherwise return False.
            skip_list: a list containing name of lib/bin that will not be used.
            extra_warning_msg_with_default_choice: a warning message that will be prompted if designated choices
            are not available and fallen back to the default choice.
        """
        if skip_list is None:
            skip_list = []
        name_local = name_input.lower()
        if name_local not in supported:
            name_local = supported[0]
            self.verbose(
                "warning",
                f"Designated {category} '{name_input}' is unknown. Changing it to '{name_local}'. \
                    Supported {category} are {supported}.",
            )
        if name_local in skip_list:
            name_local = supported[0]
            self.verbose(
                "warning",
                f"Designated {category} '{name_input}' is not applicable at this moment. Changing it to '{name_local}'\
                    . Please choose another {category} from {supported}.",
            )
        if name_local == supported[0]:
            for name in supported[2:]:
                if name in skip_list:
                    continue
                if fn(name_map[name][0]):
                    self.verbose("info", f"Use '{name_local}' => '{name}' {category}.")
                    name_local = name
                    break
            if name_local == supported[0]:
                name_local = supported[1]
                if len(supported[2:]) > 0:
                    if len(supported[2:]) == 1:
                        msg = f"'{supported[2]}' {category} is not found"
                    elif len(supported[2:]) < 3:
                        msg = f"Neither of {supported[2:]} {category} is found"
                    else:
                        msg = f"None of {supported[2:]} {category} is found"
                    self.verbose("warning", f"{msg} in {self.library_paths}.")
                if extra_warning_msg_with_default_choice != "":
                    extra_warning_msg_with_default_choice = (
                        f" {extra_warning_msg_with_default_choice}"
                    )
                self.verbose(
                    "info",
                    f"Use '{name_local}' {category}.{extra_warning_msg_with_default_choice}",
                )
        elif name_local in supported[2:]:
            if not fn(name_map[name_local][0]):
                extra_warning_msg_install_guide = ""
                if name_map[name_local][1] != "":
                    extra_warning_msg_install_guide = (
                        f' You can install it with "{name_map[name_local][1]}".'
                    )
                self.verbose(
                    "warning",
                    f"Unable to find the '{name_local}' {category} library file in {self.library_paths}.\
                        {extra_warning_msg_install_guide}",
                )
                name_local = supported[1]
                if extra_warning_msg_with_default_choice != "":
                    extra_warning_msg_with_default_choice = (
                        f" {extra_warning_msg_with_default_choice}"
                    )
                self.verbose(
                    "info",
                    f"Use '{name_local}' {category}.{extra_warning_msg_with_default_choice}",
                )
            else:
                self.verbose("info", f"Use '{name_local}' {category}.")
        else:
            self.verbose("info", f"Use '{name_local}' {category}.")
        if fn == self.add_lib_preload:
            for k, v in name_map.items():
                if k == name_local:
                    continue
                for item in self.ld_preload:
                    if item.endswith(f"lib{v[0]}.so"):
                        self.ld_preload.remove(item)
        return name_local

    def set_memory_allocator(
            self, memory_allocator="auto", benchmark=False, skip_list=None
    ):
        """
        Enable TCMalloc/JeMalloc with LD_PRELOAD and set configuration for JeMalloc.
        By default, PTMalloc will be used for PyTorch, but TCMalloc and JeMalloc can get better
        memory resue and reduce page fault to improve performance.
        """
        if skip_list is None:
            skip_list = []
        ma_lib_name = {
            "jemalloc": ["jemalloc", "conda install -c conda-forge jemalloc"],
            "tcmalloc": ["tcmalloc", "conda install -c conda-forge gperftools"],
        }
        ma_local = self.set_lib_bin_from_list(
            memory_allocator,
            ma_lib_name,
            "memory allocator",
            self.ma_supported,
            self.add_lib_preload,
            skip_list=skip_list,
            extra_warning_msg_with_default_choice="This may drop the performance.",
        )
        if ma_local == "jemalloc":
            if benchmark:
                self.add_env(
                    "MALLOC_CONF",
                    "oversize_threshold:1,background_thread:false,metadata_thp:always,dirty_decay_ms:-1,muzzy_decay_ms:-1",
                )
            else:
                self.add_env(
                    "MALLOC_CONF",
                    "oversize_threshold:1,background_thread:true,metadata_thp:auto",
                )
        return ma_local

    def set_omp_runtime(self, omp_runtime="auto", set_kmp_affinity=True):
        """
        Set OpenMP runtime
        """
        omp_lib_name = {"intel": ["iomp5", "conda install intel-openmp"]}
        omp_local = self.set_lib_bin_from_list(
            omp_runtime,
            omp_lib_name,
            "OpenMP runtime",
            self.omp_supported,
            self.add_lib_preload,
        )
        if omp_local == "intel":
            if set_kmp_affinity:
                self.add_env("KMP_AFFINITY", "granularity=fine,compact,1,0")
            self.add_env("KMP_BLOCKTIME", "1")
        elif omp_local == "default":
            self.add_env("OMP_SCHEDULE", "STATIC")
            self.add_env("OMP_PROC_BIND", "CLOSE")
        return omp_local

    def parse_list_argument(self, txt):
        ret = []
        txt = txt.strip()
        if txt != "":
            for elem in txt.split(","):
                elem = elem.strip()
                if elem.isdigit():
                    ret.append(int(elem))
                else:
                    core_range = [int(x.strip()) for x in elem.split("-")]
                    assert len(core_range) == 2, "Invalid range format detected."
                    begin, end = core_range
                    assert (
                            begin <= end
                    ), "Begining index of a range must be <= ending index."
                    ret.extend(list(range(begin, end + 1)))
        ret = list(set(ret))
        return ret
