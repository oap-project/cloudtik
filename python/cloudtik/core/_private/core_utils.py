import base64
import collections
import errno
import hashlib
import importlib
import json
import logging
import multiprocessing
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
from contextlib import closing
from typing import Optional

import ipaddr
# Import psutil after others so the packaged version is used.
import psutil

_find_unsafe = re.compile(r'[^\w@%+=:,./-]', re.ASCII).search

# Linux can bind child processes' lifetimes to that of their parents via prctl.
# prctl support is detected dynamically once, and assumed thereafter.
linux_prctl = None

# Windows can bind processes' lifetimes to that of kernel-level "job objects".
# We keep a global job object to tie its lifetime to that of our own process.
win32_job = None
win32_AssignProcessToJobObject = None

logger = logging.getLogger(__name__)

MEMORY_UNIT_GB = 1024 * 1024 * 1024


class ConcurrentCounter:
    def __init__(self):
        self._lock = threading.RLock()
        self._counter = collections.defaultdict(int)

    def inc(self, key, count):
        with self._lock:
            self._counter[key] += count
            return self.value

    def dec(self, key, count):
        with self._lock:
            self._counter[key] -= count
            assert self._counter[key] >= 0, "counter cannot go negative"
            return self.value

    def breakdown(self):
        with self._lock:
            return dict(self._counter)

    @property
    def value(self):
        with self._lock:
            return sum(self._counter.values())

    def lock(self):
        return self._lock


def load_class(path):
    """Load a class at runtime given a full path.

    Example of the path: mypkg.mysubpkg.myclass
    """
    class_data = path.split(".")
    if len(class_data) < 2:
        raise ValueError(
            "You need to pass a valid path like mymodule.class")
    module_path = ".".join(class_data[:-1])
    class_str = class_data[-1]
    module = importlib.import_module(module_path)
    return getattr(module, class_str)


def double_quote(s):
    """Return a shell-escaped version of the string *s* with double quote."""
    if not s:
        return '""'
    if _find_unsafe(s) is None:
        return s

    # use single quotes, and put single quotes into double quotes
    # the string $"b is then quoted as "$"'"'"b"
    return '"' + s.replace('"', '"\'"\'"') + '"'


def try_make_directory_shared(directory_path):
    try:
        os.chmod(directory_path, 0o0777)
    except OSError as e:
        # Silently suppress the PermissionError that is thrown by the chmod.
        # This is done because the user attempting to change the permissions
        # on a directory may not own it. The chmod is attempted whether the
        # directory is new or not to avoid race conditions.
        if e.errno in [errno.EACCES, errno.EPERM]:
            pass
        else:
            raise


def try_to_create_directory(directory_path):
    """Attempt to create a directory that is globally readable/writable.

    Args:
        directory_path: The path of the directory to create.
    """
    directory_path = os.path.expanduser(directory_path)
    os.makedirs(directory_path, exist_ok=True)
    # Change the log directory permissions so others can use it. This is
    # important when multiple people are using the same machine.
    try_make_directory_shared(directory_path)


def try_to_symlink(symlink_path, target_path):
    """Attempt to create a symlink.

    If the symlink path exists and isn't a symlink, the symlink will not be
    created. If a symlink exists in the path, it will be attempted to be
    removed and replaced.

    Args:
        symlink_path: The path at which to create the symlink.
        target_path: The path the symlink should point to.
    """
    symlink_path = os.path.expanduser(symlink_path)
    target_path = os.path.expanduser(target_path)

    if os.path.exists(symlink_path):
        if os.path.islink(symlink_path):
            # Try to remove existing symlink.
            try:
                os.remove(symlink_path)
            except OSError:
                return
        else:
            # There's an existing non-symlink file, don't overwrite it.
            return

    try:
        os.symlink(target_path, symlink_path)
    except OSError:
        return


class Unbuffered(object):
    """There's no "built-in" solution to programatically disabling buffering of
    text files. We expect stdout/err to be text files, so creating an
    unbuffered binary file is unacceptable.

    See
    https://mail.python.org/pipermail/tutor/2003-November/026645.html.
    https://docs.python.org/3/library/functions.html#open

    """

    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


def open_log(path, unbuffered=False, **kwargs):
    """
    Opens the log file at `path`, with the provided kwargs being given to
    `open`.
    """
    # Disable buffering, see test_advanced_3.py::test_logging_to_driver
    kwargs.setdefault("buffering", 1)
    kwargs.setdefault("mode", "a")
    kwargs.setdefault("encoding", "utf-8")
    stream = open(path, **kwargs)
    if unbuffered:
        return Unbuffered(stream)
    else:
        return stream


def detect_fate_sharing_support_win32():
    global win32_job, win32_AssignProcessToJobObject
    if win32_job is None and sys.platform == "win32":
        import ctypes
        try:
            from ctypes.wintypes import BOOL, DWORD, HANDLE, LPVOID, LPCWSTR
            kernel32 = ctypes.WinDLL("kernel32")
            kernel32.CreateJobObjectW.argtypes = (LPVOID, LPCWSTR)
            kernel32.CreateJobObjectW.restype = HANDLE
            sijo_argtypes = (HANDLE, ctypes.c_int, LPVOID, DWORD)
            kernel32.SetInformationJobObject.argtypes = sijo_argtypes
            kernel32.SetInformationJobObject.restype = BOOL
            kernel32.AssignProcessToJobObject.argtypes = (HANDLE, HANDLE)
            kernel32.AssignProcessToJobObject.restype = BOOL
            kernel32.IsDebuggerPresent.argtypes = ()
            kernel32.IsDebuggerPresent.restype = BOOL
        except (AttributeError, TypeError, ImportError):
            kernel32 = None
        job = kernel32.CreateJobObjectW(None, None) if kernel32 else None
        job = subprocess.Handle(job) if job else job
        if job:
            from ctypes.wintypes import DWORD, LARGE_INTEGER, ULARGE_INTEGER

            class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", LARGE_INTEGER),
                    ("PerJobUserTimeLimit", LARGE_INTEGER),
                    ("LimitFlags", DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", DWORD),
                    ("SchedulingClass", DWORD),
                ]

            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("ReadOperationCount", ULARGE_INTEGER),
                    ("WriteOperationCount", ULARGE_INTEGER),
                    ("OtherOperationCount", ULARGE_INTEGER),
                    ("ReadTransferCount", ULARGE_INTEGER),
                    ("WriteTransferCount", ULARGE_INTEGER),
                    ("OtherTransferCount", ULARGE_INTEGER),
                ]

            class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation",
                     JOBOBJECT_BASIC_LIMIT_INFORMATION),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            debug = kernel32.IsDebuggerPresent()

            # Defined in <WinNT.h>; also available here:
            # https://docs.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-setinformationjobobject
            JobObjectExtendedLimitInformation = 9
            JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x00000800
            JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
            buf = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            buf.BasicLimitInformation.LimitFlags = (
                (0 if debug else JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
                | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
                | JOB_OBJECT_LIMIT_BREAKAWAY_OK)
            infoclass = JobObjectExtendedLimitInformation
            if not kernel32.SetInformationJobObject(
                    job, infoclass, ctypes.byref(buf), ctypes.sizeof(buf)):
                job = None
        win32_AssignProcessToJobObject = (kernel32.AssignProcessToJobObject
                                          if kernel32 is not None else False)
        win32_job = job if job else False
    return bool(win32_job)


def detect_fate_sharing_support_linux():
    global linux_prctl
    if linux_prctl is None and sys.platform.startswith("linux"):
        try:
            from ctypes import c_int, c_ulong, CDLL
            prctl = CDLL(None).prctl
            prctl.restype = c_int
            prctl.argtypes = [c_int, c_ulong, c_ulong, c_ulong, c_ulong]
        except (AttributeError, TypeError):
            prctl = None
        linux_prctl = prctl if prctl else False
    return bool(linux_prctl)


def detect_fate_sharing_support():
    result = None
    if sys.platform == "win32":
        result = detect_fate_sharing_support_win32()
    elif sys.platform.startswith("linux"):
        result = detect_fate_sharing_support_linux()
    return result


def set_kill_on_parent_death_linux():
    """Ensures this process dies if its parent dies (fate-sharing).

    Linux-only. Must be called in preexec_fn (i.e. by the child).
    """
    if detect_fate_sharing_support_linux():
        import signal
        PR_SET_PDEATHSIG = 1
        if linux_prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
            import ctypes
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
    else:
        assert False, "PR_SET_PDEATHSIG used despite being unavailable"


def set_kill_child_on_death_win32(child_proc):
    """Ensures the child process dies if this process dies (fate-sharing).

    Windows-only. Must be called by the parent, after spawning the child.

    Args:
        child_proc: The subprocess.Popen or subprocess.Handle object.
    """

    if isinstance(child_proc, subprocess.Popen):
        child_proc = child_proc._handle
    assert isinstance(child_proc, subprocess.Handle)

    if detect_fate_sharing_support_win32():
        if not win32_AssignProcessToJobObject(win32_job, int(child_proc)):
            import ctypes
            raise OSError(ctypes.get_last_error(),
                          "AssignProcessToJobObject() failed")
    else:
        assert False, "AssignProcessToJobObject used despite being unavailable"


def set_sigterm_handler(sigterm_handler):
    """Registers a handler for SIGTERM in a platform-compatible manner."""
    if sys.platform == "win32":
        # Note that these signal handlers only work for console applications.
        # TODO: implement graceful process termination mechanism
        # SIGINT is Ctrl+C, SIGBREAK is Ctrl+Break.
        signal.signal(signal.SIGBREAK, sigterm_handler)
    else:
        signal.signal(signal.SIGTERM, sigterm_handler)


def check_process_exists(pid):
    return psutil.pid_exists(int(pid))


def kill_process_by_pid(pid):
    try:
        os.kill(pid, signal.SIGKILL)
        logger.debug("The process with PID {} has been killed.".format(pid))
    except OSError as e:
        logger.info("There is no process with PID {}".format(pid))


def stop_process_tree(pid, include_parent=True, force=False):
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    children = proc.children(recursive=True)
    if include_parent:
        children.append(proc)
    for p in children:
        try:
            if force:
                p.kill()
            else:
                p.terminate()
        except psutil.NoSuchProcess:  # pragma: no cover
            pass


def get_system_memory(
    # For cgroups v1:
    memory_limit_filename="/sys/fs/cgroup/memory/memory.limit_in_bytes",
    # For cgroups v2:
    memory_limit_filename_v2="/sys/fs/cgroup/memory.max",
):
    """Return the total amount of system memory in bytes.

    Returns:
        The total amount of system memory in bytes.
    """
    # Try to accurately figure out the memory limit if we are in a docker
    # container. Note that this file is not specific to Docker and its value is
    # often much larger than the actual amount of memory.
    docker_limit = None
    if os.path.exists(memory_limit_filename):
        with open(memory_limit_filename, "r") as f:
            docker_limit = int(f.read().strip())
    elif os.path.exists(memory_limit_filename_v2):
        with open(memory_limit_filename_v2, "r") as f:
            # Don't forget to strip() the newline:
            max_file = f.read().strip()
            if max_file.isnumeric():
                docker_limit = int(max_file)
            else:
                # max_file is "max", i.e. is unset.
                docker_limit = None

    # Use psutil if it is available.
    psutil_memory_in_bytes = psutil.virtual_memory().total

    if docker_limit is not None:
        # We take the min because the cgroup limit is very large if we aren't
        # in Docker.
        return min(docker_limit, psutil_memory_in_bytes)

    return psutil_memory_in_bytes


def get_cgroupv1_used_memory(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        cache_bytes = -1
        rss_bytes = -1
        inactive_file_bytes = -1
        working_set = -1
        for line in lines:
            if "total_rss " in line:
                rss_bytes = int(line.split()[1])
            elif "cache " in line:
                cache_bytes = int(line.split()[1])
            elif "inactive_file" in line:
                inactive_file_bytes = int(line.split()[1])
        if cache_bytes >= 0 and rss_bytes >= 0 and inactive_file_bytes >= 0:
            working_set = rss_bytes + cache_bytes - inactive_file_bytes
            assert working_set >= 0
            return working_set
        return None


def get_cgroupv2_used_memory(stat_file, usage_file):
    # Uses same calculation as libcontainer, that is:
    # memory.current - memory.stat[inactive_file]
    # Source: https://github.com/google/cadvisor/blob/24dd1de08a72cfee661f6178454db995900c0fee/container/libcontainer/handler.go#L836  # noqa: E501
    inactive_file_bytes = -1
    current_usage = -1
    with open(usage_file, "r") as f:
        current_usage = int(f.read().strip())
    with open(stat_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "inactive_file" in line:
                inactive_file_bytes = int(line.split()[1])
        if current_usage >= 0 and inactive_file_bytes >= 0:
            working_set = current_usage - inactive_file_bytes
            assert working_set >= 0
            return working_set
        return None


def get_used_memory():
    """Return the currently used system memory in bytes

    Returns:
        The total amount of used memory
    """
    # Try to accurately figure out the memory usage if we are in a docker
    # container.
    docker_usage = None
    # For cgroups v1:
    memory_usage_filename = "/sys/fs/cgroup/memory/memory.stat"
    # For cgroups v2:
    memory_usage_filename_v2 = "/sys/fs/cgroup/memory.current"
    memory_stat_filename_v2 = "/sys/fs/cgroup/memory.stat"
    if os.path.exists(memory_usage_filename):
        docker_usage = get_cgroupv1_used_memory(memory_usage_filename)
    elif os.path.exists(memory_usage_filename_v2) and os.path.exists(
        memory_stat_filename_v2
    ):
        docker_usage = get_cgroupv2_used_memory(
            memory_stat_filename_v2, memory_usage_filename_v2
        )

    if docker_usage is not None:
        return docker_usage
    return psutil.virtual_memory().used


def estimate_available_memory():
    """Return the currently available amount of system memory in bytes.

    Returns:
        The total amount of available memory in bytes. Based on the used
        and total memory.

    """
    return get_system_memory() - get_used_memory()


def get_shared_memory_bytes():
    """Get the size of the shared memory file system.

    Returns:
        The size of the shared memory file system in bytes.
    """
    # Make sure this is only called on Linux.
    assert sys.platform == "linux" or sys.platform == "linux2"

    shm_fd = os.open("/dev/shm", os.O_RDONLY)
    try:
        shm_fs_stats = os.fstatvfs(shm_fd)
        # The value shm_fs_stats.f_bsize is the block size and the
        # value shm_fs_stats.f_bavail is the number of available
        # blocks.
        shm_avail = shm_fs_stats.f_bsize * shm_fs_stats.f_bavail
    finally:
        os.close(shm_fd)

    return shm_avail


def _get_docker_cpus(
    cpu_quota_file_name="/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
    cpu_period_file_name="/sys/fs/cgroup/cpu/cpu.cfs_period_us",
    cpuset_file_name="/sys/fs/cgroup/cpuset/cpuset.cpus",
    cpu_max_file_name="/sys/fs/cgroup/cpu.max",
) -> Optional[float]:
    # Docker has 2 underyling ways of implementing CPU limits:
    # https://docs.docker.com/config/containers/resource_constraints/#configure-the-default-cfs-scheduler
    # 1. --cpuset-cpus 2. --cpus or --cpu-quota/--cpu-period (--cpu-shares is a
    # soft limit so we don't worry about it). For our purposes, if we use
    # docker, the number of vCPUs on a machine is whichever is set (ties broken
    # by smaller value).

    cpu_quota = None
    # See: https://bugs.openjdk.java.net/browse/JDK-8146115
    if os.path.exists(cpu_quota_file_name) and os.path.exists(cpu_period_file_name):
        try:
            with open(cpu_quota_file_name, "r") as quota_file, open(
                cpu_period_file_name, "r"
            ) as period_file:
                cpu_quota = float(quota_file.read()) / float(period_file.read())
        except Exception:
            logger.exception("Unexpected error calculating docker cpu quota.")
    # Look at cpu.max for cgroups v2
    elif os.path.exists(cpu_max_file_name):
        try:
            max_file = open(cpu_max_file_name).read()
            quota_str, period_str = max_file.split()
            if quota_str.isnumeric() and period_str.isnumeric():
                cpu_quota = float(quota_str) / float(period_str)
            else:
                # quota_str is "max" meaning the cpu quota is unset
                cpu_quota = None
        except Exception:
            logger.exception("Unexpected error calculating docker cpu quota.")
    if (cpu_quota is not None) and (cpu_quota < 0):
        cpu_quota = None
    elif cpu_quota == 0:
        # Round up in case the cpu limit is less than 1.
        cpu_quota = 1

    cpuset_num = None
    if os.path.exists(cpuset_file_name):
        try:
            with open(cpuset_file_name) as cpuset_file:
                ranges_as_string = cpuset_file.read()
                ranges = ranges_as_string.split(",")
                cpu_ids = []
                for num_or_range in ranges:
                    if "-" in num_or_range:
                        start, end = num_or_range.split("-")
                        cpu_ids.extend(list(range(int(start), int(end) + 1)))
                    else:
                        cpu_ids.append(int(num_or_range))
                cpuset_num = len(cpu_ids)
        except Exception:
            logger.exception("Unexpected error calculating docker cpuset ids.")
    # Possible to-do: Parse cgroups v2's cpuset.cpus.effective for the number
    # of accessible CPUs.

    if cpu_quota and cpuset_num:
        return min(cpu_quota, cpuset_num)
    return cpu_quota or cpuset_num


def get_num_cpus() -> int:
    """
    Get the number of CPUs available on this node.
    Depending on the situation, use multiprocessing.cpu_count() or cgroups.
    """
    cpu_count = multiprocessing.cpu_count()
    if os.environ.get("CLOUDTIK_USE_MULTIPROCESSING_CPU_COUNT"):
        logger.info(
            "Detected CLOUDTIK_USE_MULTIPROCESSING_CPU_COUNT=1: Using "
            "multiprocessing.cpu_count() to detect the number of CPUs. "
            "This may be inconsistent when used inside docker. "
            "To correctly detect CPUs, unset the env var: "
            "`CLOUDTIK_USE_MULTIPROCESSING_CPU_COUNT`."
        )
        return cpu_count
    try:
        # Not easy to get cpu count in docker, see:
        # https://bugs.python.org/issue36054
        docker_count = _get_docker_cpus()
        if docker_count is not None and docker_count != cpu_count:
            # TODO: We should probably add support for fractional cpus.
            if int(docker_count) != float(docker_count):
                logger.warning(
                    f"CloudTik currently does not support initializing "
                    f"with fractional cpus. Your num_cpus will be "
                    f"truncated from {docker_count} to "
                    f"{int(docker_count)}."
                )
            docker_count = int(docker_count)
            cpu_count = docker_count

    except Exception:
        # `nproc` and cgroup are linux-only. If docker only works on linux
        # (will run in a linux VM on other platforms), so this is fine.
        pass

    return cpu_count


def get_cuda_visible_devices():
    """Get the device IDs in the CUDA_VISIBLE_DEVICES environment variable.

    Returns:
        devices (List[str]): If CUDA_VISIBLE_DEVICES is set, returns a
            list of strings representing the IDs of the visible GPUs.
            If it is not set or is set to NoDevFiles, returns empty list.
    """
    gpu_ids_str = os.environ.get("CUDA_VISIBLE_DEVICES", None)

    if gpu_ids_str is None:
        return None

    if gpu_ids_str == "":
        return []

    if gpu_ids_str == "NoDevFiles":
        return []

    # GPU identifiers are given as strings representing integers or UUIDs.
    return list(gpu_ids_str.split(","))


def decode(byte_str, allow_none=False):
    """Make this unicode in Python 3, otherwise leave it as bytes.

    Args:
        byte_str: The byte string to decode.
        allow_none: If true, then we will allow byte_str to be None in which
            case we will return an empty string. TODO(rkn): Remove this flag.
            This is only here to simplify upgrading to flatbuffers 1.10.0.

    Returns:
        A byte string in Python 2 and a unicode string in Python 3.
    """
    if byte_str is None and allow_none:
        return ""

    if not isinstance(byte_str, bytes):
        raise ValueError(f"The argument {byte_str} must be a bytes object.")
    if sys.version_info >= (3, 0):
        return byte_str.decode("ascii")
    else:
        return byte_str


def ensure_str(s, encoding="utf-8", errors="strict"):
    """Coerce *s* to `str`.

      - `str` -> `str`
      - `bytes` -> decoded to `str`
    """
    if isinstance(s, str):
        return s
    else:
        assert isinstance(s, bytes)
        return s.decode(encoding, errors)


MEMORY_SIZE_UNITS = {
    "K": 2**10,
    "M": 2**20,
    "G": 2**30,
    "T": 2**40,
    "P": 2**50
}


def parse_memory_resource(resource):
    resource_str = str(resource)
    try:
        return int(resource_str)
    except ValueError:
        pass
    memory_size = re.sub(r"([KMGTP]+)", r" \1", resource_str)
    number, unit_index = [item.strip() for item in memory_size.split()]
    unit_index = unit_index[0]
    return float(number) * MEMORY_SIZE_UNITS[unit_index]


def get_memory_in_bytes(memory_size):
    if memory_size is None:
        return 0

    if isinstance(memory_size, int):
        return memory_size

    parsed_value = parse_memory_resource(memory_size)
    return 0 if parsed_value == float("inf") else int(parsed_value)


def format_memory(memory_in_bytes, precision=2):
    if memory_in_bytes >= MEMORY_SIZE_UNITS["P"]:
        memory_in_pb = round(memory_in_bytes / MEMORY_SIZE_UNITS["P"], 2)
        return "{:g}PB".format(memory_in_pb)
    elif memory_in_bytes >= MEMORY_SIZE_UNITS["T"]:
        memory_in_tb = round(memory_in_bytes / MEMORY_SIZE_UNITS["T"], 2)
        return "{:g}TB".format(memory_in_tb)
    elif memory_in_bytes >= MEMORY_SIZE_UNITS["G"]:
        memory_in_gb = round(memory_in_bytes / MEMORY_SIZE_UNITS["G"], 2)
        return "{:g}GB".format(memory_in_gb)
    elif memory_in_bytes >= MEMORY_SIZE_UNITS["M"]:
        memory_in_mb = round(memory_in_bytes / MEMORY_SIZE_UNITS["M"], 2)
        return "{:g}MB".format(memory_in_mb)
    elif memory_in_bytes >= MEMORY_SIZE_UNITS["K"]:
        memory_in_kb = round(memory_in_bytes / MEMORY_SIZE_UNITS["K"], 2)
        return "{:g}KB".format(memory_in_kb)
    else:
        return "{}B".format(memory_in_bytes)


def get_ip_by_name(name):
    return socket.gethostbyname(name)


def get_user_temp_dir():
    if "CLOUDTIK_TMPDIR" in os.environ:
        return os.environ["CLOUDTIK_TMPDIR"]
    elif sys.platform.startswith("linux") and "TMPDIR" in os.environ:
        return os.environ["TMPDIR"]
    elif sys.platform.startswith("darwin") or sys.platform.startswith("linux"):
        # Ideally we wouldn't need this fallback, but keep it for now for
        # for compatibility
        tempdir = os.path.join(os.sep, "tmp")
    else:
        tempdir = tempfile.gettempdir()
    return tempdir


def get_cloudtik_temp_dir():
    return os.path.join(get_user_temp_dir(), "cloudtik")


def exec_with_output(cmd):
    return subprocess.check_output(cmd, shell=True)


def exec_with_call(cmd):
    return subprocess.check_call(cmd, shell=True)


def is_private_ip(ip_addr):
    return ipaddr.IPv4Address(ip_addr).is_private


def get_host_address(address_type="all"):
    addresses = set()
    for iface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET and addr.address != '127.0.0.1':
                if address_type == "private":
                    if is_private_ip(addr.address):
                        addresses.add(addr.address)
                elif address_type == "public":
                    if not is_private_ip(addr.address):
                        addresses.add(addr.address)
                else:
                    addresses.add(addr.address)
    return addresses


def get_free_port(bind_address='127.0.0.1', default_port=6000):
    """ Get free port"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        test_port = default_port
        while True:
            result = s.connect_ex((bind_address, test_port))
            if result != 0:
                return test_port
            else:
                test_port += 1


def generate_public_key(private_key_file, public_key_file: str = None):
    if not public_key_file:
        public_key_file = private_key_file + ".pub"
    if not os.path.exists(public_key_file):
        exec_with_output(
            f"ssh-keygen -y "
            f"-f {private_key_file} "
            f"> {public_key_file} "
            f"&& chmod 600 {public_key_file}"
        )
    return public_key_file


def memory_to_gb(mem_bytes, precision=2):
    return round(mem_bytes / MEMORY_UNIT_GB, precision)


def memory_to_gb_string(mem_bytes, precision=2):
    mem_gb = memory_to_gb(mem_bytes, precision)
    return "{}GB".format(mem_gb)


def strip_quote(value):
    # strip has problem because it will remove all front and tailing quotes
    if not value:
        return value
    if value.startswith('"') or value.startswith("'"):
        value = value[1:]
    if value.endswith('"') or value.endswith("'"):
        value = value[:-1]
    return value


def get_named_log_file_handles(logs_dir, name, redirect_output=None):
    """Open log files with filenames unique by the name, returning the
    file handles. If output redirection has been disabled, no files will
    be opened and `(None, None)` will be returned.

    Args:
        logs_dir (str): The logs dir for storing the log files.
        name (str): descriptive string for this log file.
        redirect_output (bool): Whether to redirect

    Returns:
        A tuple of two file handles for redirecting (stdout, stderr), or
        `(None, None)` if output redirection is disabled.
    """
    if redirect_output is None:
        # Make the default behavior match that of glog.
        redirect_output = os.getenv("GLOG_logtostderr") != "1"

    if not redirect_output:
        return None, None

    log_stdout = os.path.join(logs_dir, f"{name}.out")
    log_stderr = os.path.join(logs_dir, f"{name}.err")
    return open_log(log_stdout), open_log(log_stderr)


def get_json_object_hash(json_object):
    if json_object is None:
        json_data = ""
    else:
        json_data = json.dumps(json_object, sort_keys=True)
    return get_string_hash(json_data)


def get_string_hash(str_data):
    hasher = hashlib.sha1()
    hasher.update(str_data.encode("utf-8"))
    return hasher.hexdigest()


def serialize_config(config):
    json_str = json.dumps(config)
    return base64.b64encode(
        json_str.encode("utf-8")).decode("utf-8")


def deserialize_config(config_str):
    json_str = base64.b64decode(
        config_str.encode('utf-8')).decode("utf-8")
    return json.loads(json_str)
