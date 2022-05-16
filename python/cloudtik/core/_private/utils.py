import base64
import collections
import collections.abc
import copy
from datetime import datetime
import logging
import hashlib
import json
import os
import re
import threading
from typing import Any, Dict, Optional, Tuple, List, Union
import sys
import tempfile
import signal
import subprocess
import errno
import binascii
import uuid
import time
import math
import multiprocessing
import ipaddr
import socket
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor

import yaml

import cloudtik
from cloudtik.core._private import constants, services
from cloudtik.core._private.call_context import CallContext
from cloudtik.core._private.cli_logger import cli_logger
from cloudtik.core._private.cluster.cluster_metrics import ClusterMetricsSummary
from cloudtik.core._private.constants import CLOUDTIK_WHEELS, CLOUDTIK_CLUSTER_PYTHON_VERSION, \
    CLOUDTIK_DEFAULT_MAX_WORKERS, CLOUDTIK_NODE_SSH_INTERVAL_S, CLOUDTIK_NODE_START_WAIT_S, MAX_PARALLEL_EXEC_NODES, \
    CLOUDTIK_CLUSTER_URI_TEMPLATE, CLOUDTIK_RUNTIME_NAME, CLOUDTIK_RUNTIME_ENV_NODE_IP, CLOUDTIK_RUNTIME_ENV_HEAD_IP, \
    CLOUDTIK_RUNTIME_ENV_SECRETS, CLOUDTIK_DEFAULT_PORT, CLOUDTIK_REDIS_DEFAULT_PASSWORD, CLOUDTIK_RUNTIME_ENV_NODE_TYPE
from cloudtik.core._private.crypto import AESCipher
from cloudtik.core._private.runtime_factory import _get_runtime, _get_runtime_cls
from cloudtik.core.node_provider import NodeProvider
from cloudtik.core._private.providers import _get_default_config, _get_node_provider, _get_provider_config_object, \
    _get_node_provider_cls
from cloudtik.core._private.docker import validate_docker_config
from cloudtik.core._private.providers import _get_workspace_provider
from cloudtik.core.tags import CLOUDTIK_TAG_USER_NODE_TYPE, CLOUDTIK_TAG_NODE_STATUS, STATUS_UP_TO_DATE, \
    STATUS_UPDATE_FAILED, CLOUDTIK_TAG_NODE_KIND, NODE_KIND_HEAD

# Import psutil after others so the packaged version is used.
import psutil

REQUIRED, OPTIONAL = True, False
CLOUDTIK_CONFIG_SCHEMA_PATH = os.path.join(
    os.path.dirname(cloudtik.core.__file__), "config-schema.json")
CLOUDTIK_WORKSPACE_SCHEMA_PATH = os.path.join(
    os.path.dirname(cloudtik.core.__file__), "workspace-schema.json")

# Internal kv keys for storing debug status.
CLOUDTIK_CLUSTER_SCALING_ERROR = "__cluster_scaling_error"
CLOUDTIK_CLUSTER_SCALING_STATUS = "__cluster_scaling_status"

# Internal kv key for publish runtime config.
CLOUDTIK_CLUSTER_RUNTIME_CONFIG = "__cluster_runtime_config"
CLOUDTIK_CLUSTER_RUNTIME_CONFIG_NODE_TYPE = "__cluster_runtime_config_{}"
CLOUDTIK_CLUSTER_NODES_INFO_NODE_TYPE = "__cluster_nodes_info_{}"
CLOUDTIK_CLUSTER_VARIABLE = "__cluster_variable_{}"

PLACEMENT_GROUP_RESOURCE_BUNDLED_PATTERN = re.compile(
    r"(.+)_group_(\d+)_([0-9a-zA-Z]+)")
PLACEMENT_GROUP_RESOURCE_PATTERN = re.compile(r"(.+)_group_([0-9a-zA-Z]+)")

ResourceBundle = Dict[str, Union[int, float]]

COMMAND_KEYS = ["initialization_commands",
                "head_initialization_commands",
                "worker_initialization_commands",
                "setup_commands",
                "head_setup_commands",
                "worker_setup_commands",
                "bootstrap_commands",
                "start_commands",
                "head_start_commands",
                "worker_start_commands",
                "stop_commands",
                "head_stop_commands",
                "worker_stop_commands"]

NODE_TYPE_COMMAND_KEYS = ["worker_initialization_commands",
                          "worker_setup_commands",
                          "bootstrap_commands",
                          "worker_start_commands",
                          "worker_stop_commands"]

DOCKER_COMMAND_KEYS = [
                "initialization_commands",
                "head_initialization_commands",
                "worker_initialization_commands"]

TEMPORARY_COMMAND_KEYS = [
                "initialization_commands",
                "setup_commands",
                "start_commands",
                "bootstrap_commands",
                "stop_commands"]

MERGED_COMMAND_KEY = "merged_commands"
RUNTIME_CONFIG_KEY = "runtime"
DOCKER_CONFIG_KEY = "docker"
RUNTIME_TYPES_CONFIG_KEY = "types"

pwd = None
if sys.platform != "win32":
    pass

logger = logging.getLogger(__name__)

# Linux can bind child processes' lifetimes to that of their parents via prctl.
# prctl support is detected dynamically once, and assumed thereafter.
linux_prctl = None

# Windows can bind processes' lifetimes to that of kernel-level "job objects".
# We keep a global job object to tie its lifetime to that of our own process.
win32_job = None
win32_AssignProcessToJobObject = None


def round_memory_size_to_gb(memory_size: int) -> int:
    gb = int(memory_size / 1024)
    if gb < 1:
        gb = 1
    return gb * 1024


def run_system_command(cmd: str):
    result = os.system(cmd)
    if result != 0:
        raise RuntimeError(f"Error happened in running: {cmd}")


def run_bash_scripts(command: str, script_path, script_args):
    cmds = [
        "bash",
        script_path,
    ]

    cmds += [command]
    if script_args:
        cmds += list(script_args)
    final_cmd = " ".join(cmds)

    run_system_command(final_cmd)


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


def binary_to_hex(identifier):
    hex_identifier = binascii.hexlify(identifier)
    if sys.version_info >= (3, 0):
        hex_identifier = hex_identifier.decode()
    return hex_identifier


def hex_to_binary(hex_identifier):
    return binascii.unhexlify(hex_identifier)


def _random_string():
    id_hash = hashlib.shake_128()
    id_hash.update(uuid.uuid4().bytes)
    id_bytes = id_hash.digest(constants.ID_SIZE)
    assert len(id_bytes) == constants.ID_SIZE
    return id_bytes


def format_error_message(exception_message, task_exception=False):
    """Improve the formatting of an exception thrown by a remote function.

    This method takes a traceback from an exception and makes it nicer by
    removing a few uninformative lines and adding some space to indent the
    remaining lines nicely.

    Args:
        exception_message (str): A message generated by traceback.format_exc().

    Returns:
        A string of the formatted exception message.
    """
    lines = exception_message.split("\n")
    if task_exception:
        # For errors that occur inside of tasks, remove lines 1 and 2 which are
        # always the same, they just contain information about the worker code.
        lines = lines[0:1] + lines[3:]
        pass
    return "\n".join(lines)


def publish_error(error_type,
                            message,
                            redis_client=None):
    """Push an error message to Redis.

    Args:
        error_type (str): The type of the error.
        message (str): The message that will be printed in the background
            on the driver.
        redis_client: The redis client to use.
    """
    # TODO (haifeng) : improve to the right format, current we simply use the string
    if redis_client:
        message = (f"ERROR: {time.time()}: {error_type}: \n"
                   f"{message}")
        redis_client.publish("ERROR_INFO",
                             message)
    else:
        raise ValueError(
            "redis_client needs to be specified!")


def get_system_memory():
    """Return the total amount of system memory in bytes.

    Returns:
        The total amount of system memory in bytes.
    """
    # Try to accurately figure out the memory limit if we are in a docker
    # container. Note that this file is not specific to Docker and its value is
    # often much larger than the actual amount of memory.
    docker_limit = None
    memory_limit_filename = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    if os.path.exists(memory_limit_filename):
        with open(memory_limit_filename, "r") as f:
            docker_limit = int(f.read())

    # Use psutil if it is available.
    psutil_memory_in_bytes = psutil.virtual_memory().total

    if docker_limit is not None:
        # We take the min because the cgroup limit is very large if we aren't
        # in Docker.
        return min(docker_limit, psutil_memory_in_bytes)

    return psutil_memory_in_bytes


def get_used_memory():
    """Return the currently used system memory in bytes

    Returns:
        The total amount of used memory
    """
    # Try to accurately figure out the memory usage if we are in a docker
    # container.
    docker_usage = None
    memory_usage_filename = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
    if os.path.exists(memory_usage_filename):
        with open(memory_usage_filename, "r") as f:
            docker_usage = int(f.read())

    # Use psutil if it is available.
    psutil_memory_in_bytes = psutil.virtual_memory().used

    if docker_usage is not None:
        # We take the min because the cgroup limit is very large if we aren't
        # in Docker.
        return min(docker_usage, psutil_memory_in_bytes)

    return psutil_memory_in_bytes


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
        cpuset_file_name="/sys/fs/cgroup/cpuset/cpuset.cpus"
) -> Optional[float]:
    # TODO: Don't implement this logic oursleves.
    # Docker has 2 underyling ways of implementing CPU limits:
    # https://docs.docker.com/config/containers/resource_constraints/#configure-the-default-cfs-scheduler
    # 1. --cpuset-cpus 2. --cpus or --cpu-quota/--cpu-period (--cpu-shares is a
    # soft limit so we don't worry about it). For our purposes, if we use
    # docker, the number of vCPUs on a machine is whichever is set (ties broken
    # by smaller value).

    cpu_quota = None
    # See: https://bugs.openjdk.java.net/browse/JDK-8146115
    if os.path.exists(cpu_quota_file_name) and os.path.exists(
            cpu_quota_file_name):
        try:
            with open(cpu_quota_file_name, "r") as quota_file, open(
                    cpu_period_file_name, "r") as period_file:
                cpu_quota = float(quota_file.read()) / float(
                    period_file.read())
        except Exception as e:
            logger.exception("Unexpected error calculating docker cpu quota.",
                             e)
    if (cpu_quota is not None) and (cpu_quota < 0):
        cpu_quota = None

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
        except Exception as e:
            logger.exception("Unexpected error calculating docker cpuset ids.",
                             e)

    if cpu_quota and cpuset_num:
        return min(cpu_quota, cpuset_num)
    else:
        return cpu_quota or cpuset_num


def get_k8s_cpus(cpu_share_file_name="/sys/fs/cgroup/cpu/cpu.shares") -> float:
    """Get number of CPUs available for use by this container, in terms of
    cgroup cpu shares.

    This is the number of CPUs K8s has assigned to the container based
    on pod spec requests and limits.

    Note: using cpu_quota as in _get_docker_cpus() works
    only if the user set CPU limit in their pod spec (in addition to CPU
    request). Otherwise, the quota is unset.
    """
    try:
        cpu_shares = int(open(cpu_share_file_name).read())
        container_num_cpus = cpu_shares / 1024
        return container_num_cpus
    except Exception as e:
        logger.exception("Error computing CPU limit of Kubernetes pod.", e)
        return 1.0


def get_num_cpus() -> int:
    if "KUBERNETES_SERVICE_HOST" in os.environ:
        # If in a K8S pod, use cgroup cpu shares and round up.
        return int(math.ceil(get_k8s_cpus()))
    cpu_count = multiprocessing.cpu_count()
    if os.environ.get("CLOUDTIK_USE_MULTIPROCESSING_CPU_COUNT"):
        logger.info(
            "Detected CLOUDTIK_USE_MULTIPROCESSING_CPU_COUNT=1: Using "
            "multiprocessing.cpu_count() to detect the number of CPUs. "
            "This may be inconsistent when used inside docker. "
            "To correctly detect CPUs, unset the env var: "
            "`CLOUDTIK_USE_MULTIPROCESSING_CPU_COUNT`.")
        return cpu_count
    try:
        # Not easy to get cpu count in docker, see:
        # https://bugs.python.org/issue36054
        docker_count = _get_docker_cpus()
        if docker_count is not None and docker_count != cpu_count:
            # TODO: We should probably add support for fractional cpus.
            if int(docker_count) != float(docker_count):
                logger.warning(
                    f"We currently does not support initializing "
                    f"with fractional cpus. Your num_cpus will be "
                    f"truncated from {docker_count} to "
                    f"{int(docker_count)}.")
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


def run_in_paralell_on_nodes(run_exec,
                             call_context: CallContext,
                             nodes):
    # This is to ensure that the parallel SSH calls below do not mess with
    # the users terminal.
    output_redir = call_context.is_output_redirected()
    call_context.set_output_redirected(True)
    allow_interactive = call_context.does_allow_interactive()
    call_context.set_allow_interactive(False)

    with ThreadPoolExecutor(
            max_workers=MAX_PARALLEL_EXEC_NODES) as executor:
        for node_id in nodes:
            executor.submit(
                run_exec, node_id=node_id, call_context=call_context.new_call_context())
    call_context.set_output_redirected(output_redir)
    call_context.set_allow_interactive(allow_interactive)


def validate_config(config: Dict[str, Any]) -> None:
    """Required Dicts indicate that no extra fields can be introduced."""
    if not isinstance(config, dict):
        raise ValueError("Config {} is not a dictionary".format(config))

    with open(CLOUDTIK_CONFIG_SCHEMA_PATH) as f:
        schema = json.load(f)

    try:
        import jsonschema
    except (ModuleNotFoundError, ImportError) as e:
        # Don't log a warning message here. Logging be handled by upstream.
        raise e from None

    try:
        jsonschema.validate(config, schema)
    except jsonschema.ValidationError as e:
        # The validate method show very long message of the schema
        # and the instance data, we need show this only at verbose mode
        if cli_logger.verbosity > 0:
            raise e from None
        else:
            # For none verbose mode, show short message
            raise RuntimeError("JSON schema validation error: {}.".format(e.message)) from None

    # Detect out of date defaults. This happens when the cluster scaler that filled
    # out the default values is older than the version of the cluster scaler that
    # is running on the cluster.
    if "cluster_synced_files" not in config:
        raise RuntimeError(
            "Missing 'cluster_synced_files' field in the cluster "
            "configuration. ")

    if "available_node_types" in config:
        if "head_node_type" not in config:
            raise ValueError(
                "You must specify `head_node_type` if `available_node_types "
                "is set.")
        if config["head_node_type"] not in config["available_node_types"]:
            raise ValueError(
                "`head_node_type` must be one of `available_node_types`.")

        sum_min_workers = sum(
            config["available_node_types"][node_type].get("min_workers", 0)
            for node_type in config["available_node_types"])
        if sum_min_workers > config["max_workers"]:
            raise ValueError(
                "The specified global `max_workers` is smaller than the "
                "sum of `min_workers` of all the available node types.")

    provider = _get_node_provider(config["provider"], config["cluster_name"])
    provider.validate_config(config["provider"])

    # add runtime config validate and testing
    runtime_validate_config(config.get(RUNTIME_CONFIG_KEY), config, provider)


def verify_config(config: Dict[str, Any]):
    """Verify the configurations. Usually verify may mean to involve slow process"""
    provider_config = config["provider"]
    provider = _get_node_provider(provider_config, config["cluster_name"])

    provider.verify_config(provider_config)

    # add runtime config validate and testing
    runtime_verify_config(config.get(RUNTIME_CONFIG_KEY), config, provider)


def prepare_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    The returned config has the following properties:
    - Uses the multi-node-type cluster scaler configuration.
    - Merged with the appropriate defaults.yaml
    - Has a valid Docker configuration if provided.
    - Has max_worker set for each node type.
    """
    provider_cls = _get_node_provider_cls(config["provider"])
    config = provider_cls.prepare_config(config)

    with_defaults = fillout_defaults(config)
    merge_cluster_config(with_defaults)
    validate_docker_config(with_defaults)
    fill_node_type_min_max_workers(with_defaults)
    return with_defaults


def prepare_workspace_config(config: Dict[str, Any]) -> Dict[str, Any]:
    with_defaults = fillout_workspace_defaults(config)
    return with_defaults


def fillout_workspace_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    # Merge the config with user inheritance hierarchy and system defaults hierarchy
    merged_config = merge_config_hierarchy(
        config["provider"], config, False, "workspace-defaults")
    return merged_config


def _get_user_template_file(template_name: str):
    if constants.CLOUDTIK_USER_TEMPLATES in os.environ:
        user_template_dir = os.environ[constants.CLOUDTIK_USER_TEMPLATES]
        if user_template_dir:
            template_file = os.path.join(user_template_dir, template_name)
            if os.path.exists(template_file):
                return template_file

    return None


def _get_template_config(template_name: str, system: bool = False) -> Dict[str, Any]:
    """Load the template config"""
    import cloudtik as cloudtik_home

    # Append .yaml extension if the name doesn't include
    if not template_name.endswith(".yaml"):
        template_name += ".yaml"

    if system:
        # System templates
        template_file = os.path.join(
            os.path.dirname(cloudtik_home.__file__), "providers", template_name)
    else:
        # Check user templates
        template_file = _get_user_template_file(template_name)
        if not template_file:
            # Check built templates
            template_file = os.path.join(
                os.path.dirname(cloudtik_home.__file__), "templates", template_name)

    with open(template_file) as f:
        template_config = yaml.safe_load(f)

    return template_config


def merge_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    return update_nested_dict(config, updates)


def get_merged_base_config(provider, base_config_name: str,
                           system: bool = False, object_name: str = None) -> Dict[str, Any]:
    template_config = _get_template_config(base_config_name, system=system)

    # if provider config exists, verify the provider.type are the same
    template_provider_type = template_config.get("provider", {}).get("type", None)
    if template_provider_type and template_provider_type != provider["type"]:
        raise RuntimeError("Template provider type ({}) doesn't match ({})!".format(
            template_provider_type, provider["type"]))

    merged_config = merge_config_hierarchy(provider, template_config,
                                           system=system, object_name=object_name)
    return merged_config


def get_merged_default_config(provider, object_name: str = None) -> Dict[str, Any]:
    if object_name is None:
        config_object = _get_default_config(provider)
    else:
        config_object = _get_provider_config_object(provider, object_name)
    return merge_config_hierarchy(provider, config_object, system=True, object_name=object_name)


def merge_config_hierarchy(provider, config: Dict[str, Any],
                           system: bool = False, object_name: str = None) -> Dict[str, Any]:
    base_config_name = config.get("from", None)
    if base_config_name:
        # base config is provided, we need to merge with base configuration
        merged_base_config = get_merged_base_config(provider, base_config_name,
                                                    system, object_name)
        merged_config = merge_config(merged_base_config, config)
    elif system:
        merged_config = config
    else:
        # no base, use the system defaults for specific provider as base
        merged_defaults = get_merged_default_config(provider, object_name)
        merged_defaults = merge_config(merged_defaults, config)
        merged_config = copy.deepcopy(merged_defaults)

    return merged_config


def _get_rooted_template_config(root: str, template_name: str) -> Dict[str, Any]:
    """Load the template config from root"""
    # Append .yaml extension if the name doesn't include
    if not template_name.endswith(".yaml"):
        template_name += ".yaml"

    template_file = os.path.join(root, template_name)
    with open(template_file) as f:
        template_config = yaml.safe_load(f)

    return template_config


def get_rooted_merged_base_config(root: str, base_config_name: str,
                                  object_name: str = None) -> Dict[str, Any]:
    template_config = _get_rooted_template_config(root, base_config_name)
    merged_config = merge_rooted_config_hierarchy(
        root, template_config, object_name=object_name)
    return merged_config


def merge_rooted_config_hierarchy(root: str, config: Dict[str, Any],
                                  object_name: str = None) -> Dict[str, Any]:
    base_config_name = config.get("from", None)
    if base_config_name:
        # base config is provided, we need to merge with base configuration
        merged_base_config = get_rooted_merged_base_config(
            root, base_config_name, object_name)
        merged_config = merge_config(merged_base_config, config)
    else:
        merged_config = config

    return merged_config


def fillout_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    # Merge the config with user inheritance hierarchy and system defaults hierarchy
    merged_config = merge_config_hierarchy(config["provider"], config)

    # Fill auth field to avoid key errors.
    # This field is accessed when calling NodeUpdater but is not relevant to
    # certain node providers and is thus left out of some cluster launching
    # configs.
    merged_config["auth"] = merged_config.get("auth", {})

    # Take care of this here, in case a config does not specify any of head,
    # workers, node types, but does specify min workers:
    merged_config.pop("min_workers", None)

    return merged_config


def prepare_internal_commands(config, built_in_commands):
    setup_commands = built_in_commands.get("setup_commands", [])
    cloudtik_setup_command = get_cloudtik_setup_command(config)
    setup_commands += [cloudtik_setup_command]
    built_in_commands["setup_commands"] = setup_commands


def merge_command_key(merged_commands, group_name, from_config, command_key):
    if command_key not in merged_commands:
        merged_commands[command_key] = []

    commands = from_config.get(command_key, [])
    # Commands for this group, don't add the group
    if len(commands) == 0:
        return

    # Append a command group to the command key groups
    command_groups = merged_commands[command_key]
    command_group = {"group_name": group_name, "commands": commands}
    command_groups += [command_group]


def merge_runtime_commands(config, commands_root):
    runtime_config = commands_root.get(RUNTIME_CONFIG_KEY)
    if runtime_config is None:
        return

    runtime_types = runtime_config.get(RUNTIME_TYPES_CONFIG_KEY, [])
    for runtime_type in runtime_types:
        runtime = _get_runtime(runtime_type, runtime_config)
        runtime_commands = runtime.get_runtime_commands(config)
        if runtime_commands:
            merge_commands_from(commands_root, runtime_type, runtime_commands)


def merge_built_in_commands(config, commands_root, built_in_commands):
    # Merge runtime commands with built-in: runtime after built-in
    merge_commands_from(commands_root, CLOUDTIK_RUNTIME_NAME, built_in_commands)
    merge_runtime_commands(config, commands_root)


def merge_commands_from(config, group_name, from_config):
    if MERGED_COMMAND_KEY not in config:
        config[MERGED_COMMAND_KEY] = {}
    merged_commands = config[MERGED_COMMAND_KEY]

    for command_key in COMMAND_KEYS:
        merge_command_key(merged_commands, group_name, from_config, command_key)

    merge_docker_initialization_commands(merged_commands, group_name, from_config)


def merge_docker_initialization_commands(merged_commands, group_name, from_config):
    if DOCKER_CONFIG_KEY not in merged_commands:
        merged_commands[DOCKER_CONFIG_KEY] = {}

    docker = merged_commands[DOCKER_CONFIG_KEY]
    from_docker = from_config.get(DOCKER_CONFIG_KEY, {})

    for command_key in DOCKER_COMMAND_KEYS:
      merge_command_key(docker, group_name, from_docker, command_key)


def merge_cluster_config(config):
    reorder_runtimes_for_dependency(config)
    merge_commands(config)
    merge_runtime_config(config)


def make_sure_dependency_order(
        reordered_runtimes: List[str],
        runtime: str, dependency: str):
    try:
        dependency_index = reordered_runtimes.index(dependency)
    except ValueError:
        # dependency not found
        return

    runtime_index = reordered_runtimes.index(runtime)
    if dependency_index > runtime_index:
        # Needs to move dependency ahead
        reordered_runtimes.pop(dependency_index)
        reordered_runtimes.insert(runtime_index, dependency)


def reorder_runtimes_for_dependency(config):
    _reorder_runtimes_for_dependency(config)
    _reorder_runtimes_for_dependency_of_node_types(config)


def _reorder_runtimes_for_dependency_of_node_types(config):
    global_runtime_types = get_enabled_runtimes(config)
    global_runtime_types_set = set(global_runtime_types)

    # Check and reorder node type specific runtime types
    node_types = config["available_node_types"]
    for node_type in node_types:
        node_type_config = node_types[node_type]
        if RUNTIME_CONFIG_KEY not in node_type_config:
            continue
        node_type_runtime_config = node_type_config[RUNTIME_CONFIG_KEY]
        if RUNTIME_TYPES_CONFIG_KEY not in node_type_runtime_config:
            continue

        node_type_runtime_types = node_type_runtime_config.get(RUNTIME_TYPES_CONFIG_KEY, [])
        # Make sure it is a subset of the global runtime types
        node_type_runtime_types_set = set(node_type_runtime_types)

        if not node_type_runtime_types_set.issubset(global_runtime_types_set):
            raise ValueError(
                "Node type {} runtime types {} is not a subset of global runtime types {}.".format(
                    node_type, node_type_runtime_types, global_runtime_types))

        _reorder_runtimes_for_dependency(node_type_config)


def _reorder_runtimes_for_dependency(config):
    runtime_config = config.get(RUNTIME_CONFIG_KEY)
    if runtime_config is None:
        return

    runtime_types = runtime_config.get(RUNTIME_TYPES_CONFIG_KEY, [])
    if len(runtime_types) == 0:
        return

    reordered_runtimes = copy.deepcopy(runtime_types)
    for runtime_type in runtime_types:
        runtime_cls = _get_runtime_cls(runtime_type)
        dependencies = runtime_cls.get_dependencies()
        if dependencies is None:
            continue

        # For each dependency, if it is appeared behind the current runtime, move it ahead
        for dependency in dependencies:
            make_sure_dependency_order(
                reordered_runtimes, runtime=runtime_type, dependency=dependency)

    runtime_config[RUNTIME_TYPES_CONFIG_KEY] = reordered_runtimes


def merge_runtime_config(config):
    runtime_config = config.get(RUNTIME_CONFIG_KEY)
    if runtime_config is None:
        return

    merge_global_runtime_config(config)

    # Handle node type specific runtime config defaults (based on roles)
    merge_runtime_config_for_node_types(config)


def merge_global_runtime_config(config):
    runtime_config = config[RUNTIME_CONFIG_KEY]
    runtime_types = runtime_config.get(RUNTIME_TYPES_CONFIG_KEY, [])
    for runtime_type in runtime_types:
        runtime = _get_runtime(runtime_type, runtime_config)
        defaults_config = runtime.get_defaults_config(config)
        if defaults_config is None:
            continue

        if runtime_type not in runtime_config:
            runtime_config[runtime_type] = {}
        user_config = runtime_config[runtime_type]
        merged_config = merge_config(defaults_config, user_config)
        runtime_config[runtime_type] = merged_config


def merge_runtime_config_for_node_types(config):
    node_types = config["available_node_types"]
    for node_type in node_types:
        node_type_config = node_types[node_type]
        if RUNTIME_CONFIG_KEY not in node_type_config:
            continue

        runtime_config = node_type_config[RUNTIME_CONFIG_KEY]
        merge_runtime_config_for_node_type(config, runtime_config)


def merge_runtime_config_for_node_type(config, runtime_config):
    global_runtime_config = copy.deepcopy(config[RUNTIME_CONFIG_KEY])
    runtime_config_merged = merge_config(global_runtime_config, runtime_config)

    runtime_types = runtime_config_merged.get(RUNTIME_TYPES_CONFIG_KEY, [])
    for runtime_type in runtime_types:
        if runtime_type not in runtime_config:
            continue

        runtime = _get_runtime(runtime_type, runtime_config_merged)
        defaults_config = runtime.get_defaults_config(config)
        if defaults_config is None:
            continue

        user_config = runtime_config[runtime_type]
        merged_config = merge_config(defaults_config, user_config)
        runtime_config[runtime_type] = merged_config


def merge_user_commands(config):
    merge_commands_from(config, "user", config)


def merge_commands(config):
    # Load the built-in commands and merge with defaults
    built_in_commands = merge_config_hierarchy(config["provider"], {},
                                               False, "commands")
    # Populate some internal command which is generated on the fly
    prepare_internal_commands(config, built_in_commands)

    merge_global_commands(config, built_in_commands=built_in_commands)

    # Merge commands for node types if needed
    merge_commands_for_node_types(config, built_in_commands=built_in_commands)


def merge_global_commands(config, built_in_commands):
    merge_commands_for(config, config, built_in_commands=built_in_commands)


def merge_commands_for_node_types(config, built_in_commands):
    node_types = config["available_node_types"]
    head_node_type = config["head_node_type"]
    for node_type in node_types:
        # No need for head
        if node_type == head_node_type:
            continue

        node_type_config = node_types[node_type]
        if not is_commands_merge_needed(config, node_type_config):
            continue

        # Special care needs to take here
        # For a command that if we don't have an override, we need copy it from global
        # If we don't specify the runtime types, we need copy it from global
        new_node_type_config = copy.deepcopy(node_type_config)
        inherit_commands_from(new_node_type_config, config)

        # inherit runtime config
        inherit_runtime_config_from(new_node_type_config, config)

        merge_commands_for(config, new_node_type_config,
                           built_in_commands=built_in_commands)

        node_type_config[MERGED_COMMAND_KEY] = new_node_type_config[MERGED_COMMAND_KEY]


def inherit_commands_from(target_config, from_config):
    for command_key in COMMAND_KEYS:
        inherit_key_from(target_config, from_config, command_key)


def inherit_runtime_config_from(target_config, from_config):
    if RUNTIME_CONFIG_KEY in from_config:
        if RUNTIME_CONFIG_KEY in target_config:
            runtime_config = target_config[RUNTIME_CONFIG_KEY]
        else:
            runtime_config = {}
        base_runtime_config = copy.deepcopy(from_config[RUNTIME_CONFIG_KEY])
        target_config[RUNTIME_CONFIG_KEY] = merge_config(
            base_runtime_config, runtime_config)


def inherit_key_from(target_config, from_config, config_key):
    if config_key not in target_config:
        if config_key in from_config:
            target_config[config_key] = from_config[config_key]


def is_commands_merge_needed(config, node_type_config) -> bool:
    # Check whether commands merge is needed
    # 1. If any of these commands override, we need merge
    for command_key in NODE_TYPE_COMMAND_KEYS:
        if command_key in node_type_config:
            return True

    # 2. If there is node type specific runtime config
    # Which may means to generate a different runtime commands for this node type
    if RUNTIME_CONFIG_KEY in node_type_config:
        return True

    return False


def merge_commands_for(config, commands_root, built_in_commands):
    merge_built_in_commands(config, commands_root,
                            built_in_commands=built_in_commands)

    # Merge user commands after built-in
    merge_user_commands(commands_root)

    # Combine commands
    merged_commands = commands_root[MERGED_COMMAND_KEY]
    combine_initialization_commands_from_docker(config, merged_commands)

    combine_initialization_commands(merged_commands)
    combine_setup_commands(merged_commands)
    combine_start_commands(merged_commands)
    combine_stop_commands(merged_commands)

    clean_temporary_commands(merged_commands)
    return commands_root


def clean_temporary_commands(merged_commands):
    for command_key in TEMPORARY_COMMAND_KEYS:
        if command_key in merged_commands:
            merged_commands.pop(command_key, None)

    # docker / initialization commands
    if DOCKER_CONFIG_KEY in merged_commands:
        merged_commands.pop(DOCKER_CONFIG_KEY, None)


def combine_initialization_commands_from_docker(config, merged_commands):
    for command_key in DOCKER_COMMAND_KEYS:
        combine_commands_from_docker(config, merged_commands, command_key)


def combine_commands_from_docker(config, merged_commands, command_key):
    # Check if docker enabled
    commands = merged_commands[command_key]
    if is_docker_enabled(config):
        docker_commands = merged_commands.get(DOCKER_CONFIG_KEY, {}).get(command_key)
        if docker_commands:
            commands += docker_commands

    merged_commands[command_key] = commands


def _get_node_specific_runtime_types(config, node_id: str):
    provider = _get_node_provider(config["provider"], config["cluster_name"])
    node_type_config = _get_node_specific_config(config, provider, node_id)
    if (node_type_config is not None) and (
            RUNTIME_CONFIG_KEY in node_type_config) and (
            RUNTIME_TYPES_CONFIG_KEY in node_type_config[RUNTIME_CONFIG_KEY]):
        return node_type_config[RUNTIME_CONFIG_KEY][RUNTIME_TYPES_CONFIG_KEY]

    return get_enabled_runtimes(config)


def _get_node_specific_commands(config, provider, node_id: str,
                                command_key: str) -> Any:
    node_type = get_node_type(provider, node_id)
    return _get_node_type_specific_commands(config, node_type, command_key)


def _get_node_type_config(config, node_type: str) -> Any:
    if node_type is None:
        return None

    available_node_types = config["available_node_types"]
    if node_type not in available_node_types:
        raise ValueError(f"Unknown node type: {node_type}.")
    return available_node_types[node_type]


def _get_node_type_specific_fields(config, node_type: str,
                                   fields_key: str) -> Any:
    fields = None
    node_specific_config = _get_node_type_config(config, node_type)
    if (node_specific_config is not None) and (
            fields_key in node_specific_config):
        fields = node_specific_config[fields_key]

    return fields


def _has_node_type_specific_commands(config, node_type: str):
    return _has_node_type_specific_config(config, node_type, MERGED_COMMAND_KEY)


def _get_node_type_specific_commands(config, node_type: str,
                                     command_key: str) -> Any:
    commands = get_commands_to_run(config, command_key)
    node_type_config = _get_node_type_config(config, node_type)
    if (node_type_config is not None) and (
            MERGED_COMMAND_KEY in node_type_config):
        commands = get_commands_to_run(node_type_config, command_key)
    return commands


def _get_node_specific_config(config, provider, node_id: str) -> Any:
    node_tags = provider.node_tags(node_id)
    if CLOUDTIK_TAG_USER_NODE_TYPE in node_tags:
        node_type = node_tags[CLOUDTIK_TAG_USER_NODE_TYPE]
        available_node_types = config["available_node_types"]
        if node_type not in available_node_types:
            raise ValueError(f"Unknown node type tag: {node_type}.")
        return available_node_types[node_type]

    return None


def _get_node_specific_fields(config, provider, node_id: str,
                              fields_key: str) -> Any:
    fields = None
    node_specific_config = _get_node_specific_config(config, provider, node_id)
    if (node_specific_config is not None) and (
            fields_key in node_specific_config):
        fields = node_specific_config[fields_key]

    return fields


def _get_node_specific_docker_config(config, provider, node_id):
    if DOCKER_CONFIG_KEY not in config:
        return {}
    docker_config = config[DOCKER_CONFIG_KEY]
    node_specific_docker = _get_node_specific_fields(
        config, provider, node_id, DOCKER_CONFIG_KEY)
    if node_specific_docker is not None:
        docker_config = copy.deepcopy(docker_config)
        return merge_config(docker_config, node_specific_docker)
    return docker_config


def _get_node_specific_runtime_config(config, provider, node_id):
    node_type = get_node_type(provider, node_id)
    return _get_node_type_specific_runtime_config(config, node_type)


def _has_node_type_specific_config(config, node_type: str, config_key: str):
    node_type_config = _get_node_type_config(config, node_type)
    if (node_type_config is not None) and (
            config_key in node_type_config):
        return True
    return False


def _has_node_type_specific_runtime_config(config, node_type: str):
    return _has_node_type_specific_config(config, node_type, RUNTIME_CONFIG_KEY)


def _get_node_type_specific_runtime_config(config, node_type: str):
    if RUNTIME_CONFIG_KEY not in config:
        return None
    runtime_config = config[RUNTIME_CONFIG_KEY]
    node_specific_runtime = _get_node_type_specific_fields(
        config, node_type, RUNTIME_CONFIG_KEY)
    if node_specific_runtime is not None:
        runtime_config = copy.deepcopy(runtime_config)
        return merge_config(runtime_config, node_specific_runtime)
    return runtime_config


def get_commands_to_run(config, commands_key):
    merged_commands = config[MERGED_COMMAND_KEY]
    return merged_commands.get(commands_key, [])


def get_node_specific_commands_of_runtimes(
        config, provider, node_id: str,
        command_key: str, runtimes: Optional[List[str]] = None) -> Any:
    commands_to_run = _get_node_specific_commands(
        config, provider, node_id=node_id, command_key=command_key)
    return filter_commands_of_runtimes(commands_to_run, runtimes)


def get_commands_of_runtimes(config, command_key,
                             runtimes: Optional[List[str]] = None):
    commands_to_run = get_commands_to_run(config, command_key)
    return filter_commands_of_runtimes(commands_to_run, runtimes)


def filter_commands_of_runtimes(commands_to_run, runtimes: Optional[List[str]] = None):
    if runtimes is None:
        return commands_to_run

    runtime_commands = []
    for command_group in commands_to_run:
        group_name = command_group.get("group_name", "")
        if group_name in runtimes:
            runtime_commands += [command_group]

    return runtime_commands


def get_default_cloudtik_wheel_url() -> str:
    wheel_url = CLOUDTIK_WHEELS
    wheel_url += "/cloudtik-"
    wheel_url += cloudtik.__version__

    if CLOUDTIK_CLUSTER_PYTHON_VERSION == "3.8":
        wheel_url += "-cp38-cp38"
    elif CLOUDTIK_CLUSTER_PYTHON_VERSION == "3.9":
        wheel_url += "-cp39-cp39"
    else:
        # Default python 3.7
        wheel_url += "-cp37-cp37m"

    wheel_url += "-manylinux2014_x86_64.whl"
    return wheel_url


def get_cloudtik_setup_command(config) -> str:
    provider_type = config["provider"]["type"]
    setup_command = "which cloudtik || pip -qq install -U \"cloudtik["
    setup_command += provider_type
    setup_command += "] @ "
    setup_command += config.get("cloudtik_wheel_url", get_default_cloudtik_wheel_url())
    setup_command += "\""
    return setup_command


def combine_initialization_commands(config):
    initialization_commands = config["initialization_commands"]
    config["head_initialization_commands"] = (initialization_commands + config["head_initialization_commands"])
    config["worker_initialization_commands"] = (initialization_commands + config["worker_initialization_commands"])


def combine_setup_commands(config):
    setup_commands = config["setup_commands"]
    bootstrap_commands = config["bootstrap_commands"]

    config["head_setup_commands"] = (
            setup_commands + config["head_setup_commands"] + bootstrap_commands)
    config["worker_setup_commands"] = (
            setup_commands + config["worker_setup_commands"] + bootstrap_commands)


def combine_start_commands(config):
    start_commands = config["start_commands"]
    config["head_start_commands"] = (start_commands + config["head_start_commands"])
    config["worker_start_commands"] = (start_commands + config["worker_start_commands"])


def combine_stop_commands(config):
    stop_commands = config["stop_commands"]
    config["head_stop_commands"] = (stop_commands + config["head_stop_commands"])
    config["worker_stop_commands"] = (stop_commands + config["worker_stop_commands"])


def fill_default_max_workers(config):
    if "max_workers" not in config:
        logger.debug("Global max workers not set. "
                     "Will set to the sum of min workers or {} which is larger.", CLOUDTIK_DEFAULT_MAX_WORKERS)
        sum_min_workers = 0
        node_types = config["available_node_types"]
        for node_type_name in node_types:
            node_type_data = node_types[node_type_name]
            sum_min_workers += node_type_data.get("min_workers", 0)
        config["max_workers"] = max(sum_min_workers, CLOUDTIK_DEFAULT_MAX_WORKERS)


def fill_node_type_min_max_workers(config):
    """Sets default per-node max workers to global max_workers.
    This equivalent to setting the default per-node max workers to infinity,
    with the only upper constraint coming from the global max_workers.
    Sets default per-node min workers to zero.
    Also sets default max_workers for the head node to zero.
    """
    fill_default_max_workers(config)

    node_types = config["available_node_types"]
    for node_type_name in node_types:
        node_type_data = node_types[node_type_name]

        node_type_data.setdefault("min_workers", 0)
        if "max_workers" not in node_type_data:
            if node_type_name == config["head_node_type"]:
                logger.debug("setting max workers for head node type to 0")
                node_type_data.setdefault("max_workers", 0)
            else:
                global_max_workers = config["max_workers"]
                logger.debug(f"setting max workers for {node_type_name} to "
                            f"{global_max_workers}")
                node_type_data.setdefault("max_workers", global_max_workers)


def with_head_node_ip_environment_variables(head_ip, envs: Dict[str, Any] = None) -> Dict[str, Any]:
    if head_ip is None:
        head_ip = services.get_node_ip_address()
    if envs is None:
        envs = {}
    envs[CLOUDTIK_RUNTIME_ENV_HEAD_IP] = head_ip
    return envs


def with_node_ip_environment_variables(
        call_context, node_ip, provider, node_id):
    if node_ip is None:
        # Waiting for node internal ip for node
        if (provider is None) or (node_id is None):
            raise RuntimeError("Missing provider or node id for retrieving node ip.")

        deadline = time.time() + CLOUDTIK_NODE_START_WAIT_S
        node_ip = wait_for_cluster_ip(call_context, provider, node_id, deadline)
        if node_ip is None:
            raise RuntimeError("Failed to get node ip for node {}.".format(node_id))

    ip_envs = {CLOUDTIK_RUNTIME_ENV_NODE_IP: node_ip}
    return ip_envs


def hash_launch_conf(node_conf, auth):
    hasher = hashlib.sha1()
    # For hashing, we replace the path to the key with the
    # key itself. This is to make sure the hashes are the
    # same even if keys live at different locations on different
    # machines.
    full_auth = auth.copy()
    for key_type in ["ssh_private_key", "ssh_public_key"]:
        if key_type in auth:
            with open(os.path.expanduser(auth[key_type])) as key:
                full_auth[key_type] = key.read()
    hasher.update(
        json.dumps([node_conf, full_auth], sort_keys=True).encode("utf-8"))
    return hasher.hexdigest()


# Cache the file hashes to avoid rescanning it each time. Also, this avoids
# inadvertently restarting workers if the file mount content is mutated on the
# head node.
# TODO: This global cache needs to be protected for thread concurrency for future cases
_hash_cache = {}


def add_content_hashes(hasher, path, allow_non_existing_paths: bool = False):
    def add_hash_of_file(fpath):
        with open(fpath, "rb") as f:
            for chunk in iter(lambda: f.read(2 ** 20), b""):
                hasher.update(chunk)

    path = os.path.expanduser(path)
    if allow_non_existing_paths and not os.path.exists(path):
        return
    if os.path.isdir(path):
        dirs = []
        for dirpath, _, filenames in os.walk(path):
            dirs.append((dirpath, sorted(filenames)))
        for dirpath, filenames in sorted(dirs):
            hasher.update(dirpath.encode("utf-8"))
            for name in filenames:
                hasher.update(name.encode("utf-8"))
                fpath = os.path.join(dirpath, name)
                add_hash_of_file(fpath)
    else:
        add_hash_of_file(path)


def hash_runtime_conf(file_mounts,
                      cluster_synced_files,
                      extra_objs,
                      generate_file_mounts_contents_hash=False,
                      generate_node_types_runtime_hash=False,
                      config: Dict[str, Any] = None):
    """Returns two hashes, a runtime hash and file_mounts_content hash.

    The runtime hash is used to determine if the configuration or file_mounts
    contents have changed. It is used at launch time (cloudtik up) to determine if
    a restart is needed.

    The file_mounts_content hash is used to determine if the file_mounts or
    cluster_synced_files contents have changed. It is used at monitor time to
    determine if additional file syncing is needed.
    """
    runtime_hasher = hashlib.sha1()
    contents_hasher = hashlib.sha1()

    file_mounts_str = json.dumps(file_mounts, sort_keys=True).encode("utf-8")
    conf_str = (file_mounts_str +
                json.dumps(extra_objs, sort_keys=True).encode("utf-8"))

    runtime_hash_for_node_types = None
    head_node_contents_hash = None

    # Only generate a contents hash if generate_contents_hash is true or
    # if we need to generate the runtime_hash
    if conf_str not in _hash_cache or generate_file_mounts_contents_hash:
        head_node_contents_hash = hash_contents(
            contents_hasher, file_mounts)

        # Generate a new runtime_hash if its not cached
        # The runtime hash does not depend on the cluster_synced_files hash
        # because we do not want to restart nodes only if cluster_synced_files
        # contents have changed.
        if conf_str not in _hash_cache:
            runtime_hasher.update(conf_str)
            runtime_hasher.update(head_node_contents_hash.encode("utf-8"))
            _hash_cache[conf_str] = runtime_hasher.hexdigest()

        # Add cluster_synced_files to the file_mounts_content hash
        if cluster_synced_files is not None:
            for local_path in sorted(cluster_synced_files):
                # For cluster_synced_files, we let the path be non-existant
                # because its possible that the source directory gets set up
                # anytime over the life of the head node.
                add_content_hashes(contents_hasher, local_path, allow_non_existing_paths=True)

        file_mounts_contents_hash = contents_hasher.hexdigest()
    else:
        file_mounts_contents_hash = None

    if generate_node_types_runtime_hash:
        runtime_hash_for_node_types = hash_runtime_conf_for_node_types(
            config, file_mounts, file_mounts_str, head_node_contents_hash)

    return _hash_cache[conf_str], file_mounts_contents_hash, runtime_hash_for_node_types


def hash_contents(hasher, file_mounts):
    for local_path in sorted(file_mounts.values()):
        add_content_hashes(hasher, local_path)
    return hasher.hexdigest()


def hash_runtime_conf_for_node_types(
        config, file_mounts, file_mounts_str, head_node_contents_hash):
    runtime_hash_for_node_types = {}
    available_node_types = config["available_node_types"]
    head_node_type = config["head_node_type"]
    for node_type in available_node_types:
        if node_type == head_node_type:
            continue

        # Check whether we have node type specific commands
        # And runtime config, otherwise, we don't need
        if (not _has_node_type_specific_commands(config, node_type)) and (
                not _has_node_type_specific_runtime_config(config, node_type)):
            continue

        node_type_runtime_conf = {
            "worker_setup_commands": _get_node_type_specific_commands(
                config, node_type, "worker_setup_commands"),
            "worker_start_commands": _get_node_type_specific_commands(
                config, node_type, "worker_start_commands"),
            "runtime": _get_node_type_specific_runtime_config(
                config, node_type)
        }

        node_type_conf_str = (file_mounts_str +
                              json.dumps(node_type_runtime_conf, sort_keys=True).encode("utf-8"))
        if node_type_conf_str not in _hash_cache:
            if head_node_contents_hash is None:
                contents_hasher = hashlib.sha1()
                head_node_contents_hash = hash_contents(
                    contents_hasher, file_mounts)

            node_type_runtime_hasher = hashlib.sha1()
            node_type_runtime_hasher.update(node_type_conf_str)
            node_type_runtime_hasher.update(head_node_contents_hash.encode("utf-8"))

            node_type_runtime_hash = node_type_runtime_hasher.hexdigest()
            _hash_cache[node_type_conf_str] = node_type_runtime_hash
            runtime_hash_for_node_types[node_type] = node_type_runtime_hash
        else:
            runtime_hash_for_node_types[node_type] = _hash_cache[node_type_conf_str]

    return runtime_hash_for_node_types


def add_prefix(info_string, prefix):
    """Prefixes each line of info_string, except the first, by prefix."""
    lines = info_string.split("\n")
    prefixed_lines = [lines[0]]
    for line in lines[1:]:
        prefixed_line = ":".join([prefix, line])
        prefixed_lines.append(prefixed_line)
    prefixed_info_string = "\n".join(prefixed_lines)
    return prefixed_info_string


def format_pg(pg):
    strategy = pg["strategy"]
    bundles = pg["bundles"]
    shape_strs = []
    for bundle, count in bundles:
        shape_strs.append(f"{bundle} * {count}")
    bundles_str = ", ".join(shape_strs)
    return f"{bundles_str} ({strategy})"


def parse_placement_group_resource_str(
        placement_group_resource_str: str) -> Tuple[str, Optional[str]]:
    """Parse placement group resource in the form of following 3 cases:
    {resource_name}_group_{bundle_id}_{group_name};
    -> This case is ignored as it is duplicated to the case below.
    {resource_name}_group_{group_name};
    {resource_name}

    Returns:
        Tuple of (resource_name, placement_group_name, is_countable_resource).
        placement_group_name could be None if its not a placement group
        resource. is_countable_resource is True if the resource
        doesn't contain bundle index. We shouldn't count resources
        with bundle index because it will
        have duplicated resource information as
        wildcard resources (resource name without bundle index).
    """
    result = PLACEMENT_GROUP_RESOURCE_BUNDLED_PATTERN.match(
        placement_group_resource_str)
    if result:
        return (result.group(1), result.group(3), False)
    result = PLACEMENT_GROUP_RESOURCE_PATTERN.match(
        placement_group_resource_str)
    if result:
        return (result.group(1), result.group(2), True)
    return (placement_group_resource_str, None, True)


def get_usage_report(cluster_metrics_summary: ClusterMetricsSummary) -> str:
    # first collect resources used in placement groups
    placement_group_resource_usage = {}
    placement_group_resource_total = collections.defaultdict(float)
    for resource, (used, total) in cluster_metrics_summary.usage.items():
        (pg_resource_name, pg_name,
         is_countable) = parse_placement_group_resource_str(resource)
        if pg_name:
            if pg_resource_name not in placement_group_resource_usage:
                placement_group_resource_usage[pg_resource_name] = 0
            if is_countable:
                placement_group_resource_usage[pg_resource_name] += used
                placement_group_resource_total[pg_resource_name] += total
            continue

    usage_lines = []
    for resource, (used, total) in sorted(cluster_metrics_summary.usage.items()):
        if "node:" in resource:
            continue  # Skip the auto-added per-node "node:<ip>" resource.

        (_, pg_name, _) = parse_placement_group_resource_str(resource)
        if pg_name:
            continue  # Skip resource used by placement groups

        pg_used = 0
        pg_total = 0
        used_in_pg = resource in placement_group_resource_usage
        if used_in_pg:
            pg_used = placement_group_resource_usage[resource]
            pg_total = placement_group_resource_total[resource]
            # Used includes pg_total because when pgs are created
            # it allocates resources.
            # To get the real resource usage, we should subtract the pg
            # reserved resources from the usage and add pg used instead.
            used = used - pg_total + pg_used

        if resource in ["memory"]:
            to_GiB = 1 / 2**30
            line = (f" {(used * to_GiB):.2f}/"
                    f"{(total * to_GiB):.3f} GiB {resource}")
            if used_in_pg:
                line = line + (f" ({(pg_used * to_GiB):.2f} used of "
                               f"{(pg_total * to_GiB):.2f} GiB " +
                               "reserved in placement groups)")
            usage_lines.append(line)
        else:
            line = f" {used}/{total} {resource}"
            if used_in_pg:
                line += (f" ({pg_used} used of "
                         f"{pg_total} reserved in placement groups)")
            usage_lines.append(line)
    usage_report = "\n".join(usage_lines)
    return usage_report


def format_resource_demand_summary(
        resource_demand: List[Tuple[ResourceBundle, int]]) -> List[str]:
    def filter_placement_group_from_bundle(bundle: ResourceBundle):
        """filter placement group from bundle resource name. returns
        filtered bundle and a bool indicate if the bundle is using
        placement group.

        Example: {"CPU_group_groupid": 1} returns {"CPU": 1}, True
                 {"memory": 1} return {"memory": 1}, False
        """
        using_placement_group = False
        result_bundle = dict()
        for pg_resource_str, resource_count in bundle.items():
            (resource_name, pg_name,
             _) = parse_placement_group_resource_str(pg_resource_str)
            result_bundle[resource_name] = resource_count
            if pg_name:
                using_placement_group = True
        return (result_bundle, using_placement_group)

    bundle_demand = collections.defaultdict(int)
    pg_bundle_demand = collections.defaultdict(int)

    for bundle, count in resource_demand:
        (pg_filtered_bundle,
         using_placement_group) = filter_placement_group_from_bundle(bundle)

        # bundle is a special keyword for placement group ready tasks
        # do not report the demand for this.
        if "bundle" in pg_filtered_bundle.keys():
            continue

        bundle_demand[tuple(sorted(pg_filtered_bundle.items()))] += count
        if using_placement_group:
            pg_bundle_demand[tuple(sorted(
                pg_filtered_bundle.items()))] += count

    demand_lines = []
    for bundle, count in bundle_demand.items():
        line = f" {dict(bundle)}: {count}+ pending tasks/actors"
        if bundle in pg_bundle_demand:
            line += f" ({pg_bundle_demand[bundle]}+ using placement groups)"
        demand_lines.append(line)
    return demand_lines


def get_demand_report(lm_summary: ClusterMetricsSummary):
    demand_lines = []
    if lm_summary.resource_demand:
        demand_lines.extend(
            format_resource_demand_summary(lm_summary.resource_demand))
    for bundle, count in lm_summary.request_demand:
        line = f" {bundle}: {count}+ from request_resources()"
        demand_lines.append(line)
    if len(demand_lines) > 0:
        demand_report = "\n".join(demand_lines)
    else:
        demand_report = " (no resource demands)"
    return demand_report


def decode_cluster_scaling_time(status):
    status = status.decode("utf-8")
    as_dict = json.loads(status)
    report_time = float(as_dict["time"])
    return report_time


def format_info_string(lm_summary, scaler_summary, time=None):
    if time is None:
        time = datetime.now()
    header = "=" * 8 + f" Cluster Scaler status: {time} " + "=" * 8
    separator = "-" * len(header)
    available_node_report_lines = []
    for node_type, count in scaler_summary.active_nodes.items():
        line = f" {count} {node_type}"
        available_node_report_lines.append(line)
    available_node_report = "\n".join(available_node_report_lines)

    pending_lines = []
    for node_type, count in scaler_summary.pending_launches.items():
        line = f" {node_type}, {count} launching"
        pending_lines.append(line)
    for ip, node_type, status in scaler_summary.pending_nodes:
        line = f" {ip}: {node_type}, {status.lower()}"
        pending_lines.append(line)
    if pending_lines:
        pending_report = "\n".join(pending_lines)
    else:
        pending_report = " (no pending nodes)"

    failure_lines = []
    for ip, node_type in scaler_summary.failed_nodes:
        line = f" {ip}: {node_type}"
        failure_lines.append(line)
    failure_lines = failure_lines[:
                                  -constants.CLOUDTIK_MAX_FAILURES_DISPLAYED:
                                  -1]
    failure_report = "Recent failures:\n"
    if failure_lines:
        failure_report += "\n".join(failure_lines)
    else:
        failure_report += " (no failures)"

    # TODO: temporarily remove usage and deman report. To restore in the future
    #usage_report = get_usage_report(lm_summary)
    #demand_report = get_demand_report(lm_summary)

    #Resources
    #{separator}
    #Usage:
    #{usage_report}

    #Demands:
    #{demand_report}

    formatted_output = f"""{header}
Node status
{separator}
Healthy:
{available_node_report}
Pending:
{pending_report}
{failure_report}"""
    return formatted_output


def format_readonly_node_type(node_id: str):
    """The anonymous node type for readonly node provider nodes."""
    return "node_{}".format(node_id)


def format_no_node_type_string(node_type: dict):
    placement_group_resource_usage = {}
    regular_resource_usage = collections.defaultdict(float)
    for resource, total in node_type.items():
        (pg_resource_name, pg_name,
         is_countable) = parse_placement_group_resource_str(resource)
        if pg_name:
            if not is_countable:
                continue
            if pg_resource_name not in placement_group_resource_usage:
                placement_group_resource_usage[pg_resource_name] = 0
            placement_group_resource_usage[pg_resource_name] += total
        else:
            regular_resource_usage[resource] += total

    output_lines = [""]
    for resource, total in regular_resource_usage.items():
        output_line = f"{resource}: {total}"
        if resource in placement_group_resource_usage:
            pg_resource = placement_group_resource_usage[resource]
            output_line += f" ({pg_resource} reserved in placement groups)"
        output_lines.append(output_line)

    return "\n  ".join(output_lines)


def validate_workspace_config(config: Dict[str, Any]) -> None:
    """Required Dicts indicate that no extra fields can be introduced."""
    if not isinstance(config, dict):
        raise ValueError("Config {} is not a dictionary".format(config))

    with open(CLOUDTIK_WORKSPACE_SCHEMA_PATH) as f:
        schema = json.load(f)

    try:
        import jsonschema
    except (ModuleNotFoundError, ImportError) as e:
        # Don't log a warning message here. Logging be handled by upstream.
        raise e from None

    try:
        jsonschema.validate(config, schema)
    except jsonschema.ValidationError as e:
        # The validate method show very long message of the schema
        # and the instance data, we need show this only at verbose mode
        if cli_logger.verbosity > 0:
            raise e from None
        else:
            # For none verbose mode, show short message
            raise RuntimeError("JSON schema validation error: {}.".format(e.message)) from None

    provider = _get_workspace_provider(config["provider"], config["workspace_name"])
    provider.validate_config(config["provider"])


def check_cidr_conflict(cidr_block, cidr_blocks):
    existed_nets = [ipaddr.IPNetwork(cidr_block) for cidr_block in cidr_blocks]
    net = ipaddr.IPNetwork(cidr_block)

    for existed_net in existed_nets:
        if net.overlaps(existed_net):
            return False

    return True


def get_free_port():
    """ Get free port"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        test_port = constants.DEFAULT_PROXY_PORT
        while True:
            result = s.connect_ex(('127.0.0.1', test_port))
            if result != 0:
                return test_port
            else:
                test_port += 1


def kill_process_by_pid(pid):
    try:
        os.kill(pid, signal.SIGKILL)
        logger.debug("The process with PID {} has been killed.".format(pid))
    except OSError as e:
        logger.info("There is no process with PID {}".format(pid))


def check_process_exists(pid):
    return psutil.pid_exists(int(pid))


def get_proxy_info_file(cluster_name: str):
    proxy_info_file = os.path.join(tempfile.gettempdir(),
                                   "cloudtik-proxy-{}".format(cluster_name))
    return proxy_info_file


def _get_proxy_process_info(proxy_info_file: str):
    if os.path.exists(proxy_info_file):
        process_info = json.loads(open(proxy_info_file).read())
        if process_info.get("proxy") and process_info["proxy"].get("pid"):
            proxy_info = process_info["proxy"]
            return proxy_info["pid"], proxy_info.get("bind_address"), proxy_info["port"]
    return None, None, None


def get_safe_proxy_process_info(proxy_info_file: str):
    pid, bind_address, port = _get_proxy_process_info(proxy_info_file)
    if pid is None:
        return None, None, None

    if not check_process_exists(pid):
        return None, None, None

    return pid, bind_address, port


def get_proxy_bind_address_to_show(bind_address: str):
    if bind_address is None or bind_address == "":
        bind_address_to_show = "127.0.0.1"
    elif bind_address == "*" or bind_address == "0.0.0.0":
        bind_address_to_show = "this-node-ip"
    else:
        bind_address_to_show = bind_address
    return bind_address_to_show


def is_use_internal_ip(config: Dict[str, Any]) -> bool:
    return _is_use_internal_ip(config.get("provider", {}))


def _is_use_internal_ip(provider_config: Dict[str, Any]) -> bool:
    return provider_config.get("use_internal_ips", False)


def get_node_cluster_ip(provider: NodeProvider, node: str) -> str:
    return provider.internal_ip(node)


def wait_for_cluster_ip(call_context, provider, node_id, deadline):
    # if we have IP do not print waiting info
    ip = get_node_cluster_ip(provider, node_id)
    if ip is not None:
        return ip

    interval = CLOUDTIK_NODE_SSH_INTERVAL_S
    with call_context.cli_logger.group("Waiting for IP"):
        while time.time() < deadline and \
                not provider.is_terminated(node_id):
            ip = get_node_cluster_ip(provider, node_id)
            if ip is not None:
                call_context.cli_logger.labeled_value("Received", ip)
                return ip
            call_context.cli_logger.print(
                "Not yet available, retrying in {} seconds", str(interval))
            time.sleep(interval)

    return None


def get_node_working_ip(config: Dict[str, Any],
                        provider:NodeProvider, node:str) -> str:
    if is_use_internal_ip(config):
        node_ip = provider.internal_ip(node)
    else:
        node_ip = provider.external_ip(node)
    return node_ip


def get_head_working_ip(config: Dict[str, Any],
                        provider: NodeProvider, node: str) -> str:
    return get_node_working_ip(config, provider, node)


def update_nested_dict(target_dict, new_dict):
    for k, v in new_dict.items():
        if isinstance(v, collections.abc.Mapping):
            target_dict[k] = update_nested_dict(target_dict.get(k, {}), v)
        else:
            target_dict[k] = v
    return target_dict


def find_name_in_command(cmdline, name_to_find) -> bool:
    for arglist in cmdline:
        if name_to_find in arglist:
            return True
    return False


def is_alive_time(report_time):
    # TODO: We probably shouldn't rely on time here, but cloud providers
    # have very well synchronized NTP servers, so this should be fine in
    # practice.
    cur_time = time.time()

    # If the status is too old, the service has probably already died.
    delta = cur_time - report_time

    return delta < constants.HEALTHCHECK_EXPIRATION_S


def get_head_bootstrap_config():
    bootstrap_config_file = os.path.expanduser("~/cloudtik_bootstrap_config.yaml")
    if os.path.exists(bootstrap_config_file):
        return bootstrap_config_file
    raise RuntimeError("Cluster boostrap config not found. Incorrect head environment!")


def load_head_cluster_config() -> Dict[str, Any]:
    config_file = get_head_bootstrap_config()
    config = yaml.safe_load(open(config_file).read())
    return config


def get_attach_command(use_screen: bool,
                       use_tmux: bool,
                       new: bool = False):
    if use_tmux:
        if new:
            cmd = "tmux new"
        else:
            cmd = "tmux attach || tmux new"
    elif use_screen:
        if new:
            cmd = "screen -L"
        else:
            cmd = "screen -L -xRR"
    else:
        if new:
            raise ValueError(
                "--new only makes sense if passing --screen or --tmux")
        cmd = "$SHELL"
    return cmd


def is_docker_enabled(config: Dict[str, Any]) -> bool:
    return config.get(DOCKER_CONFIG_KEY, {}).get("enabled", False)


def kill_process_tree(pid, include_parent=True):
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    children = proc.children(recursive=True)
    if include_parent:
        children.append(proc)
    for p in children:
        try:
            p.kill()
        except psutil.NoSuchProcess:  # pragma: no cover
            pass


def get_nodes_info(provider, nodes, extras: bool = False,
                   available_node_types: Dict[str, Any] = None):
    return [get_node_info(provider, node,
                          extras, available_node_types) for node in nodes]


def get_node_info(provider, node, extras: bool = False,
                  available_node_types: Dict[str, Any] = None):
    node_info = provider.get_node_info(node)

    if extras:
        node_type = node_info.get(CLOUDTIK_TAG_USER_NODE_TYPE)
        if node_type is not None and node_type in available_node_types:
            resources = available_node_types[node_type].get("resources", {})
            node_info["CPU"] = resources.get("CPU", 0)
            node_info["memory-GB"] = resources.get("memory", 0) / pow(1024, 3)

    return node_info


def sum_worker_cpus(workers_info):
    total_cpus = 0
    for worker_info in workers_info:
        total_cpus += worker_info["CPU"]
    return total_cpus


def sum_worker_memory(workers_info):
    total_memory = 0
    for worker_info in workers_info:
        total_memory += worker_info["memory-GB"]
    return total_memory


def unescape_private_key(private_key: str):
    if private_key is None:
        return private_key

    if not private_key.startswith("-----BEGIN PRIVATE KEY-----\\n"):
        return private_key

    # Unescape "/n" to the real newline characters
    # use json load to do the work
    unescaped_private_key = json.loads("\"" + private_key + "\"")
    return unescaped_private_key


def escape_private_key(private_key: str):
    if private_key is None:
        return private_key

    if not private_key.startswith("-----BEGIN PRIVATE KEY-----\n"):
        return private_key

    # Escape the real newline characters
    # Use json dumps to do the work
    escaped_private_key = json.dumps(private_key)
    escaped_private_key = escaped_private_key.strip("\"\'")
    return escaped_private_key


def with_runtime_environment_variables(runtime_config, config, provider, node_id: str):
    all_runtime_envs = {}
    if runtime_config is None:
        return all_runtime_envs

    runtime_types = runtime_config.get(RUNTIME_TYPES_CONFIG_KEY, [])

    if len(runtime_types) > 0:
        all_runtime_envs[constants.CLOUDTIK_RUNTIME_ENV_RUNTIMES] = ",".join(runtime_types)

    # Iterate through all the runtimes
    for runtime_type in runtime_types:
        runtime = _get_runtime(runtime_type, runtime_config)
        runtime_envs = runtime.with_environment_variables(
            config, provider=provider, node_id=node_id)
        all_runtime_envs.update(runtime_envs)

    return all_runtime_envs


def runtime_validate_config(runtime_config, config, provider):
    if runtime_config is None:
        return

    # Iterate through all the runtimes
    runtime_types = runtime_config.get(RUNTIME_TYPES_CONFIG_KEY, [])
    for runtime_type in runtime_types:
        runtime = _get_runtime(runtime_type, runtime_config)
        runtime.validate_config(config, provider)


def runtime_prepare_config(
        runtime_config: Dict[str, Any],
        config: Dict[str, Any]) -> Dict[str, Any]:
    if runtime_config is None:
        return config

    # Iterate through all the runtimes
    runtime_types = runtime_config.get(RUNTIME_TYPES_CONFIG_KEY, [])
    for runtime_type in runtime_types:
        runtime = _get_runtime(runtime_type, runtime_config)
        config = runtime.prepare_config(config)

    return config


def runtime_verify_config(runtime_config, config, provider):
    if runtime_config is None:
        return

    # Iterate through all the runtimes
    runtime_types = runtime_config.get(RUNTIME_TYPES_CONFIG_KEY, [])
    for runtime_type in runtime_types:
        runtime = _get_runtime(runtime_type, runtime_config)
        runtime.verify_config(config, provider)


def get_runnable_command(runtime_config, target):
    if runtime_config is None:
        return None

    # Iterate through all the runtimes
    runtime_types = runtime_config.get(RUNTIME_TYPES_CONFIG_KEY, [])
    for runtime_type in runtime_types:
        runtime = _get_runtime(runtime_type, runtime_config)
        commands = runtime.get_runnable_command(target)
        if commands:
            return commands

    return None


def cluster_booting_completed(config, head_node_id):
    runtime_config = config.get(RUNTIME_CONFIG_KEY)
    if runtime_config is not  None:
        # Iterate through all the runtimes
        runtime_types = runtime_config.get(RUNTIME_TYPES_CONFIG_KEY, [])
        for runtime_type in runtime_types:
            runtime = _get_runtime(runtime_type, runtime_config)
            runtime.cluster_booting_completed(config, head_node_id)


def _get_runtime_config_object(config_home: str, provider_config, object_name: str):
    if not object_name.endswith(".yaml"):
        object_name += ".yaml"

    provider_type = provider_config["type"]

    path_to_config_file = os.path.join(config_home, provider_type, object_name)
    if not os.path.exists(path_to_config_file):
        path_to_config_file = os.path.join(config_home, object_name)

    if not os.path.exists(path_to_config_file):
        return {}

    with open(path_to_config_file) as f:
        config_object = yaml.safe_load(f) or {}

    return config_object


def get_useful_runtime_urls(runtime_config, head_cluster_ip):
    runtime_urls = []
    if runtime_config is None:
        return runtime_urls

    # Iterate through all the runtimes
    runtime_types = runtime_config.get(RUNTIME_TYPES_CONFIG_KEY, [])
    for runtime_type in runtime_types:
        runtime = _get_runtime(runtime_type, runtime_config)
        urls = runtime.get_useful_urls(head_cluster_ip)
        if urls:
            runtime_urls += urls

    return runtime_urls


def get_enabled_runtimes(config):
    return config.get(RUNTIME_CONFIG_KEY, {}).get(RUNTIME_TYPES_CONFIG_KEY, [])


def is_runtime_enabled(runtime_config, runtime_type:str):
    if runtime_config is None:
        return False

    runtime_types = runtime_config.get(RUNTIME_TYPES_CONFIG_KEY, [])
    if runtime_type in runtime_types:
        return True

    return False


def get_runtime_logs(runtimes: List[str]):
    runtime_logs = {}
    if runtimes is None:
        return runtime_logs

    # Iterate through all the runtimes
    for runtime_type in runtimes:
        runtime_cls = _get_runtime_cls(runtime_type)
        logs = runtime_cls.get_logs()
        if logs:
            runtime_logs.update(logs)

    return runtime_logs


def get_runtime_processes(runtimes: List[str]):
    runtime_processes = []
    if runtimes is None:
        return runtime_processes

    # Iterate through all the runtimes
    for runtime_type in runtimes:
        runtime_cls = _get_runtime_cls(runtime_type)
        processes = runtime_cls.get_processes()
        if processes:
            runtime_processes += processes

    return runtime_processes


def get_cluster_uri(config: Dict[str, Any]) -> str:
    return _get_cluster_uri(config["provider"]["type"], config["cluster_name"])


def _get_cluster_uri(provider_type: str, cluster_name: str) -> str:
    return CLOUDTIK_CLUSTER_URI_TEMPLATE.format(
        provider_type, cluster_name)


def _parse_runtime_list(runtimes: str):
    if runtimes is None or len(runtimes) == 0:
        return []

    runtime_list = []
    items = runtimes.split(",")
    for item in items:
        striped_item = item.strip()
        if len(striped_item) > 0:
            runtime_list.append(striped_item)
    return runtime_list


def verify_runtime_list(config: Dict[str, Any], runtimes: List[str]):
    runtime_types = config.get(RUNTIME_CONFIG_KEY, {}).get(RUNTIME_TYPES_CONFIG_KEY, [])
    for runtime in runtimes:
        if runtime in runtime_types or runtime == CLOUDTIK_RUNTIME_NAME:
            continue
        raise ValueError(f"Runtime {runtime} is not enabled in config.")


def get_verified_runtime_list(config: Dict[str, Any], runtimes: str):
    runtime_list = _parse_runtime_list(runtimes)
    verify_runtime_list(config, runtime_list)
    return runtime_list


def is_node_in_completed_status(provider, node_id) -> bool:
    node_tags = provider.node_tags(node_id)
    status = node_tags.get(CLOUDTIK_TAG_NODE_STATUS)
    if status is None:
        return False

    completed_states = [STATUS_UP_TO_DATE, STATUS_UPDATE_FAILED]
    if status in completed_states:
        return True
    return False


def check_for_single_worker_type(config: Dict[str, Any]):
    available_node_types = config["available_node_types"]
    head_node_type = config["head_node_type"]
    num_worker_type = 0
    for node_type in available_node_types:
        if node_type != head_node_type:
            num_worker_type += 1

    if num_worker_type > 1:
        raise ValueError("There are more than one worker types defined.")


def _gcd_of_numbers(numbers):
    num1 = numbers[0]
    num2 = numbers[1]
    gcd = math.gcd(num1, num2)
    for i in range(2, len(numbers)):
        gcd = math.gcd(gcd, numbers[i])
    return gcd


def get_preferred_cpu_bundle_size(config: Dict[str, Any]) -> Optional[int]:
    available_node_types = config.get("available_node_types")
    if available_node_types is None:
        return None

    cpu_sizes = []
    head_node_type = config["head_node_type"]
    for node_type in available_node_types:
        if node_type == head_node_type:
            continue

        resources = available_node_types[node_type].get("resources", {})
        cpu_total = resources.get("CPU", 0)
        if cpu_total > 0:
            cpu_sizes += [cpu_total]

    num_types = len(cpu_sizes)
    if num_types == 0:
        return None
    elif num_types == 1:
        return cpu_sizes[0]
    else:
        return _gcd_of_numbers(cpu_sizes)


def get_node_type(provider, node_id: str):
    node_tags = provider.node_tags(node_id)
    node_type = node_tags.get(CLOUDTIK_TAG_USER_NODE_TYPE)
    return node_type


def get_resource_of_node_type(config, node_type: str):
    available_node_types = config.get("available_node_types")
    if (available_node_types is None) or (node_type not in available_node_types):
        return None

    return available_node_types[node_type].get("resources", {})


def encode_cluster_secrets(secrets):
    return base64.b64encode(secrets).decode('utf-8')


def decode_cluster_secrets(encoded_secrets):
    return base64.b64decode(encoded_secrets.encode('utf-8'))


def subscribe_runtime_config():
    if CLOUDTIK_RUNTIME_ENV_SECRETS not in os.environ:
        raise RuntimeError("Not able to subscribe runtime config in lack of secrets.")

    node_type = os.environ.get(CLOUDTIK_RUNTIME_ENV_NODE_TYPE)
    if node_type:
        # Try getting node type specific runtime config
        runtime_config = retrieve_runtime_config(node_type)
        if runtime_config is not None:
            return runtime_config

    return retrieve_runtime_config()


def get_runtime_config_key(node_type: str):
    if node_type and len(node_type) > 0:
        runtime_config_key = CLOUDTIK_CLUSTER_RUNTIME_CONFIG_NODE_TYPE.format(node_type)
    else:
        runtime_config_key = CLOUDTIK_CLUSTER_RUNTIME_CONFIG
    return runtime_config_key


def retrieve_runtime_config(node_type: str = None):
    # Retrieve the runtime config
    runtime_config_key = get_runtime_config_key(node_type)
    encrypted_runtime_config = _get_key_from_kv( runtime_config_key)
    if encrypted_runtime_config is None:
        return None

    # Decrypt
    encoded_secrets = os.environ[CLOUDTIK_RUNTIME_ENV_SECRETS]
    secrets = decode_cluster_secrets(encoded_secrets)
    cipher = AESCipher(secrets)
    runtime_config_str = cipher.decrypt(encrypted_runtime_config)

    # To json object
    if runtime_config_str == "":
        return {}

    return json.loads(runtime_config_str)


def _get_minimal_nodes_before_update(config: Dict[str, Any], node_type: str):
    # Check the runtimes of the node type whether it needs to wait minimal before update
    runtime_config = _get_node_type_specific_runtime_config(config, node_type)
    if not runtime_config:
        return None

    # For each
    runtimes_require_minimal_nodes = []
    runtime_types = runtime_config.get(RUNTIME_TYPES_CONFIG_KEY, [])
    for runtime_type in runtime_types:
        runtime = _get_runtime(runtime_type, runtime_config)
        if runtime.require_minimal_nodes(config):
            runtimes_require_minimal_nodes += [runtime_type]

    if len(runtimes_require_minimal_nodes) > 0:
        node_type_config = config["available_node_types"][node_type]
        min_workers = node_type_config.get("min_workers", 0)
        if min_workers > 0:
            return {"minimal": min_workers, "runtimes": runtimes_require_minimal_nodes}
    return None


def _notify_minimal_nodes_reached(
        config: Dict[str, Any], node_type: str, nodes_info, minimal_nodes_info):
    runtime_config = _get_node_type_specific_runtime_config(config, node_type)
    if not runtime_config:
        return

    runtime_types = minimal_nodes_info["runtimes"]
    for runtime_type in runtime_types:
        runtime = _get_runtime(runtime_type, runtime_config)
        runtime.minimal_nodes_reached(config, nodes_info)


def subscribe_nodes_info():
    if CLOUDTIK_RUNTIME_ENV_NODE_TYPE not in os.environ:
        raise RuntimeError("Not able to subscribe nodes info in lack of node type.")
    node_type = os.environ[CLOUDTIK_RUNTIME_ENV_NODE_TYPE]
    return _retrieve_nodes_info(node_type)


def _retrieve_nodes_info(node_type):
    nodes_info_key = CLOUDTIK_CLUSTER_NODES_INFO_NODE_TYPE.format(node_type)
    nodes_info_str = _get_key_from_kv(nodes_info_key)
    if nodes_info_str is None:
        return None

    return json.loads(nodes_info_str)


def get_running_head_node(
        config: Dict[str, Any],
        _provider: Optional[NodeProvider] = None,
        _allow_uninitialized_state: bool = True,
) -> str:
    """Get a valid, running head node."""
    provider = _provider or _get_node_provider(config["provider"],
                                               config["cluster_name"])
    head_node_tags = {
        CLOUDTIK_TAG_NODE_KIND: NODE_KIND_HEAD,
    }
    nodes = provider.non_terminated_nodes(head_node_tags)
    head_node = None
    _backup_head_node = None
    for node in nodes:
        node_state = provider.node_tags(node).get(CLOUDTIK_TAG_NODE_STATUS)
        if node_state == STATUS_UP_TO_DATE:
            head_node = node
        else:
            _backup_head_node = node
            cli_logger.warning(f"Head node ({node}) is in state {node_state}.")

    if head_node is not None:
        return head_node
    else:
        if _allow_uninitialized_state and _backup_head_node is not None:
            cli_logger.warning(
                f"The head node being returned: {_backup_head_node} is not "
                "`up-to-date`. If you are not debugging a startup issue "
                "it is recommended to restart this cluster.")

            return _backup_head_node
        raise RuntimeError("Head node of cluster {} not found!".format(
            config["cluster_name"]))


def publish_cluster_variable(cluster_variable_name, cluster_variable_value):
    cluster_variable_key = CLOUDTIK_CLUSTER_VARIABLE.format(cluster_variable_name)
    return _put_key_to_kv(cluster_variable_key, cluster_variable_value)


def subscribe_cluster_variable(cluster_variable_name):
    cluster_variable_key = CLOUDTIK_CLUSTER_VARIABLE.format(cluster_variable_name)
    cluster_variable_value = _get_key_from_kv(cluster_variable_key)
    if cluster_variable_value is None:
        return None
    return cluster_variable_value.decode("utf-8")


def _get_key_from_kv(key):
    from cloudtik.core._private.state.kv_store import kv_get, kv_initialized, kv_initialize_with_address
    if not kv_initialized():
        if CLOUDTIK_RUNTIME_ENV_HEAD_IP not in os.environ:
            raise RuntimeError("Not able to cluster kv store in lack of head ip.")
        head_ip = os.environ[CLOUDTIK_RUNTIME_ENV_HEAD_IP]
        redis_address = "{}:{}".format(head_ip, CLOUDTIK_DEFAULT_PORT)
        kv_initialize_with_address(redis_address, CLOUDTIK_REDIS_DEFAULT_PASSWORD)

    return kv_get(key)


def _put_key_to_kv(key, value):
    from cloudtik.core._private.state.kv_store import kv_put, kv_initialized, kv_initialize_with_address
    if not kv_initialized():
        if CLOUDTIK_RUNTIME_ENV_HEAD_IP not in os.environ:
            raise RuntimeError("Not able to cluster kv store in lack of head ip.")
        head_ip = os.environ[CLOUDTIK_RUNTIME_ENV_HEAD_IP]
        redis_address = "{}:{}".format(head_ip, CLOUDTIK_DEFAULT_PORT)
        kv_initialize_with_address(redis_address, CLOUDTIK_REDIS_DEFAULT_PASSWORD)

    return kv_put(key, value)
