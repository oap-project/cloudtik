import base64
import collections
import errno
import io
import json
import logging
import os
import random
import signal
import socket
import subprocess
import sys
import time
from typing import List
import uuid
from shlex import quote
import redis

import cloudtik

# Import psutil and colorama after cloudtik so the packaged version is used.
import psutil

import cloudtik.core._private.constants as constants
import cloudtik.core._private.utils as utils
from cloudtik.core._private import core_utils
from cloudtik.core import tags
from cloudtik.core._private.core_utils import detect_fate_sharing_support, set_kill_on_parent_death_linux, \
    set_kill_child_on_death_win32
from cloudtik.core._private.state.control_state import ControlState

resource = None
if sys.platform != "win32":
    import resource

EXE_SUFFIX = ".exe" if sys.platform == "win32" else ""

# Location of the redis server.
CLOUDTIK_HOME = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "../..")
CLOUDTIK_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
CLOUDTIK_CORE_PRIVATE_SERVICE = "core/_private/service"
CLOUDTIK_REDIS_EXECUTABLE = os.path.join(
    CLOUDTIK_PATH, "core/thirdparty/redis/cloudtik-redis-server" + EXE_SUFFIX)

CLOUDTIK_JEMALLOC_LIB_PATH_ENV = "CLOUDTIK_JEMALLOC_LIB_PATH"
CLOUDTIK_JEMALLOC_CONF_ENV = "CLOUDTIK_JEMALLOC_CONF"
CLOUDTIK_JEMALLOC_PROFILE_ENV = "CLOUDTIK_JEMALLOC_PROFILE"

# Logger for this module. It should be configured at the entry point
# into the program. We provide a default configuration at
# entry points.
logger = logging.getLogger(__name__)

ProcessInfo = collections.namedtuple("ProcessInfo", [
    "process",
    "stdout_file",
    "stderr_file",
    "use_valgrind",
    "use_gdb",
    "use_valgrind_profiler",
    "use_perftools_profiler",
    "use_tmux",
])


def serialize_config(config):
    return base64.b64encode(json.dumps(config).encode("utf-8")).decode("utf-8")


def propagate_jemalloc_env_var(*, jemalloc_path: str, jemalloc_conf: str,
                               jemalloc_comps: List[str], process_type: str):
    """Read the jemalloc memory profiling related
        env var and return the dictionary that translates
        them to proper jemalloc related env vars.

        For example, if users specify `CLOUDTIK_JEMALLOC_LIB_PATH`,
        it is translated into `LD_PRELOAD` which is needed to
        run Jemalloc as a shared library.

        Params:
            jemalloc_path (str): The path to the jemalloc shared library.
            jemalloc_conf (str): `,` separated string of jemalloc config.
            jemalloc_comps List(str): The list of components
                that we will profile.
            process_type (str): The process type that needs jemalloc
                env var for memory profiling. If it doesn't match one of
                jemalloc_comps, the function will return an empty dict.

        Returns:
            dictionary of {env_var: value}
                that are needed to jemalloc profiling. The caller can
                call `dict.update(return_value_of_this_func)` to
                update the dict of env vars. If the process_type doesn't
                match jemalloc_comps, it will return an empty dict.
    """
    assert isinstance(jemalloc_comps, list)
    assert process_type is not None
    process_type = process_type.lower()
    if (not jemalloc_path or process_type not in jemalloc_comps):
        return {}

    env_vars = {
        "LD_PRELOAD": jemalloc_path,
    }
    if jemalloc_conf:
        env_vars.update({"MALLOC_CONF": jemalloc_conf})
    return env_vars


class ConsolePopen(subprocess.Popen):
    if sys.platform == "win32":

        def terminate(self):
            if isinstance(self.stdin, io.IOBase):
                self.stdin.close()
            if self._use_signals:
                self.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                super(ConsolePopen, self).terminate()

        def __init__(self, *args, **kwargs):
            # CREATE_NEW_PROCESS_GROUP is used to send Ctrl+C on Windows:
            # https://docs.python.org/3/library/subprocess.html#subprocess.Popen.send_signal
            new_pgroup = subprocess.CREATE_NEW_PROCESS_GROUP
            flags_to_add = 0
            if detect_fate_sharing_support():
                # If we don't have kernel-mode fate-sharing, then don't do this
                # because our children need to be in out process group for
                # the process reaper to properly terminate them.
                flags_to_add = new_pgroup
            flags_key = "creationflags"
            if flags_to_add:
                kwargs[flags_key] = (kwargs.get(flags_key) or 0) | flags_to_add
            self._use_signals = (kwargs[flags_key] & new_pgroup)
            super(ConsolePopen, self).__init__(*args, **kwargs)


def address(ip_address, port):
    return ip_address + ":" + str(port)


def new_port(lower_bound=10000, upper_bound=65535, denylist=None):
    if not denylist:
        denylist = set()
    port = random.randint(lower_bound, upper_bound)
    retry = 0
    while port in denylist:
        if retry > 100:
            break
        port = random.randint(lower_bound, upper_bound)
        retry += 1
    if retry > 100:
        raise ValueError("Failed to find a new port from the range "
                         f"{lower_bound}-{upper_bound}. Denylist: {denylist}")
    return port


def find_redis_address(address=None):
    """
    Attempts to find all valid redis addresses on this node.

    Returns:
        Set of detected Redis instances.
    """
    # Currently, this extracts the deprecated --redis-address from the command
    # that launched the service running on this node, if any.
    pids = psutil.pids()
    redis_addresses = set()
    for pid in pids:
        try:
            proc = psutil.Process(pid)
            # HACK: Workaround for UNIX idiosyncrasy
            # Normally, cmdline() is supposed to return the argument list.
            # But it in some cases (such as when setproctitle is called),
            # an arbitrary string resembling a command-line is stored in
            # the first argument.
            # Explanation: https://unix.stackexchange.com/a/432681
            # More info: https://github.com/giampaolo/psutil/issues/1179
            cmdline = proc.cmdline()
            # NOTE: To support Windows, we can't use
            # `os.path.basename(cmdline[0]) == "abc"` here.
            # TODO: use the right way to detect the redis
            if utils.find_name_in_command(cmdline, "cloudtik_cluster_controller") or \
                    utils.find_name_in_command(cmdline, "cloudtik_node_controller") or \
                    utils.find_name_in_command(cmdline, "cloudtik_log_monitor"):
                for arglist in cmdline:
                    # Given we're merely seeking --redis-address, we just split
                    # every argument on spaces for now.
                    for arg in arglist.split(" "):
                        # TODO: Find a robust solution for locating Redis.
                        if arg.startswith("--redis-address="):
                            proc_addr = arg.split("=")[1]
                            if address is not None and address != proc_addr:
                                continue
                            redis_addresses.add(proc_addr)
        except psutil.AccessDenied:
            pass
        except psutil.NoSuchProcess:
            pass
    return redis_addresses


def get_address_to_use_or_die():
    """
    Attempts to find an address for an existing cluster if it is not
    already specified as an environment variable.
    Returns:
        A string to redis address
    """
    return os.environ.get(constants.CLOUDTIK_ADDRESS_ENV,
                          find_redis_address_or_die())


def find_redis_address_or_die():

    redis_addresses = find_redis_address()
    if len(redis_addresses) > 1:
        raise ConnectionError(
            f"Found multiple active Redis instances: {redis_addresses}. "
            "Please specify the one to connect to by setting `address`.")
        sys.exit(1)
    elif not redis_addresses:
        raise ConnectionError(
            "Could not find any running Redis instance. "
            "Please specify the one to connect to by setting `address`.")
    return redis_addresses.pop()


def wait_for_node(redis_address,
                  node_ip_address,
                  redis_password=None,
                  timeout=30):
    """Wait until this node has appeared in the client table.

    Args:
        redis_address (str): The redis address.
        node_ip_address (str): the node ip address to wait for
        redis_password (str): the redis password.
        timeout: The amount of time in seconds to wait before raising an
            exception.

    Raises:
        TimeoutError: An exception is raised if the timeout expires before
            the node appears in the client table.
    """

    redis_ip_address, redis_port = redis_address.split(":")
    wait_for_redis_to_start(redis_ip_address, redis_port, redis_password)
    # TODO (haifeng): implement control state for node services
    global_state = ControlState()
    global_state.initialize_control_state(redis_address, redis_port, redis_password)
    start_time = time.time()
    while time.time() - start_time < timeout:
        # FIXME  It depends on the implementation of global_state_accessor to pass. Skip it temporarily.
        # clients = global_state.node_table()
        # node_ip_addresses = [
        #     client["node_ip_address"] for client in clients
        # ]
        node_ip_addresses = [node_ip_address]
        if node_ip_address in node_ip_addresses:
            return
        else:
            time.sleep(0.1)
    raise TimeoutError("Timed out while waiting for node to startup.")


def validate_redis_address(address):
    """Validates address parameter.

    Returns:
        redis_address: string containing the full <host:port> address.
        redis_ip: string representing the host portion of the address.
        redis_port: integer representing the port portion of the address.
    """

    if address == "auto":
        address = find_redis_address_or_die()
    redis_address = address_to_ip(address)

    redis_address_parts = redis_address.split(":")
    if len(redis_address_parts) != 2:
        raise ValueError(f"Malformed address. Expected '<host>:<port>',"
                         f" but got {redis_address} from {address}.")
    redis_ip = redis_address_parts[0]
    try:
        redis_port = int(redis_address_parts[1])
    except ValueError:
        raise ValueError("Malformed address port. Must be an integer.")
    if redis_port < 1024 or redis_port > 65535:
        raise ValueError("Invalid address port. Must "
                         "be between 1024 and 65535.")

    return redis_address, redis_ip, redis_port


def address_to_ip(address):
    """Convert a hostname to a numerical IP addresses in an address.

    This should be a no-op if address already contains an actual numerical IP
    address.

    Args:
        address: This can be either a string containing a hostname (or an IP
            address) and a port or it can be just an IP address.

    Returns:
        The same address but with the hostname replaced by a numerical IP
            address.
    """
    address_parts = address.split(":")
    ip_address = socket.gethostbyname(address_parts[0])
    # Make sure localhost isn't resolved to the loopback ip
    if ip_address == "127.0.0.1":
        ip_address = get_node_ip_address()
    return ":".join([ip_address] + address_parts[1:])


def node_ip_address_from_perspective(address):
    """IP address by which the local node can be reached *from* the `address`.

    Args:
        address (str): The IP address and port of any known live service on the
            network you care about.

    Returns:
        The IP address by which the local node can be reached from the address.
    """
    ip_address, port = address.split(":")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # This command will raise an exception if there is no internet
        # connection.
        s.connect((ip_address, int(port)))
        node_ip_address = s.getsockname()[0]
    except OSError as e:
        node_ip_address = "127.0.0.1"
        # [Errno 101] Network is unreachable
        if e.errno == errno.ENETUNREACH:
            try:
                # try get node ip address from host name
                host_name = socket.getfqdn(socket.gethostname())
                node_ip_address = socket.gethostbyname(host_name)
            except Exception:
                pass
    finally:
        s.close()

    return node_ip_address


def get_node_ip_address(address="8.8.8.8:53"):
    if sys.platform == "darwin" or sys.platform == "win32":
        # Due to the mac osx/windows firewall,
        # we use loopback ip as the ip address
        # to prevent security popups.
        return "127.0.0.1"
    return node_ip_address_from_perspective(address)


def create_redis_client(redis_address, password=None):
    """Create a Redis client.

    Args:
        The IP address, port, and password of the Redis server.

    Returns:
        A Redis client.
    """
    if not hasattr(create_redis_client, "instances"):
        create_redis_client.instances = {}
    else:
        cli = create_redis_client.instances.get(redis_address)
        if cli is not None:
            try:
                cli.ping()
                return cli
            except Exception:
                create_redis_client.instances.pop(redis_address)

    _, redis_ip_address, redis_port = validate_redis_address(redis_address)
    # For this command to work, some other client (on the same machine
    # as Redis) must have run "CONFIG SET protected-mode no".
    create_redis_client.instances[redis_address] = redis.StrictRedis(
        host=redis_ip_address, port=int(redis_port), password=password)

    return create_redis_client.instances[redis_address]


def start_cloudtik_process(command,
                           process_type,
                           fate_share,
                           env_updates=None,
                           cwd=None,
                           use_valgrind=False,
                           use_gdb=False,
                           use_valgrind_profiler=False,
                           use_perftools_profiler=False,
                           use_tmux=False,
                           stdout_file=None,
                           stderr_file=None,
                           pipe_stdin=False):
    """Start one of the service processes.

    TODO(rkn): We need to figure out how these commands interact. For example,
    it may only make sense to start a process in gdb if we also start it in
    tmux. Similarly, certain combinations probably don't make sense, like
    simultaneously running the process in valgrind and the profiler.

    Args:
        command (List[str]): The command to use to start the process.
        process_type (str): The type of the process that is being started
        fate_share: If true, the child will be killed if its parent (us) dies.
            True must only be passed after detection of this functionality.
        env_updates (dict): A dictionary of additional environment variables to
            run the command with (in addition to the caller's environment
            variables).
        cwd (str): The directory to run the process in.
        use_valgrind (bool): True if we should start the process in valgrind.
        use_gdb (bool): True if we should start the process in gdb.
        use_valgrind_profiler (bool): True if we should start the process in
            the valgrind profiler.
        use_perftools_profiler (bool): True if we should profile the process
            using perftools.
        use_tmux (bool): True if we should start the process in tmux.
        stdout_file: A file handle opened for writing to redirect stdout to. If
            no redirection should happen, then this should be None.
        stderr_file: A file handle opened for writing to redirect stderr to. If
            no redirection should happen, then this should be None.
        pipe_stdin: If true, subprocess.PIPE will be passed to the process as
            stdin.

    Returns:
        Information about the process that was started including a handle to
            the process that was started.
    """
    # Detect which flags are set through environment variables.
    valgrind_env_var = f"CLOUDTIK_{process_type.upper()}_VALGRIND"
    if os.environ.get(valgrind_env_var) == "1":
        logger.info("Detected environment variable '%s'.", valgrind_env_var)
        use_valgrind = True
    valgrind_profiler_env_var = f"CLOUDTIK_{process_type.upper()}_VALGRIND_PROFILER"
    if os.environ.get(valgrind_profiler_env_var) == "1":
        logger.info("Detected environment variable '%s'.",
                    valgrind_profiler_env_var)
        use_valgrind_profiler = True
    perftools_profiler_env_var = (f"CLOUDTIK_{process_type.upper()}"
                                  "_PERFTOOLS_PROFILER")
    if os.environ.get(perftools_profiler_env_var) == "1":
        logger.info("Detected environment variable '%s'.",
                    perftools_profiler_env_var)
        use_perftools_profiler = True
    tmux_env_var = f"CLOUDTIK_{process_type.upper()}_TMUX"
    if os.environ.get(tmux_env_var) == "1":
        logger.info("Detected environment variable '%s'.", tmux_env_var)
        use_tmux = True
    gdb_env_var = f"CLOUDTIK_{process_type.upper()}_GDB"
    if os.environ.get(gdb_env_var) == "1":
        logger.info("Detected environment variable '%s'.", gdb_env_var)
        use_gdb = True
    # Jemalloc memory profiling.
    jemalloc_lib_path = os.environ.get(CLOUDTIK_JEMALLOC_LIB_PATH_ENV)
    jemalloc_conf = os.environ.get(CLOUDTIK_JEMALLOC_CONF_ENV)
    jemalloc_comps = os.environ.get(CLOUDTIK_JEMALLOC_PROFILE_ENV)
    jemalloc_comps = [] if not jemalloc_comps else jemalloc_comps.split(",")
    jemalloc_env_vars = propagate_jemalloc_env_var(
        jemalloc_path=jemalloc_lib_path,
        jemalloc_conf=jemalloc_conf,
        jemalloc_comps=jemalloc_comps,
        process_type=process_type)
    use_jemalloc_mem_profiler = len(jemalloc_env_vars) > 0

    if sum([
            use_gdb,
            use_valgrind,
            use_valgrind_profiler,
            use_perftools_profiler,
            use_jemalloc_mem_profiler,
    ]) > 1:
        raise ValueError("At most one of the 'use_gdb', 'use_valgrind', "
                         "'use_valgrind_profiler', 'use_perftools_profiler', "
                         "and 'use_jemalloc_mem_profiler' flags can "
                         "be used at a time.")
    if env_updates is None:
        env_updates = {}
    if not isinstance(env_updates, dict):
        raise ValueError("The 'env_updates' argument must be a dictionary.")

    modified_env = os.environ.copy()
    modified_env.update(env_updates)

    if use_gdb:
        if not use_tmux:
            raise ValueError(
                "If 'use_gdb' is true, then 'use_tmux' must be true as well.")

        # TODO(suquark): Any better temp file creation here?
        gdb_init_path = os.path.join(utils.get_cloudtik_temp_dir(),
                                     f"gdb_init_{process_type}_{time.time()}")
        process_path = command[0]
        process_args = command[1:]
        run_args = " ".join(["'{}'".format(arg) for arg in process_args])
        with open(gdb_init_path, "w") as gdb_init_file:
            gdb_init_file.write(f"run {run_args}")
        command = ["gdb", process_path, "-x", gdb_init_path]

    if use_valgrind:
        command = [
            "valgrind",
            "--track-origins=yes",
            "--leak-check=full",
            "--show-leak-kinds=all",
            "--leak-check-heuristics=stdstring",
            "--error-exitcode=1",
        ] + command

    if use_valgrind_profiler:
        command = ["valgrind", "--tool=callgrind"] + command

    if use_perftools_profiler:
        modified_env["LD_PRELOAD"] = os.environ["PERFTOOLS_PATH"]
        modified_env["CPUPROFILE"] = os.environ["PERFTOOLS_LOGFILE"]

    if use_jemalloc_mem_profiler:
        logger.info(f"Jemalloc profiling will be used for {process_type}. "
                    f"env vars: {jemalloc_env_vars}")
        modified_env.update(jemalloc_env_vars)

    if use_tmux:
        # The command has to be created exactly as below to ensure that it
        # works on all versions of tmux. (Tested with tmux 1.8-5, travis'
        # version, and tmux 2.1)
        command = ["tmux", "new-session", "-d", f"{' '.join(command)}"]

    if fate_share:
        assert detect_fate_sharing_support(), (
            "kernel-level fate-sharing must only be specified if "
            "detect_fate_sharing_support() has returned True")

    def preexec_fn():
        import signal
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
        if fate_share and sys.platform.startswith("linux"):
            set_kill_on_parent_death_linux()

    win32_fate_sharing = fate_share and sys.platform == "win32"
    # With Windows fate-sharing, we need special care:
    # The process must be added to the job before it is allowed to execute.
    # Otherwise, there's a race condition: the process might spawn children
    # before the process itself is assigned to the job.
    # After that point, its children will not be added to the job anymore.
    CREATE_SUSPENDED = 0x00000004  # from Windows headers
    if sys.platform == "win32":
        # CreateProcess, which underlies Popen, is limited to
        # 32,767 characters, including the Unicode terminating null
        # character
        total_chrs = sum([len(x) for x in command])
        if total_chrs > 31766:
            raise ValueError(
                f"command is limited to a total of 31767 characters, "
                f"got {total_chrs}")

    process = ConsolePopen(
        command,
        env=modified_env,
        cwd=cwd,
        stdout=stdout_file,
        stderr=stderr_file,
        stdin=subprocess.PIPE if pipe_stdin else None,
        preexec_fn=preexec_fn if sys.platform != "win32" else None,
        creationflags=CREATE_SUSPENDED if win32_fate_sharing else 0)

    if win32_fate_sharing:
        try:
            set_kill_child_on_death_win32(process)
            psutil.Process(process.pid).resume()
        except (psutil.Error, OSError):
            process.kill()
            raise

    def _get_stream_name(stream):
        if stream is not None:
            try:
                return stream.name
            except AttributeError:
                return str(stream)
        return None

    return ProcessInfo(
        process=process,
        stdout_file=_get_stream_name(stdout_file),
        stderr_file=_get_stream_name(stderr_file),
        use_valgrind=use_valgrind,
        use_gdb=use_gdb,
        use_valgrind_profiler=use_valgrind_profiler,
        use_perftools_profiler=use_perftools_profiler,
        use_tmux=use_tmux)


def wait_for_redis_to_start(redis_ip_address, redis_port, password=None):
    """Wait for a Redis server to be available.

    This is accomplished by creating a Redis client and sending a random
    command to the server until the command gets through.

    Args:
        redis_ip_address (str): The IP address of the redis server.
        redis_port (int): The port of the redis server.
        password (str): The password of the redis server.

    Raises:
        Exception: An exception is raised if we could not connect with Redis.
    """
    redis_client = redis.StrictRedis(
        host=redis_ip_address, port=redis_port, password=password)
    # Wait for the Redis server to start.
    num_retries = constants.CLOUDTIK_START_REDIS_WAIT_RETRIES
    delay = 0.001
    for i in range(num_retries):
        try:
            # Run some random command and see if it worked.
            logger.debug(
                "Waiting for redis server at {}:{} to respond...".format(
                    redis_ip_address, redis_port))
            redis_client.client_list()
        # If the Redis service is delayed getting set up for any reason, we may
        # get a redis.ConnectionError: Error 111 connecting to host:port.
        # Connection refused.
        # Unfortunately, redis.ConnectionError is also the base class of
        # redis.AuthenticationError. We *don't* want to obscure a
        # redis.AuthenticationError, because that indicates the user provided a
        # bad password. Thus a double except clause to ensure a
        # redis.AuthenticationError isn't trapped here.
        except redis.AuthenticationError as authEx:
            raise RuntimeError("Unable to connect to Redis at {}:{}.".format(
                redis_ip_address, redis_port)) from authEx
        except redis.ConnectionError as connEx:
            if i >= num_retries - 1:
                raise RuntimeError(
                    f"Unable to connect to Redis at {redis_ip_address}:"
                    f"{redis_port} after {num_retries} retries. Check that "
                    f"{redis_ip_address}:{redis_port} is reachable from this "
                    "machine. If it is not, your firewall may be blocking "
                    "this port. If the problem is a flaky connection, try "
                    "setting the environment variable "
                    "`CLOUDTIK_START_REDIS_WAIT_RETRIES` to increase the number of"
                    " attempts to ping the Redis server.") from connEx
            # Wait a little bit.
            time.sleep(delay)
            delay *= 2
        else:
            break
    else:
        raise RuntimeError(
            f"Unable to connect to Redis (after {num_retries} retries). "
            "If the Redis instance is on a different machine, check that "
            "your firewall and relevant ports are configured properly. "
            "You can also set the environment variable "
            "`CLOUDTIK_START_REDIS_WAIT_RETRIES` to increase the number of "
            "attempts to ping the Redis server.")


def _compute_version_info():
    """Compute the versions of Python, and CloudTik.

    Returns:
        A tuple containing the version information.
    """
    cloudtik_version = cloudtik.__version__
    python_version = ".".join(map(str, sys.version_info[:3]))
    return cloudtik_version, python_version


def _put_version_info_in_redis(redis_client):
    """Store version information in Redis.

    This will be used to detect if workers or drivers are started using
    different versions of Python, or CloudTik.

    Args:
        redis_client: A client for the primary Redis shard.
    """
    redis_client.set("VERSION_INFO", json.dumps(_compute_version_info()))


def check_version_info(redis_client):
    """Check if various version info of this process is correct.

    This will be used to detect if workers or drivers are started using
    different versions of Python, or CloudTik. If the version
    information is not present in Redis, then no check is done.

    Args:
        redis_client: A client for the primary Redis shard.

    Raises:
        Exception: An exception is raised if there is a version mismatch.
    """
    redis_reply = redis_client.get("VERSION_INFO")

    # Don't do the check if there is no version information in Redis. This
    # is to make it easier to do things like start the processes by hand.
    if redis_reply is None:
        return

    true_version_info = tuple(
        json.loads(core_utils.decode(redis_reply)))
    version_info = _compute_version_info()
    if version_info != true_version_info:
        node_ip_address = get_node_ip_address()
        error_message = ("Version mismatch: The cluster was started with:\n"
                         "    CloudTik: " + true_version_info[0] + "\n"
                         "    Python: " + true_version_info[1] + "\n"
                         "This process on node " + node_ip_address +
                         " was started with:" + "\n"
                         "    CloudTik: " + version_info[0] + "\n"
                         "    Python: " + version_info[1] + "\n")
        if version_info[:2] != true_version_info[:2]:
            raise RuntimeError(error_message)
        else:
            logger.warning(error_message)


def start_reaper(fate_share=None):
    """Start the reaper process.

    This is a lightweight process that simply
    waits for its parent process to die and then terminates its own
    process group. This allows us to ensure that cloudtik processes are always
    terminated properly so long as that process itself isn't SIGKILLed.

    Returns:
        ProcessInfo for the process that was started.
    """
    # Make ourselves a process group leader so that the reaper can clean
    # up other processes without killing the process group of the
    # process that started us.
    try:
        if sys.platform != "win32":
            os.setpgrp()
    except OSError as e:
        errcode = e.errno
        if errcode == errno.EPERM and os.getpgrp() == os.getpid():
            # Nothing to do; we're already a session leader.
            pass
        else:
            logger.warning("setpgrp failed, processes may not be "
                           "cleaned up properly: {}.".format(e))
            # Don't start the reaper in this case as it could result in killing
            # other user processes.
            return None

    reaper_filepath = os.path.join(CLOUDTIK_PATH, CLOUDTIK_CORE_PRIVATE_SERVICE,
                                   "cloudtik_process_reaper.py")
    command = [sys.executable, "-u", reaper_filepath]
    process_info = start_cloudtik_process(
        command,
        constants.PROCESS_TYPE_REAPER,
        pipe_stdin=True,
        fate_share=fate_share)
    return process_info


def start_redis(node_ip_address,
                redirect_files,
                resource_spec,
                session_dir_path,
                port=None,
                redis_shard_ports=None,
                num_redis_shards=1,
                redis_max_clients=None,
                redirect_worker_output=False,
                password=None,
                fate_share=None,
                external_addresses=None,
                port_denylist=None):
    """Start the Redis global state store.

    Args:
        node_ip_address: The IP address of the current node. This is only used
            for recording the log filenames in Redis.
        redirect_files: The list of (stdout, stderr) file pairs.
        resource_spec (ResourceSpec): Resources for the node.
        session_dir_path (str): Path to the session directory of
            this cluster.
        port (int): If provided, the primary Redis shard will be started on
            this port.
        redis_shard_ports: A list of the ports to use for the non-primary Redis
            shards.
        num_redis_shards (int): If provided, the number of Redis shards to
            start, in addition to the primary one. The default value is one
            shard.
        redis_max_clients: If this is provided, we will attempt to configure
            Redis with this maxclients number.
        redirect_worker_output (bool): True if worker output should be
            redirected to a file and false otherwise. Workers will have access
            to this value when they start up.
        password (str): Prevents external clients without the password
            from connecting to Redis if provided.
        port_denylist (set): A set of denylist ports that shouldn't
            be used when allocating a new port.

    Returns:
        A tuple of the address for the primary Redis shard, a list of
            addresses for the remaining shards, and the processes that were
            started.
    """
    processes = []

    if external_addresses is not None:
        primary_redis_address = external_addresses[0]
        [primary_redis_ip, port] = primary_redis_address.split(":")
        port = int(port)
        redis_address = address(primary_redis_ip, port)
        primary_redis_client = create_redis_client(
            "%s:%s" % (primary_redis_ip, port), password=password)
    else:
        if len(redirect_files) != 1 + num_redis_shards:
            raise ValueError(
                "The number of redirect file pairs should be equal "
                "to the number of redis shards (including the "
                "primary shard) we will start.")
        if redis_shard_ports is None:
            redis_shard_ports = num_redis_shards * [None]
        elif len(redis_shard_ports) != num_redis_shards:
            raise RuntimeError(
                "The number of Redis shard ports does not match "
                "the number of Redis shards.")
        redis_executable = CLOUDTIK_REDIS_EXECUTABLE

        redis_stdout_file, redis_stderr_file = redirect_files[0]
        # If no port is given, fallback to default Redis port for the primary
        # shard.
        if port is None:
            port = constants.CLOUDTIK_DEFAULT_PORT
            num_retries = 20
        else:
            num_retries = 1
        # Start the primary Redis shard.
        port, p = _start_redis_instance(
            redis_executable,
            session_dir_path,
            bind_address=node_ip_address,
            port=port,
            password=password,
            redis_max_clients=redis_max_clients,
            num_retries=num_retries,
            # Below we use None to indicate no limit on the memory of the
            # primary Redis shard.
            redis_max_memory=None,
            stdout_file=redis_stdout_file,
            stderr_file=redis_stderr_file,
            fate_share=fate_share,
            port_denylist=port_denylist,
            listen_to_localhost_only=(node_ip_address == "127.0.0.1"))
        processes.append(p)
        redis_address = address(node_ip_address, port)
        primary_redis_client = redis.StrictRedis(
            host=node_ip_address, port=port, password=password)

    # Register the number of Redis shards in the primary shard, so that clients
    # know how many redis shards to expect under RedisShards.
    primary_redis_client.set("NumRedisShards", str(num_redis_shards))

    # Deleting the key to avoid duplicated rpush.
    primary_redis_client.delete("RedisShards")

    # Put the redirect_worker_output bool in the Redis shard so that workers
    # can access it and know whether or not to redirect their output.
    primary_redis_client.set("RedirectOutput", 1
                             if redirect_worker_output else 0)

    # Init job counter to GCS.
    primary_redis_client.set("JobCounter", 0)

    # Store version information in the primary Redis shard.
    _put_version_info_in_redis(primary_redis_client)

    # Calculate the redis memory.
    # TODO (haifeng): if not specified, calculate according to the node memory
    assert resource_spec.resolved()
    redis_max_memory = resource_spec.redis_max_memory

    # Start other Redis shards. Each Redis shard logs to a separate file,
    # prefixed by "redis-<shard number>".
    redis_shards = []
    # If Redis shard ports are not provided, start the port range of the
    # other Redis shards at a high, random port.
    last_shard_port = new_port(denylist=port_denylist) - 1
    for i in range(num_redis_shards):
        if external_addresses is not None:
            shard_address = external_addresses[i + 1]
        else:
            redis_stdout_file, redis_stderr_file = redirect_files[i + 1]
            redis_executable = CLOUDTIK_REDIS_EXECUTABLE
            redis_shard_port = redis_shard_ports[i]
            # If no shard port is given, try to start this shard's Redis
            # instance on the port right after the last shard's port.
            if redis_shard_port is None:
                redis_shard_port = last_shard_port + 1
                num_retries = 20
            else:
                num_retries = 1

            redis_shard_port, p = _start_redis_instance(
                redis_executable,
                session_dir_path,
                bind_address=node_ip_address,
                port=redis_shard_port,
                password=password,
                redis_max_clients=redis_max_clients,
                num_retries=num_retries,
                redis_max_memory=redis_max_memory,
                stdout_file=redis_stdout_file,
                stderr_file=redis_stderr_file,
                fate_share=fate_share,
                port_denylist=port_denylist,
                listen_to_localhost_only=(node_ip_address == "127.0.0.1"))
            processes.append(p)

            shard_address = address(node_ip_address, redis_shard_port)
            last_shard_port = redis_shard_port

        redis_shards.append(shard_address)
        # Store redis shard information in the primary redis shard.
        primary_redis_client.rpush("RedisShards", shard_address)

    return redis_address, redis_shards, processes


def _start_redis_instance(executable,
                          session_dir_path,
                          bind_address,
                          port,
                          redis_max_clients=None,
                          num_retries=20,
                          stdout_file=None,
                          stderr_file=None,
                          password=None,
                          redis_max_memory=None,
                          fate_share=None,
                          port_denylist=None,
                          listen_to_localhost_only=False):
    """Start a single Redis server.

    Notes:
        We will initially try to start the Redis instance at the given port,
        and then try at most `num_retries - 1` times to start the Redis
        instance at successive random ports.

    Args:
        executable (str): Full path of the redis-server executable.
        session_dir_path (str): Path to the session directory of
            this cluster.
        bind_address: The address to bind. None to bind all
        port (int): Try to start a Redis server at this port.
        redis_max_clients: If this is provided, we will attempt to configure
            Redis with this maxclients number.
        num_retries (int): The number of times to attempt to start Redis at
            successive ports.
        stdout_file: A file handle opened for writing to redirect stdout to. If
            no redirection should happen, then this should be None.
        stderr_file: A file handle opened for writing to redirect stderr to. If
            no redirection should happen, then this should be None.
        password (str): Prevents external clients without the password
            from connecting to Redis if provided.
        redis_max_memory: The max amount of memory (in bytes) to allow redis
            to use, or None for no limit. Once the limit is exceeded, redis
            will start LRU eviction of entries.
        port_denylist (set): A set of denylist ports that shouldn't
            be used when allocating a new port.
        listen_to_localhost_only (bool): Redis server only listens to
            localhost (127.0.0.1) if it's true,
            otherwise it listens to all network interfaces.

    Returns:
        A tuple of the port used by Redis and ProcessInfo for the process that
            was started. If a port is passed in, then the returned port value
            is the same.

    Raises:
        Exception: An exception is raised if Redis could not be started.
    """
    assert os.path.isfile(executable)
    counter = 0

    while counter < num_retries:
        # Construct the command to start the Redis server.
        command = [executable]
        if password:
            if " " in password:
                raise ValueError("Spaces not permitted in redis password.")
            command += ["--requirepass", password]
        command += (["--port", str(port), "--loglevel", "warning"])
        command += (["--dir", session_dir_path])
        if listen_to_localhost_only:
            command += ["--bind", "127.0.0.1"]
        elif bind_address is not None:
            command += ["--bind", "127.0.0.1", bind_address]

        pidfile = os.path.join(session_dir_path,
                               "redis-" + uuid.uuid4().hex + ".pid")
        command += ["--pidfile", pidfile]
        process_info = start_cloudtik_process(
            command,
            constants.PROCESS_TYPE_REDIS_SERVER,
            stdout_file=stdout_file,
            stderr_file=stderr_file,
            fate_share=fate_share)
        try:
            wait_for_redis_to_start("127.0.0.1", port, password=password)
        except (redis.exceptions.ResponseError, RuntimeError):
            # Connected to redis with the wrong password, or exceeded
            # the number of retries. This means we got the wrong redis
            # or there is some error in starting up redis.
            # Try the next port by looping again.
            pass
        else:
            r = redis.StrictRedis(
                host="127.0.0.1", port=port, password=password)
            # Check if Redis successfully started and we connected
            # to the right server.
            if r.config_get("pidfile")["pidfile"] == pidfile:
                break
        port = new_port(denylist=port_denylist)
        counter += 1
    if counter == num_retries:
        raise RuntimeError("Couldn't start Redis. "
                           "Check log files: {} {}".format(
                               stdout_file.name if stdout_file is not None else
                               "<stdout>", stderr_file.name
                               if stdout_file is not None else "<stderr>"))

    # Create a Redis client just for configuring Redis.
    redis_client = redis.StrictRedis(
        host="127.0.0.1", port=port, password=password)
    # Wait for the Redis server to start.
    wait_for_redis_to_start("127.0.0.1", port, password=password)
    # Configure Redis to generate keyspace notifications. TODO(rkn): Change
    # this to only generate notifications for the export keys.
    redis_client.config_set("notify-keyspace-events", "Kl")

    # Configure Redis to not run in protected mode so that processes on other
    # hosts can connect to it. TODO(rkn): Do this in a more secure way.
    redis_client.config_set("protected-mode", "no")

    # Discard old task and object metadata.
    if redis_max_memory is not None:
        redis_client.config_set("maxmemory", str(redis_max_memory))
        redis_client.config_set("maxmemory-policy", "allkeys-lru")
        redis_client.config_set("maxmemory-samples", "10")
        logger.debug("Starting Redis shard with {} GB max memory.".format(
            round(redis_max_memory / 1e9, 2)))

    # If redis_max_clients is provided, attempt to raise the number of maximum
    # number of Redis clients.
    if redis_max_clients is not None:
        redis_client.config_set("maxclients", str(redis_max_clients))
    elif resource is not None:
        # If redis_max_clients is not provided, determine the current ulimit.
        # We will use this to attempt to raise the maximum number of Redis
        # clients.
        current_max_clients = int(
            redis_client.config_get("maxclients")["maxclients"])
        # The below command should be the same as doing ulimit -n.
        ulimit_n = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        # The quantity redis_client_buffer appears to be the required buffer
        # between the maximum number of redis clients and ulimit -n. That is,
        # if ulimit -n returns 10000, then we can set maxclients to
        # 10000 - redis_client_buffer.
        redis_client_buffer = 32
        if current_max_clients < ulimit_n - redis_client_buffer:
            redis_client.config_set("maxclients",
                                    ulimit_n - redis_client_buffer)

    # Increase the hard and soft limits for the redis client pubsub buffer to
    # 128MB. This is a hack to make it less likely for pubsub messages to be
    # dropped and for pubsub connections to therefore be killed.
    cur_config = (redis_client.config_get("client-output-buffer-limit")[
        "client-output-buffer-limit"])
    cur_config_list = cur_config.split()
    assert len(cur_config_list) == 12
    cur_config_list[8:] = ["pubsub", "134217728", "134217728", "60"]
    redis_client.config_set("client-output-buffer-limit",
                            " ".join(cur_config_list))
    # Put a time stamp in Redis to indicate when it was started.
    redis_client.set("redis_start_time", time.time())
    return port, process_info


def start_log_monitor(redis_address,
                      logs_dir,
                      stdout_file=None,
                      stderr_file=None,
                      redis_password=None,
                      fate_share=None,
                      logging_level=None,
                      max_bytes=0,
                      backup_count=0):
    """Start a log monitor process.

    Args:
        redis_address (str): The address of the Redis instance.
        logs_dir (str): The directory of logging files.
        stdout_file: A file handle opened for writing to redirect stdout to. If
            no redirection should happen, then this should be None.
        stderr_file: A file handle opened for writing to redirect stderr to. If
            no redirection should happen, then this should be None.
        redis_password (str): The password of the redis server.
        logging_level (str): The logging level to use for the process.
        max_bytes (int): Log rotation parameter. Corresponding to
            RotatingFileHandler's maxBytes.
        backup_count (int): Log rotation parameter. Corresponding to
            RotatingFileHandler's backupCount.

    Returns:
        ProcessInfo for the process that was started.
    """
    log_monitor_filepath = os.path.join(CLOUDTIK_PATH, CLOUDTIK_CORE_PRIVATE_SERVICE,
                                        "cloudtik_log_monitor.py")
    command = [
        sys.executable, "-u", log_monitor_filepath,
        f"--redis-address={redis_address}", f"--logs-dir={logs_dir}",
        f"--logging-rotate-bytes={max_bytes}",
        f"--logging-rotate-backup-count={backup_count}",
    ]
    if logging_level:
        command.append("--logging-level=" + logging_level)
    if redis_password:
        command += ["--redis-password", redis_password]
    process_info = start_cloudtik_process(
        command,
        constants.PROCESS_TYPE_LOG_MONITOR,
        stdout_file=stdout_file,
        stderr_file=stderr_file,
        fate_share=fate_share)
    return process_info


def start_cluster_controller(redis_address,
                             logs_dir,
                             stdout_file=None,
                             stderr_file=None,
                             cluster_scaling_config=None,
                             redis_password=None,
                             fate_share=None,
                             logging_level=None,
                             max_bytes=0,
                             backup_count=0,
                             controller_ip=None):
    """Run a process to control the cluster.

    Args:
        redis_address (str): The address that the Redis server is listening on.
        logs_dir(str): The path to the log directory.
        stdout_file: A file handle opened for writing to redirect stdout to. If
            no redirection should happen, then this should be None.
        stderr_file: A file handle opened for writing to redirect stderr to. If
            no redirection should happen, then this should be None.
        cluster_scaling_config: path to autoscaling config file.
        redis_password (str): The password of the redis server.
        logging_level (str): The logging level to use for the process.
        max_bytes (int): Log rotation parameter. Corresponding to
            RotatingFileHandler's maxBytes.
        backup_count (int): Log rotation parameter. Corresponding to
            RotatingFileHandler's backupCount.
        controller_ip (str): IP address of the machine that the controller will be
            run on. Can be excluded, but required for scaler metrics.
    Returns:
        ProcessInfo for the process that was started.
    """
    controller_path = os.path.join(CLOUDTIK_PATH, CLOUDTIK_CORE_PRIVATE_SERVICE, "cloudtik_cluster_controller.py")
    command = [
        sys.executable,
        "-u",
        controller_path,
        f"--logs-dir={logs_dir}",
        f"--redis-address={redis_address}",
        f"--logging-rotate-bytes={max_bytes}",
        f"--logging-rotate-backup-count={backup_count}",
    ]
    if logging_level:
        command.append("--logging-level=" + logging_level)
    if cluster_scaling_config:
        command.append("--cluster-scaling-config=" + str(cluster_scaling_config))
    if redis_password:
        command.append("--redis-password=" + redis_password)
    if controller_ip:
        command.append("--controller-ip=" + controller_ip)
    process_info = start_cloudtik_process(
        command,
        constants.PROCESS_TYPE_CLUSTER_CONTROLLER,
        stdout_file=stdout_file,
        stderr_file=stderr_file,
        fate_share=fate_share)
    return process_info


def start_node_controller(head, redis_address,
                          logs_dir,
                          resource_spec,
                          stdout_file=None,
                          stderr_file=None,
                          redis_password=None,
                          fate_share=None,
                          logging_level=None,
                          max_bytes=0,
                          backup_count=0,
                          controller_ip=None,
                          runtimes=None):
    """Run a process to controller the other processes.

    Args:
        head (bool): Whether to run this on head or worker
        redis_address (str): The address that the Redis server is listening on.
        logs_dir(str): The path to the log directory.
        resource_spec (ResourceSpec): Resources for the node.
        stdout_file: A file handle opened for writing to redirect stdout to. If
            no redirection should happen, then this should be None.
        stderr_file: A file handle opened for writing to redirect stderr to. If
            no redirection should happen, then this should be None.
        redis_password (str): The password of the redis server.
        logging_level (str): The logging level to use for the process.
        max_bytes (int): Log rotation parameter. Corresponding to
            RotatingFileHandler's maxBytes.
        backup_count (int): Log rotation parameter. Corresponding to
            RotatingFileHandler's backupCount.
        controller_ip (str): IP address of the machine that the controller will be
            run on. Can be excluded, but required for scaler metrics.
        runtimes (str): List of runtimes to pass to node controller
    Returns:
        ProcessInfo for the process that was started.
    """
    controller_path = os.path.join(CLOUDTIK_PATH, CLOUDTIK_CORE_PRIVATE_SERVICE, "cloudtik_node_controller.py")
    command = [
        sys.executable,
        "-u",
        controller_path,
        f"--logs-dir={logs_dir}",
        f"--redis-address={redis_address}",
        f"--logging-rotate-bytes={max_bytes}",
        f"--logging-rotate-backup-count={backup_count}",
    ]
    if logging_level:
        command.append("--logging-level=" + logging_level)

    node_type = tags.NODE_KIND_HEAD if head else tags.NODE_KIND_WORKER
    command.append("--node-type=" + node_type)

    if redis_password:
        command.append("--redis-password=" + redis_password)
    if controller_ip:
        command.append("--controller-ip=" + controller_ip)

    assert resource_spec.resolved()
    static_resources = resource_spec.to_resource_dict()

    # Format the resource argument in a form like 'CPU,1.0,GPU,0,Custom,3'.
    resource_argument = ",".join(
        ["{},{}".format(*kv) for kv in static_resources.items()])

    command.append(f"--static_resource_list={resource_argument}")

    if runtimes and len(runtimes) > 0:
        command.append("--runtimes=" + quote(runtimes))

    process_info = start_cloudtik_process(
        command,
        constants.PROCESS_TYPE_NODE_CONTROLLER,
        stdout_file=stdout_file,
        stderr_file=stderr_file,
        fate_share=fate_share)
    return process_info
