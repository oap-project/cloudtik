import atexit
import collections
import datetime
import errno
import json
import logging
import os
import random
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

from typing import Optional, Dict
from collections import defaultdict

import cloudtik
import cloudtik.core._private.constants as constants
import cloudtik.core._private.services as services
from cloudtik.core._private.state.control_state import StateClient
from cloudtik.core._private import utils
from cloudtik.core._private import core_utils
from cloudtik.core._private.state import kv_store
from cloudtik.core._private.resource_spec import ResourceSpec

from cloudtik.core._private.core_utils import try_to_create_directory, try_to_symlink, open_log, \
    detect_fate_sharing_support, set_sigterm_handler

# Logger for this module.
logger = logging.getLogger(__name__)

SESSION_LATEST = "session_latest"
NUM_PORT_RETRIES = 40
NUM_REDIS_GET_RETRIES = 20


class NodeServicesStarter:
    """An encapsulation of the CloudTik processes on a single node.

    This class is responsible for starting service processes and killing them,
    and it also controls the temp file policy.

    Attributes:
        all_processes (dict): A mapping from process type (str) to a list of
            ProcessInfo objects. All lists have length one except for the Redis
            server list, which has multiple.
    """

    def __init__(self,
                 start_params,
                 head=False,
                 shutdown_at_exit=True,
                 spawn_reaper=True,
                 connect_only=False):
        """Start a node with all service started

        Args:
            start_params (StartParams): The parameters to use to
                configure the node.
            head (bool): True if this is the head node, which means it will
                start additional processes like the Redis servers, controller
                processes, and web UI.
            shutdown_at_exit (bool): If true, spawned processes will be cleaned
                up if this process exits normally.
            spawn_reaper (bool): If true, spawns a process that will clean up
                other spawned processes if this process dies unexpectedly.
            connect_only (bool): If true, connect to the node without starting
                new processes.
        """
        if shutdown_at_exit:
            if connect_only:
                raise ValueError("'shutdown_at_exit' and 'connect_only' "
                                 "cannot both be true.")
            self._register_shutdown_hooks()

        self.head = head
        self.kernel_fate_share = bool(
            spawn_reaper and detect_fate_sharing_support())
        self.all_processes = {}
        self.removal_lock = threading.Lock()

        # Try to get node IP address with the parameters.
        if start_params.node_ip_address:
            node_ip_address = start_params.node_ip_address
        elif start_params.redis_address:
            node_ip_address = services.get_node_ip_address(
                start_params.redis_address)
        else:
            node_ip_address = services.get_node_ip_address()
        self._node_ip_address = node_ip_address

        self._state_client = None

        start_params.update_if_absent(
            include_log_monitor=True,
            temp_dir=utils.get_cloudtik_temp_dir()
            )

        self._resource_spec = None
        self._localhost = socket.gethostbyname("localhost")
        self._start_params = start_params
        self._redis_address = start_params.redis_address

        # Configure log parameters.
        self.logging_level = os.getenv(constants.CLOUDTIK_LOGGING_LEVEL_ENV,
                                       constants.LOGGER_LEVEL_INFO)
        self.max_bytes = int(
            os.getenv(constants.CLOUDTIK_LOGGING_ROTATE_MAX_BYTES_ENV,
                      constants.LOGGING_ROTATE_MAX_BYTES))
        self.backup_count = int(
            os.getenv(constants.CLOUDTIK_LOGGING_ROTATE_BACKUP_COUNT_ENV,
                      constants.LOGGING_ROTATE_BACKUP_COUNT))

        assert self.max_bytes >= 0
        assert self.backup_count >= 0

        if head:
            start_params.update_if_absent(num_redis_shards=1)

        # Register the temp dir.
        if head:
            # date including microsecond
            date_str = datetime.datetime.today().strftime(
                "%Y-%m-%d_%H-%M-%S_%f")
            self.session_name = f"session_{date_str}_{os.getpid()}"
        else:
            session_name = self._kv_get_with_retry(
                "session_name", constants.KV_NAMESPACE_SESSION)
            self.session_name = core_utils.decode(session_name)
            # setup state client
            self.get_state_client()

        self._init_temp()

        self._metrics_export_port = self._get_cached_port(
            "metrics_export_port", default_port=start_params.metrics_export_port)

        start_params.update_if_absent(
            metrics_export_port=self._metrics_export_port)

        if not connect_only and spawn_reaper and not self.kernel_fate_share:
            self.start_reaper_process()
        if not connect_only:
            self._start_params.update_pre_selected_port()

        # Start processes.
        if head:
            self.start_head_processes()

        if not connect_only:
            self.start_node_processes()
            # we should update the address info after the node has been started
            try:
                services.wait_for_node(
                    self.redis_address, self._node_ip_address,
                    self.redis_password)
            except TimeoutError:
                raise Exception(
                    "The current node has not been updated within 30 "
                    "seconds, this could happen because of some of "
                    "the processes failed to startup.")

    def _register_shutdown_hooks(self):
        # Register the atexit handler. In this case, we shouldn't call sys.exit
        # as we're already in the exit procedure.
        def atexit_handler(*args):
            self.kill_all_processes(check_alive=False, allow_graceful=True)

        atexit.register(atexit_handler)

        # Register the handler to be called if we get a SIGTERM.
        # In this case, we want to exit with an error code (1) after
        # cleaning up child processes.
        def sigterm_handler(signum, frame):
            self.kill_all_processes(check_alive=False, allow_graceful=True)
            sys.exit(1)

        set_sigterm_handler(sigterm_handler)

    def _init_temp(self):
        # Create a dictionary to store temp file index.
        self._incremental_dict = collections.defaultdict(lambda: 0)

        if self.head:
            self._temp_dir = self._start_params.temp_dir
        else:
            temp_dir = self._kv_get_with_retry(
                "temp_dir", constants.KV_NAMESPACE_SESSION)
            self._temp_dir = core_utils.decode(temp_dir)

        try_to_create_directory(self._temp_dir)

        if self.head:
            self._session_dir = os.path.join(self._temp_dir, self.session_name)
        else:
            session_dir = self._kv_get_with_retry(
                "session_dir", constants.KV_NAMESPACE_SESSION)
            self._session_dir = core_utils.decode(session_dir)
        session_symlink = os.path.join(self._temp_dir, SESSION_LATEST)

        # Send a warning message if the session exists.
        try_to_create_directory(self._session_dir)
        try_to_symlink(session_symlink, self._session_dir)

        # Create a directory to be used for process log files.
        self._logs_dir = os.path.join(self._session_dir, "logs")
        try_to_create_directory(self._logs_dir)
        old_logs_dir = os.path.join(self._logs_dir, "old")
        try_to_create_directory(old_logs_dir)

        # Create a directory to be used for runtime environment.
        self._runtime_env_dir = os.path.join(
            self._session_dir, self._start_params.runtime_dir_name)
        try_to_create_directory(self._runtime_env_dir)

    def get_resource_spec(self):
        """Resolve and return the current resource spec for the node."""

        def merge_resources(env_dict, params_dict):
            """Separates special case params and merges two dictionaries, picking from the
            first in the event of a conflict. Also emit a warning on every
            conflict.
            """
            num_cpus = env_dict.pop("CPU", None)
            num_gpus = env_dict.pop("GPU", None)
            memory = env_dict.pop("memory", None)

            result = params_dict.copy()
            result.update(env_dict)

            for key in set(env_dict.keys()).intersection(
                    set(params_dict.keys())):
                if params_dict[key] != env_dict[key]:
                    logger.warning("Cluster Scaler is overriding your resource:"
                                   "{}: {} with {}.".format(
                                       key, params_dict[key], env_dict[key]))
            return num_cpus, num_gpus, memory,  result

        if not self._resource_spec:
            env_resources = {}
            env_string = os.getenv(
                constants.CLOUDTIK_RESOURCES_ENV)
            if env_string:
                try:
                    env_resources = json.loads(env_string)
                except Exception:
                    logger.exception("Failed to load {}".format(env_string))
                    raise
                logger.debug(
                    f"Cluster Scaler overriding resources: {env_resources}.")
            num_cpus, num_gpus, memory, resources = \
                merge_resources(env_resources, self._start_params.resources)
            self._resource_spec = ResourceSpec(
                self._start_params.num_cpus
                if num_cpus is None else num_cpus, self._start_params.num_gpus
                if num_gpus is None else num_gpus, self._start_params.memory
                if memory is None else memory,
                resources, self._start_params.redis_max_memory).resolve(
                    is_head=self.head)
        return self._resource_spec

    @property
    def node_ip_address(self):
        """Get the IP address of this node."""
        return self._node_ip_address

    @property
    def address(self):
        """Get the cluster address."""
        return self._redis_address

    @property
    def redis_address(self):
        """Get the cluster Redis address."""
        return self._redis_address

    @property
    def redis_password(self):
        """Get the cluster Redis password"""
        return self._start_params.redis_password

    @property
    def unique_id(self):
        """Get a unique identifier for this node."""
        # TODO (haifeng): what's the right for a unique_id?
        return f"{self.node_ip_address}"

    @property
    def metrics_export_port(self):
        """Get the port that exposes metrics"""
        return self._metrics_export_port

    @property
    def socket(self):
        """Get the socket reserving the node manager's port"""
        try:
            return self._socket
        except AttributeError:
            return None

    @property
    def logging_config(self):
        """Get the logging config of the current node."""
        return {
            "logging_level": self.logging_level,
            "logging_rotation_max_bytes": self.max_bytes,
            "logging_rotation_backup_count": self.backup_count
        }

    @property
    def address_info(self):
        """Get a dictionary of addresses."""
        return {
            "node_ip_address": self._node_ip_address,
            "redis_address": self._redis_address,
            "metrics_export_port": self._metrics_export_port,
            "address": self._redis_address
        }

    def is_head(self):
        return self.head

    def create_redis_client(self):
        """Create a redis client."""
        return services.create_redis_client(
            self._redis_address, self._start_params.redis_password)

    def get_state_client(self):
        if self._state_client is None:
            num_retries = NUM_REDIS_GET_RETRIES
            for i in range(num_retries):
                try:
                    redis_cli = self.create_redis_client()
                    self._state_client = StateClient(redis_cli)
                    break
                except Exception as e:
                    time.sleep(1)
                    logger.debug(f"Waiting for gcs up {e}")
            assert self._state_client is not None
            kv_store.kv_initialize(
                self._state_client)

        return self._state_client

    def get_temp_dir_path(self):
        """Get the path of the temporary directory."""
        return self._temp_dir

    def get_runtime_env_dir_path(self):
        """Get the path of the runtime env."""
        return self._runtime_env_dir

    def get_session_dir_path(self):
        """Get the path of the session directory."""
        return self._session_dir

    def get_logs_dir_path(self):
        """Get the path of the log files directory."""
        return self._logs_dir

    def _make_inc_temp(self, suffix="", prefix="", directory_name=None):
        """Return a incremental temporary file name. The file is not created.

        Args:
            suffix (str): The suffix of the temp file.
            prefix (str): The prefix of the temp file.
            directory_name (str) : The base directory of the temp file.

        Returns:
            A string of file name. If there existing a file having
                the same name, the returned name will look like
                "{directory_name}/{prefix}.{unique_index}{suffix}"
        """
        if directory_name is None:
            directory_name = utils.get_cloudtik_temp_dir()
        directory_name = os.path.expanduser(directory_name)
        index = self._incremental_dict[suffix, prefix, directory_name]
        # `tempfile.TMP_MAX` could be extremely large,
        # so using `range` in Python2.x should be avoided.
        while index < tempfile.TMP_MAX:
            if index == 0:
                filename = os.path.join(directory_name, prefix + suffix)
            else:
                filename = os.path.join(directory_name,
                                        prefix + "." + str(index) + suffix)
            index += 1
            if not os.path.exists(filename):
                # Save the index.
                self._incremental_dict[suffix, prefix, directory_name] = index
                return filename

        raise FileExistsError(errno.EEXIST,
                              "No usable temporary filename found")

    def get_log_file_handles(self, name, unique=False):
        """Open log files with partially randomized filenames, returning the
        file handles. If output redirection has been disabled, no files will
        be opened and `(None, None)` will be returned.

        Args:
            name (str): descriptive string for this log file.
            unique (bool): if true, a counter will be attached to `name` to
                ensure the returned filename is not already used.

        Returns:
            A tuple of two file handles for redirecting (stdout, stderr), or
            `(None, None)` if output redirection is disabled.
        """
        redirect_output = self._start_params.redirect_output

        if redirect_output is None:
            # Make the default behavior match that of glog.
            redirect_output = os.getenv("GLOG_logtostderr") != "1"

        if not redirect_output:
            return None, None

        log_stdout, log_stderr = self._get_log_file_names(name, unique=unique)
        return open_log(log_stdout), open_log(log_stderr)

    def _get_log_file_names(self, name, unique=False):
        """Generate partially randomized filenames for log files.

        Args:
            name (str): descriptive string for this log file.
            unique (bool): if true, a counter will be attached to `name` to
                ensure the returned filename is not already used.

        Returns:
            A tuple of two file names for redirecting (stdout, stderr).
        """

        if unique:
            log_stdout = self._make_inc_temp(
                suffix=".out", prefix=name, directory_name=self._logs_dir)
            log_stderr = self._make_inc_temp(
                suffix=".err", prefix=name, directory_name=self._logs_dir)
        else:
            log_stdout = os.path.join(self._logs_dir, f"{name}.out")
            log_stderr = os.path.join(self._logs_dir, f"{name}.err")
        return log_stdout, log_stderr

    def _get_unused_port(self, allocated_ports=None):
        if allocated_ports is None:
            allocated_ports = set()

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        port = s.getsockname()[1]

        # Try to generate a port that is far above the 'next available' one.
        # This solves issue #8254 where GRPC fails because the port assigned
        # from this method has been used by a different process.
        for _ in range(NUM_PORT_RETRIES):
            new_port = random.randint(port, 65535)
            if new_port in allocated_ports:
                # This port is allocated for other usage already,
                # so we shouldn't use it even if it's not in use right now.
                continue
            new_s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                new_s.bind(("", new_port))
            except OSError:
                new_s.close()
                continue
            s.close()
            new_s.close()
            return new_port
        logger.error("Unable to succeed in selecting a random port.")
        s.close()
        return port

    def _get_cached_port(self,
                         port_name: str,
                         default_port: Optional[int] = None) -> int:
        """Get a port number from a cache on this node.

        Different driver processes on a node should use the same ports for
        some purposes, e.g. exporting metrics.  This method returns a port
        number for the given port name and caches it in a file.  If the
        port isn't already cached, an unused port is generated and cached.

        Args:
            port_name (str): the name of the port, e.g. metrics_export_port
            default_port (Optional[int]): The port to return and cache if no
            port has already been cached for the given port_name.  If None, an
            unused port is generated and cached.
        Returns:
            port (int): the port number.
        """
        file_path = os.path.join(self.get_session_dir_path(),
                                 "ports_by_node.json")

        # Maps a Node.unique_id to a dict that maps port names to port numbers.
        ports_by_node: Dict[str, Dict[str, int]] = defaultdict(dict)

        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                json.dump({}, f)

        with open(file_path, "r") as f:
            ports_by_node.update(json.load(f))

        if (self.unique_id in ports_by_node
                and port_name in ports_by_node[self.unique_id]):
            # The port has already been cached at this node, so use it.
            port = int(ports_by_node[self.unique_id][port_name])
        else:
            # Pick a new port to use and cache it at this node.
            port = (default_port or self._get_unused_port(
                set(ports_by_node[self.unique_id].values())))
            ports_by_node[self.unique_id][port_name] = port
            with open(file_path, "w") as f:
                json.dump(ports_by_node, f)

        return port

    def start_reaper_process(self):
        """
        Start the reaper process.

        This must be the first process spawned and should only be called when
        cloudtik processes should be cleaned up if this process dies.
        """
        assert not self.kernel_fate_share, (
            "a reaper should not be used with kernel fate-sharing")
        process_info = services.start_reaper(fate_share=False)
        assert constants.PROCESS_TYPE_REAPER not in self.all_processes
        if process_info is not None:
            self.all_processes[constants.PROCESS_TYPE_REAPER] = [
                process_info,
            ]

    def start_redis(self):
        """Start the Redis servers."""
        assert self._redis_address is None
        redis_log_files = []
        if self._start_params.external_addresses is None:
            redis_log_files = [self.get_log_file_handles("redis", unique=True)]
            for i in range(self._start_params.num_redis_shards):
                redis_log_files.append(
                    self.get_log_file_handles(f"redis-shard_{i}", unique=True))

        (self._redis_address, redis_shards,
         process_infos) = services.start_redis(
             self._node_ip_address,
             redis_log_files,
             self.get_resource_spec(),
             self.get_session_dir_path(),
             port=self._start_params.redis_port,
             redis_shard_ports=self._start_params.redis_shard_ports,
             num_redis_shards=self._start_params.num_redis_shards,
             redis_max_clients=self._start_params.redis_max_clients,
             redirect_worker_output=True,
             password=self._start_params.redis_password,
             fate_share=self.kernel_fate_share,
             external_addresses=self._start_params.external_addresses,
             port_denylist=self._start_params.reserved_ports)
        assert (
            constants.PROCESS_TYPE_REDIS_SERVER not in self.all_processes)
        self.all_processes[constants.PROCESS_TYPE_REDIS_SERVER] = (
            process_infos)

    def start_cluster_controller(self):
        """Start the cluster controller.

        Controller output goes to these controller.err/out files, and
        any modification to these files may break existing
        cluster launching commands.
        """
        stdout_file, stderr_file = self.get_log_file_handles(
            "cloudtik_cluster_controller", unique=True)
        process_info = services.start_cluster_controller(
            self._redis_address,
            self._logs_dir,
            stdout_file=stdout_file,
            stderr_file=stderr_file,
            cluster_scaling_config=self._start_params.cluster_scaling_config,
            redis_password=self._start_params.redis_password,
            fate_share=self.kernel_fate_share,
            logging_level=self.logging_level,
            max_bytes=self.max_bytes,
            backup_count=self.backup_count,
            controller_ip=self._node_ip_address)
        assert constants.PROCESS_TYPE_CLUSTER_CONTROLLER not in self.all_processes
        self.all_processes[constants.PROCESS_TYPE_CLUSTER_CONTROLLER] = [process_info]

    def start_node_controller(self):
        """Start the node controller.

        Controller output goes to these controller.err/out files, and
        any modification to these files may break existing
        cluster launching commands.
        """
        stdout_file, stderr_file = self.get_log_file_handles(
            "cloudtik_node_controller", unique=True)
        process_info = services.start_node_controller(
            self.head,
            self._redis_address,
            self._logs_dir,
            self.get_resource_spec(),
            stdout_file=stdout_file,
            stderr_file=stderr_file,
            redis_password=self._start_params.redis_password,
            fate_share=self.kernel_fate_share,
            logging_level=self.logging_level,
            max_bytes=self.max_bytes,
            backup_count=self.backup_count,
            controller_ip=self._node_ip_address,
            runtimes=self._start_params.runtimes
        )
        assert constants.PROCESS_TYPE_NODE_CONTROLLER not in self.all_processes
        self.all_processes[constants.PROCESS_TYPE_NODE_CONTROLLER] = [process_info]

    def start_head_processes(self):
        """Start head processes on the node."""
        logger.debug(f"Process STDOUT and STDERR is being "
                     f"redirected to {self._logs_dir}.")
        assert self._redis_address is None
        # If this is the head node, start the relevant head node processes.
        self.start_redis()
        self._write_cluster_info_to_state()

        if not self._start_params.no_controller:
            self.start_cluster_controller()

    def start_node_processes(self):
        """Start all of the processes on the node."""
        logger.debug(f"Process STDOUT and STDERR is being "
                     f"redirected to {self._logs_dir}.")
        # TODO (haifeng): any service needs start on each worker node management
        self.start_node_controller()
        if self._start_params.include_log_monitor:
            self.start_log_monitor()

    def _write_cluster_info_to_state(self):
        # Make sure redis is up
        self.get_state_client().kv_put(
            b"session_name", self.session_name.encode(), True,
            constants.KV_NAMESPACE_SESSION)
        self.get_state_client().kv_put(
            b"session_dir", self._session_dir.encode(), True,
            constants.KV_NAMESPACE_SESSION)
        self.get_state_client().kv_put(
            b"temp_dir", self._temp_dir.encode(), True,
            constants.KV_NAMESPACE_SESSION)

    def _kill_process_type(self,
                           process_type,
                           allow_graceful=False,
                           check_alive=True,
                           wait=False):
        """Kill a process of a given type.

        If the process type is PROCESS_TYPE_REDIS_SERVER, then we will kill all
        of the Redis servers.

        If the process was started in valgrind, then we will raise an exception
        if the process has a non-zero exit code.

        Args:
            process_type: The type of the process to kill.
            allow_graceful (bool): Send a SIGTERM first and give the process
                time to exit gracefully. If that doesn't work, then use
                SIGKILL. We usually want to do this outside of tests.
            check_alive (bool): If true, then we expect the process to be alive
                and will raise an exception if the process is already dead.
            wait (bool): If true, then this method will not return until the
                process in question has exited.

        Raises:
            This process raises an exception in the following cases:
                1. The process had already died and check_alive is true.
                2. The process had been started in valgrind and had a non-zero
                   exit code.
        """

        # Ensure thread safety
        with self.removal_lock:
            self._kill_process_impl(
                process_type,
                allow_graceful=allow_graceful,
                check_alive=check_alive,
                wait=wait)

    def _kill_process_impl(self,
                           process_type,
                           allow_graceful=False,
                           check_alive=True,
                           wait=False):
        """See `_kill_process_type`."""
        if process_type not in self.all_processes:
            return
        process_infos = self.all_processes[process_type]
        if process_type != constants.PROCESS_TYPE_REDIS_SERVER:
            assert len(process_infos) == 1
        for process_info in process_infos:
            process = process_info.process
            # Handle the case where the process has already exited.
            if process.poll() is not None:
                if check_alive:
                    raise RuntimeError(
                        "Attempting to kill a process of type "
                        "'{}', but this process is already dead."
                        .format(process_type))
                else:
                    continue

            if process_info.use_valgrind:
                process.terminate()
                process.wait()
                if process.returncode != 0:
                    message = ("Valgrind detected some errors in process of "
                               "type {}. Error code {}.".format(
                                   process_type, process.returncode))
                    if process_info.stdout_file is not None:
                        with open(process_info.stdout_file, "r") as f:
                            message += "\nPROCESS STDOUT:\n" + f.read()
                    if process_info.stderr_file is not None:
                        with open(process_info.stderr_file, "r") as f:
                            message += "\nPROCESS STDERR:\n" + f.read()
                    raise RuntimeError(message)
                continue

            if process_info.use_valgrind_profiler:
                # Give process signal to write profiler data.
                os.kill(process.pid, signal.SIGINT)
                # Wait for profiling data to be written.
                time.sleep(0.1)

            if allow_graceful:
                process.terminate()
                # Allow the process one second to exit gracefully.
                timeout_seconds = 1
                try:
                    process.wait(timeout_seconds)
                except subprocess.TimeoutExpired:
                    pass

            # If the process did not exit, force kill it.
            if process.poll() is None:
                process.kill()
                # The reason we usually don't call process.wait() here is that
                # there's some chance we'd end up waiting a really long time.
                if wait:
                    process.wait()

        del self.all_processes[process_type]

    def kill_redis(self, check_alive=True):
        """Kill the Redis servers.

        Args:
            check_alive (bool): Raise an exception if any of the processes
                were already dead.
        """
        self._kill_process_type(
            constants.PROCESS_TYPE_REDIS_SERVER, check_alive=check_alive)

    def kill_controller(self, check_alive=True):
        """Kill the controller.

        Args:
            check_alive (bool): Raise an exception if the process was already
                dead.
        """
        self._kill_process_type(
            constants.PROCESS_TYPE_CLUSTER_CONTROLLER, check_alive=check_alive)

    def kill_reaper(self, check_alive=True):
        """Kill the reaper process.

        Args:
            check_alive (bool): Raise an exception if the process was already
                dead.
        """
        self._kill_process_type(
            constants.PROCESS_TYPE_REAPER, check_alive=check_alive)

    def kill_all_processes(self, check_alive=True, allow_graceful=False):
        """Kill all of the processes.

        Note that This is slower than necessary because it calls kill, wait,
        kill, wait, ... instead of kill, kill, ..., wait, wait, ...

        Args:
            check_alive (bool): Raise an exception if any of the processes were
                already dead.
        """

        # We call "list" to copy the keys because we are modifying the
        # dictionary while iterating over it.
        for process_type in list(self.all_processes.keys()):
            # Need to kill the reaper process last in case we die unexpectedly
            # while cleaning up.
            if process_type != constants.PROCESS_TYPE_REAPER:
                self._kill_process_type(
                    process_type,
                    check_alive=check_alive,
                    allow_graceful=allow_graceful)

        if constants.PROCESS_TYPE_REAPER in self.all_processes:
            self._kill_process_type(
                constants.PROCESS_TYPE_REAPER,
                check_alive=check_alive,
                allow_graceful=allow_graceful)

    def live_processes(self):
        """Return a list of the live processes.

        Returns:
            A list of the live processes.
        """
        result = []
        for process_type, process_infos in self.all_processes.items():
            for process_info in process_infos:
                if process_info.process.poll() is None:
                    result.append((process_type, process_info.process))
        return result

    def dead_processes(self):
        """Return a list of the dead processes.

        Note that this ignores processes that have been explicitly killed

        Returns:
            A list of the dead processes ignoring the ones that have been
                explicitly killed.
        """
        result = []
        for process_type, process_infos in self.all_processes.items():
            for process_info in process_infos:
                if process_info.process.poll() is not None:
                    result.append((process_type, process_info.process))
        return result

    def any_processes_alive(self):
        """Return true if any processes are still alive.

        Returns:
            True if any process is still alive.
        """
        return any(self.live_processes())

    def remaining_processes_alive(self):
        """Return true if all remaining processes are still alive.

        Note that this ignores processes that have been explicitly killed

        Returns:
            True if any process that wasn't explicitly killed is still alive.
        """
        return not any(self.dead_processes())

    def _kv_get_with_retry(self,
                                    key,
                                    namespace,
                                    num_retries=NUM_REDIS_GET_RETRIES):
        result = None
        if isinstance(key, str):
            key = key.encode()
        for i in range(num_retries):
            try:
                result = self.get_state_client().kv_get(key, namespace)
            except Exception as e:
                logger.error(f"ERROR as {e}")
                result = None

            if result is not None:
                break
            else:
                logger.debug(f"Fetched {key}=None from redis. Retrying.")
                time.sleep(2)
        if not result:
            raise RuntimeError(f"Could not read '{key}' from GCS (redis). "
                               "Has redis started correctly on the head node?")
        return result

    def start_log_monitor(self):
        """Start the log monitor."""
        process_info = cloudtik.core._private.services.start_log_monitor(
            self.redis_address,
            self._logs_dir,
            stdout_file=subprocess.DEVNULL,
            stderr_file=subprocess.DEVNULL,
            redis_password=self._start_params.redis_password,
            fate_share=self.kernel_fate_share,
            logging_level=self.logging_level,
            max_bytes=self.max_bytes,
            backup_count=self.backup_count)
        assert constants.PROCESS_TYPE_LOG_MONITOR not in self.all_processes
        self.all_processes[constants.PROCESS_TYPE_LOG_MONITOR] = [
            process_info,
        ]