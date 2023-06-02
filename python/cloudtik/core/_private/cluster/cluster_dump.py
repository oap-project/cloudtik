import copy
import os
import re
import subprocess
import tarfile
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from shlex import quote
from typing import Any, Dict, Optional, List, Sequence, Tuple

# Import psutil after cloudtik so the packaged version is used.
import psutil
import yaml

from cloudtik.core._private.call_context import CallContext
from cloudtik.core._private.cli_logger import cli_logger
from cloudtik.core._private.cluster.cluster_exec import exec_cluster, exec_on_head, rsync_cluster, rsync_on_head
from cloudtik.core._private.providers import _get_node_provider
from cloudtik.core._private.utils import get_head_working_ip, get_node_cluster_ip, get_runtime_logs, \
    get_runtime_processes, _get_node_specific_runtime_types
from cloudtik.core.tags import CLOUDTIK_TAG_NODE_KIND, NODE_KIND_HEAD, \
    NODE_KIND_WORKER

MAX_PARALLEL_SSH_WORKERS = 8


class CommandFailed(RuntimeError):
    pass


class LocalCommandFailed(CommandFailed):
    pass


class RemoteCommandFailed(CommandFailed):
    pass


class GetParameters:
    def __init__(self,
                 logs: bool = True,
                 debug_state: bool = True,
                 pip: bool = True,
                 processes: bool = True,
                 processes_verbose: bool = True,
                 processes_list: Optional[List[Tuple[str, bool, str, str]]] = None,
                 runtimes: List[str] = None):
        self.logs = logs
        self.debug_state = debug_state
        self.pip = pip
        self.processes = processes
        self.processes_verbose = processes_verbose
        self.processes_list = processes_list
        self.runtimes = runtimes

    def set_runtimes(self, runtimes):
        self.runtimes = runtimes


class Node:
    """Node (as in "machine")"""

    def __init__(self,
                 node_id: str,
                 host: str,
                 is_head: bool = False):
        self.node_id = node_id
        self.host = host
        self.is_head = is_head


class Archive:
    """Archive object to collect and compress files into a single file.

    Objects of this class can be passed around to different data collection
    functions. These functions can use the :meth:`subdir` method to add
    files to a sub directory of the archive.

    """

    def __init__(self, file: Optional[str] = None):
        self.file = file or tempfile.mktemp(
            prefix="cloudtik_logs_", suffix=".tar.gz")
        self.tar = None
        self._lock = threading.Lock()

    @property
    def is_open(self):
        return bool(self.tar)

    def open(self):
        self.tar = tarfile.open(self.file, "w:gz")

    def close(self):
        self.tar.close()
        self.tar = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @contextmanager
    def subdir(self, subdir: str, root: Optional[str] = "/"):
        """Open a context to add files to the archive.

        Example:

            .. code-block:: python

                with Archive("file.tar.gz") as archive:
                    with archive.subdir("logfiles", root="/tmp/logs") as sd:
                        # Will be added as `logfiles/nested/file.txt`
                        sd.add("/tmp/logs/nested/file.txt")

        Args:
            subdir (str): Subdir to which to add files to. Calling the
                ``add(path)`` command will place files into the ``subdir``
                directory of the archive.
            root (str): Root path. Files without an explicit ``arcname``
                will be named relatively to this path.

        Yields:
            A context object that can be used to add files to the archive.
        """
        root = os.path.abspath(root)

        class _Context:
            @staticmethod
            def add(path: str, arcname: Optional[str] = None):
                path = os.path.abspath(path)
                arcname = arcname or os.path.join(subdir,
                                                  os.path.relpath(path, root))

                self._lock.acquire()
                self.tar.add(path, arcname=arcname)
                self._lock.release()

        yield _Context()


###
# Functions to gather logs and information on the local node
###
def get_local_logs(
        archive: Archive,
        exclude: Optional[Sequence[str]] = None,
        runtimes: List[str] = None) -> Archive:
    """Copy local log files into an archive.
        Args:
            archive (Archive): Archive object to add log files to.
            exclude (Sequence[str]): Sequence of regex patterns. Files that match
                any of these patterns will not be included in the archive.
            runtimes: List of runtimes for collect logs from
        Returns:
            Open archive object.
    """
    get_cloudtik_local_logs(archive, exclude)
    get_runtime_local_logs(archive, exclude, runtimes=runtimes)
    return archive


def get_cloudtik_local_logs(
        archive: Archive,
        exclude: Optional[Sequence[str]] = None,
        session_log_dir: str = "/tmp/cloudtik/session_latest") -> Archive:
    log_dir = os.path.join(session_log_dir, "logs")
    get_local_logs_for(archive, "cloudtik", log_dir, exclude)
    return archive


def get_runtime_local_logs(
        archive: Archive,
        exclude: Optional[Sequence[str]] = None,
        runtimes: List[str] = None) -> Archive:
    runtime_logs = get_runtime_logs(runtimes)
    for category in runtime_logs:
        log_dir = runtime_logs[category]
        get_local_logs_for(archive, category, log_dir, exclude)
    return archive


def get_local_logs_for(
        archive: Archive,
        category: str,
        log_dir: str,
        exclude: Optional[Sequence[str]] = None) -> Archive:
    """Copy local log files into an archive.
    Returns:
        Open archive object.

    """
    if not os.path.isdir(log_dir):
        return archive

    if not archive.is_open:
        archive.open()

    exclude = exclude or []

    final_log_dir = os.path.expanduser(log_dir)

    with archive.subdir(category, root=final_log_dir) as sd:
        for root, dirs, files in os.walk(final_log_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, start=final_log_dir)
                # Skip file if it matches any pattern in `exclude`
                if any(re.match(pattern, rel_path) for pattern in exclude):
                    continue
                sd.add(file_path)

    return archive


def get_local_debug_state(archive: Archive,
                          session_dir: str = "/tmp/cloudtik/session_latest"
                          ) -> Archive:
    """Copy local log files into an archive.

    Args:
        archive (Archive): Archive object to add log files to.
        session_dir (str): Path to the session files. Defaults to
            ``/tmp/cloudtik/session_latest``

    Returns:
        Open archive object.

    """
    if not archive.is_open:
        archive.open()

    session_dir = os.path.expanduser(session_dir)
    debug_state_file = os.path.join(session_dir, "logs/debug_state.txt")

    if not os.path.exists(debug_state_file):
        return archive

    with archive.subdir("", root=session_dir) as sd:
        sd.add(debug_state_file)

    return archive


def get_local_pip_packages(archive: Archive):
    """Get currently installed pip packages and write into an archive.

    Args:
        archive (Archive): Archive object to add meta files to.

    Returns:
        Open archive object.
    """
    if not archive.is_open:
        archive.open()

    try:
        from pip._internal.operations import freeze
    except ImportError:  # pip < 10.0
        from pip.operations import freeze

    with tempfile.NamedTemporaryFile("wt") as fp:
        for line in freeze.freeze():
            fp.writelines([line, "\n"])

        fp.flush()
        with archive.subdir("") as sd:
            sd.add(fp.name, "pip_packages.txt")

    return archive


def get_local_processes(archive: Archive,
                        processes: Optional[List[Tuple[str, bool, str, str]]] = None,
                        verbose: bool = False,
                        runtimes: List[str] = None):
    """Get the status of all the relevant processes.
    Args:
        archive (Archive): Archive object to add process info files to.
        processes (list): List of processes to get information on. The first
            element of the tuple is a string to filter by, and the second
            element is a boolean indicating if we should filter by command
            name (True) or command line including parameters (False). The third
            element is meaningful name for the process. The fourth element is the
            node type on which the process should be (head, worker, node)
        verbose (bool): If True, show entire executable command line.
            If False, show just the first term.
        runtimes (List[str]): The list of runtimes
    Returns:
        Open archive object.
    """
    if not processes:
        # local import to avoid circular dependencies
        from cloudtik.core._private.constants import CLOUDTIK_PROCESSES
        processes = CLOUDTIK_PROCESSES
        processes.extend(get_runtime_processes(runtimes))

    process_infos = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "status"]):
        try:
            with process.oneshot():
                cmdline = " ".join(process.cmdline())
                process_infos.append(({
                    "executable": cmdline
                    if verbose else cmdline.split("--", 1)[0][:-1],
                    "name": process.name(),
                    "pid": process.pid,
                    "status": process.status(),
                }, process.cmdline()))
        except Exception as exc:
            raise LocalCommandFailed(exc) from exc

    relevant_processes = {}
    for process_dict, cmdline in process_infos:
        for keyword, filter_by_cmd, _, _ in processes:
            if filter_by_cmd:
                corpus = process_dict["name"]
            else:
                corpus = subprocess.list2cmdline(cmdline)
            if keyword in corpus and process_dict["pid"] \
               not in relevant_processes:
                relevant_processes[process_dict["pid"]] = process_dict

    with tempfile.NamedTemporaryFile("wt") as fp:
        for line in relevant_processes.values():
            fp.writelines([yaml.dump(line), "\n"])

        fp.flush()
        with archive.subdir("meta") as sd:
            sd.add(fp.name, "process_info.txt")

    return archive


def get_all_local_data(archive: Archive, parameters: GetParameters):
    """Get all local data.

    Gets:
        - The logs of the latest session
        - The currently installed pip packages

    Args:
        archive (Archive): Archive object to add meta files to.
        parameters (GetParameters): Parameters (settings) for getting data.

    Returns:
        Open archive object.
    """
    if not archive.is_open:
        archive.open()

    if parameters.logs:
        try:
            get_local_logs(archive=archive, runtimes=parameters.runtimes)
        except LocalCommandFailed as exc:
            cli_logger.error(exc)
    if parameters.debug_state:
        try:
            get_local_debug_state(archive=archive)
        except LocalCommandFailed as exc:
            cli_logger.error(exc)
    if parameters.pip:
        try:
            get_local_pip_packages(archive=archive)
        except LocalCommandFailed as exc:
            cli_logger.error(exc)
    if parameters.processes:
        try:
            get_local_processes(
                archive=archive,
                processes=parameters.processes_list,
                verbose=parameters.processes_verbose,
                runtimes=parameters.runtimes)
        except LocalCommandFailed as exc:
            cli_logger.error(exc)

    return archive


###
# Functions to invoke remote scripts and gather data from remote nodes
###


def _wrap(items: List[str], quotes="'"):
    return f"{quotes}{' '.join(items)}{quotes}"


def get_archive_from_remote_node(
        config: Dict[str, Any],
        call_context: CallContext,
        remote_node: Node,
        parameters: GetParameters
) -> Optional[str]:
    """Create an archive containing logs on a remote node and transfer.

    This will call ``cloudtik node dump --stream`` on the remote
    node. The resulting file will be saved locally in a temporary file and
    returned.

    Returns:
        Path to a temporary file containing the node's collected data.

    """
    collect_cmd = ["cloudtik", "node", "dump", "--verbosity=0"]
    collect_cmd += ["--logs"] if parameters.logs else ["--no-logs"]
    collect_cmd += ["--debug-state"] if parameters.debug_state else [
        "--no-debug-state"
    ]
    collect_cmd += ["--pip"] if parameters.pip else ["--no-pip"]
    collect_cmd += ["--processes"] if parameters.processes else [
        "--no-processes"
    ]
    if parameters.processes:
        collect_cmd += ["--processes-verbose"] \
            if parameters.processes_verbose else ["--no-processes-verbose"]

    if parameters.runtimes and len(parameters.runtimes) > 0:
        runtime_arg = ",".join(parameters.runtimes)
        collect_cmd += ["--runtimes={}".format(quote(runtime_arg))]

    kind = "worker" if not remote_node.is_head else "head"
    remote_temp_file = tempfile.mktemp(
        prefix=f"cloudtik_{kind}_{remote_node.host}_", suffix=".tar.gz")
    collect_cmd += ["--output"]
    collect_cmd += [remote_temp_file]

    cmd = " ".join(collect_cmd)

    cli_logger.print(f"Collecting data from remote node: {remote_node.host}")

    # execute the command on head, and it will dump to the output file
    exec_on_head(
        config, call_context,
        node_id=remote_node.node_id,
        cmd=cmd)

    # rsync the output file down to local temp
    local_temp_file = tempfile.mktemp(
        prefix=f"cloudtik_{kind}_{remote_node.host}_", suffix=".tar.gz")
    rsync_on_head(
        config, call_context, remote_node.node_id,
        remote_temp_file, local_temp_file, True)

    return local_temp_file


def add_archive_for_remote_node(
        config: Dict[str, Any],
        call_context: CallContext,
        archive: Archive,
        remote_node: Node,
        parameters: GetParameters):
    """Create and get data from remote node and add to local archive.

    Returns:
        Open archive object.
    """
    tmp = get_archive_from_remote_node(
        config, call_context,
        remote_node, parameters)

    if not archive.is_open:
        archive.open()

    kind = "worker" if not remote_node.is_head else "head"
    node_dir = f"{kind}_{remote_node.host}"

    add_archive_extracted(archive, node_dir, tmp)
    return archive


def add_archive_extracted(
        archive: Archive,
        subdir,
        file_to_add):
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(file_to_add, "r:gz") as source_tar:
            source_tar.extractall(path=tmpdir)
        with archive.subdir(subdir, root=tmpdir) as sd:
            for root, dirs, files in os.walk(tmpdir):
                for file in files:
                    file_path = os.path.join(root, file)
                    sd.add(file_path)
    return archive


def add_archive_for_local_node(archive: Archive,
                               local_node: Node,
                               parameters: GetParameters):
    """Create and get data from this node and add to archive.

    Args:
        archive (Archive): Archive object to add remote data to.
        local_node (Node): The local node to collect data
        parameters (GetParameters): Parameters (settings) for getting data.

    Returns:
        Open archive object.
    """
    with Archive() as local_data_archive:
        get_all_local_data(local_data_archive, parameters)

    if not archive.is_open:
        archive.open()

    kind = "worker" if not local_node.is_head else "head"
    node_dir = f"{kind}_{local_node.host}"

    tmp = local_data_archive.file
    add_archive_extracted(archive, node_dir, tmp)
    os.remove(tmp)
    return archive


def add_archive_for_remote_nodes(config: Dict[str, Any],
                                 call_context: CallContext,
                                 archive: Archive,
                                 remote_nodes: Sequence[Node],
                                 parameters: GetParameters):
    """Create an archive combining data from the remote nodes.

    This will parallelize calls to get data from remote nodes.

    Returns:
        Open archive object.

    """
    if not archive.is_open:
        archive.open()

    # This is to ensure that the parallel SSH calls below do not mess with
    # the users terminal.
    output_redir = call_context.is_output_redirected()
    call_context.set_output_redirected(True)
    allow_interactive = call_context.does_allow_interactive()
    call_context.set_allow_interactive(False)

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_SSH_WORKERS) as executor:
        for remote_node in remote_nodes:
            # get node type specific runtimes
            node_parameters = copy.deepcopy(parameters)
            node_runtimes = _get_node_specific_runtime_types(config, remote_node.node_id)
            node_parameters.set_runtimes(node_runtimes)
            executor.submit(
                add_archive_for_remote_node,
                config=config,
                call_context=call_context.new_call_context(),
                archive=archive,
                remote_node=remote_node,
                parameters=node_parameters)

    call_context.set_output_redirected(output_redir)
    call_context.set_allow_interactive(allow_interactive)

    return archive


def get_archive_from_head_node(
        config: Dict[str, Any],
        call_context: CallContext,
        nodes: Optional[List[Node]],
        parameters: GetParameters
) -> Optional[str]:
    """Create an archive containing logs on a remote node and transfer.

    This will call ``cloudtik node dump --stream`` on the remote
    node. The resulting file will be saved locally in a temporary file and
    returned.

    Returns:
        Path to a temporary file containing the node's collected data.

    """

    collect_cmd = ["cloudtik", "head", "cluster-dump", "--verbosity=0"]
    collect_cmd += ["--logs"] if parameters.logs else ["--no-logs"]
    collect_cmd += ["--debug-state"] if parameters.debug_state else [
        "--no-debug-state"
    ]
    collect_cmd += ["--pip"] if parameters.pip else ["--no-pip"]
    collect_cmd += ["--processes"] if parameters.processes else [
        "--no-processes"
    ]
    if parameters.processes:
        collect_cmd += ["--processes-verbose"] \
            if parameters.processes_verbose else ["--no-processes-verbose"]

    if nodes:
        # set hosts if worker list specified
        collect_cmd += ["--hosts"]
        collect_cmd += [",".join([node.host for node in nodes])]

    kind = "cluster"
    remote_temp_file = tempfile.mktemp(
        prefix=f"cloudtik_{kind}_", suffix=".tar.gz")
    collect_cmd += ["--output"]
    collect_cmd += [remote_temp_file]

    cmd = " ".join(collect_cmd)

    cli_logger.print(f"Collecting cluster data from head node...")

    # execute the command on head, and it will dump to the output file
    exec_cluster(
        config, call_context,
        cmd=cmd)

    # rsync the output file down to local temp
    local_temp_file = tempfile.mktemp(
        prefix=f"cloudtik_{kind}_", suffix=".tar.gz")
    rsync_cluster(
        config, call_context,
        remote_temp_file, local_temp_file, True)
    return local_temp_file


def add_archive_from_head(
        config: Dict[str, Any],
        call_context: CallContext,
        archive: Archive,
        nodes: Optional[List[Node]],
        parameters: GetParameters):
    """Create and get data from remote node and add to local archive.

    Returns:
        Open archive object.
    """
    tmp = get_archive_from_head_node(
        config, call_context,
        nodes, parameters)

    if not archive.is_open:
        archive.open()

    add_archive_extracted(archive, "", tmp)
    return archive


def add_archive_for_cluster_nodes(
        config: Dict[str, Any],
        call_context: CallContext,
        archive: Archive,
        head_node: Node,
        worker_nodes: Optional[List[Node]],
        parameters: GetParameters,
        head_only: bool = False):
    """Create an archive combining data from the remote nodes.

    This will parallelize calls to get data from remote nodes.

    Args:
        config: The config object.
        call_context: The call context.
        archive (Archive): Archive object to add remote data to.
        head_node (Node): Remote node to gather archive from.
        worker_nodes (List[Node]): List of worker nodes to dump
        parameters (GetParameters): Parameters (settings) for getting data.
        head_only: Dump the head node only

    Returns:
        Open archive object.

    """
    if not archive.is_open:
        archive.open()

    nodes = []

    if head_node:
        nodes += [head_node]

    if not head_only and worker_nodes:
        nodes += worker_nodes

    # collect data from head
    add_archive_from_head(
        config, call_context,
        archive, nodes, parameters)

    return archive


###
# get cluster nodes information
###
def _get_cluster_nodes(config: Dict[str, Any]) \
        -> Tuple[Tuple[str, str], List[Tuple[str, str]]]:
    """Get information from cluster config.
    Args:
        config (dict): The config object
    Returns:
        (head node_id, head node_ip), list of workers (node_id, node_ip)
    """
    provider = _get_node_provider(config["provider"], config["cluster_name"])
    head_nodes = provider.non_terminated_nodes({
        CLOUDTIK_TAG_NODE_KIND: NODE_KIND_HEAD
    })
    worker_nodes = provider.non_terminated_nodes({
        CLOUDTIK_TAG_NODE_KIND: NODE_KIND_WORKER
    })

    head_node = head_nodes[0] if len(head_nodes) > 0 else None
    # TODO: check which ip address to use here
    head_node_ip = get_head_working_ip(
        config, provider, head_node) if head_node else None
    workers = [(node, get_node_cluster_ip(
        provider, node)) for node in worker_nodes]

    return (head_node, head_node_ip), workers


def _get_nodes_to_dump(
        config: Dict[str, Any],
        hosts: Optional[str] = None
):
    """Parse command line arguments.

    Note: This returns a list of hosts, not a comma separated string!
    """
    head, workers = _get_cluster_nodes(config)

    if hosts:
        host_ips = hosts.split(",")
        target_workers = []
        target_head = None
        # build a set mapping node_ip to node information
        workers_set = {worker[1]: worker for worker in workers}
        for host_ip in host_ips:
            if host_ip in workers_set:
                target_workers.append(workers_set[host_ip])
            elif host_ip == head[1]:
                target_head = head
    else:
        target_workers = workers
        target_head = head

    return target_head, target_workers
