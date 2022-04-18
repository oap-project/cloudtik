from typing import Optional, List, Sequence, Tuple

import os

import re
import subprocess
import sys
import tarfile
import tempfile
import threading
import yaml

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from cloudtik.core._private.utils import get_head_working_ip, get_node_cluster_ip
from cloudtik.core.tags import CLOUDTIK_TAG_NODE_KIND, NODE_KIND_HEAD, \
    NODE_KIND_WORKER
from cloudtik.core._private.cli_logger import cli_logger
from cloudtik.core._private.providers import _get_node_provider

# Import psutil after cloudtik so the packaged version is used.
import psutil

from cloudtik.runtime.spark.utils import get_spark_runtime_logs

MAX_PARALLEL_SSH_WORKERS = 8
DEFAULT_SSH_USER = "ubuntu"
DEFAULT_SSH_KEYS = [
    "~/cloudtik_bootstrap_key.pem"
]


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
                 processes_list: Optional[List[Tuple[str, bool]]] = None):
        self.logs = logs
        self.debug_state = debug_state
        self.pip = pip
        self.processes = processes
        self.processes_verbose = processes_verbose
        self.processes_list = processes_list


class Node:
    """Node (as in "machine")"""

    def __init__(self,
                 host: str,
                 ssh_user: str = "ubuntu",
                 ssh_key: str = "~/cloudtik_bootstrap_key.pem",
                 docker_container: Optional[str] = None,
                 is_head: bool = False):
        self.host = host
        self.ssh_user = ssh_user
        self.ssh_key = ssh_key
        self.docker_container = docker_container
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
        exclude: Optional[Sequence[str]] = None) -> Archive:
    """Copy local log files into an archive.
        Args:
            archive (Archive): Archive object to add log files to.
            exclude (Sequence[str]): Sequence of regex patterns. Files that match
                any of these patterns will not be included in the archive.
        Returns:
            Open archive object.
    """
    get_cloudtik_local_logs(archive, exclude)
    get_runtime_local_logs(archive, exclude)


def get_cloudtik_local_logs(
        archive: Archive,
        exclude: Optional[Sequence[str]] = None,
        session_log_dir: str = "/tmp/cloudtik/session_latest") -> Archive:
    log_dir = os.path.join(session_log_dir, "logs")
    get_local_logs_for(archive, "cloudtik", log_dir, exclude)


def get_runtime_local_logs(
        archive: Archive,
        exclude: Optional[Sequence[str]] = None) -> Archive:
    runtime_logs = get_spark_runtime_logs()
    for category, log_dir in runtime_logs:
        get_local_logs_for(archive, category, log_dir, exclude)


def get_local_logs_for(
        archive: Archive,
        category:str,
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
                            processes: Optional[List[Tuple[str, bool]]] = None,
                            verbose: bool = False):
    """Get the status of all the relevant processes.
    Args:
        archive (Archive): Archive object to add process info files to.
        processes (list): List of processes to get information on. The first
            element of the tuple is a string to filter by, and the second
            element is a boolean indicating if we should filter by command
            name (True) or command line including parameters (False)
        verbose (bool): If True, show entire executable command line.
            If False, show just the first term.
    Returns:
        Open archive object.
    """
    if not processes:
        # local import to avoid circular dependencies
        # TO BE IMPROVED: for different runtime has different processes
        from cloudtik.core._private.constants import CLOUDTIK_PROCESSES
        processes = CLOUDTIK_PROCESSES

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
            get_local_logs(archive=archive)
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
                verbose=parameters.processes_verbose)
        except LocalCommandFailed as exc:
            cli_logger.error(exc)

    return archive


###
# Functions to invoke remote scripts and gather data from remote nodes
###


def _wrap(items: List[str], quotes="'"):
    return f"{quotes}{' '.join(items)}{quotes}"


def create_and_get_archive_from_remote_node(remote_node: Node,
                                            parameters: GetParameters,
                                            script_path: str = "cloudtik"
                                            ) -> Optional[str]:
    """Create an archive containing logs on a remote node and transfer.

    This will call ``cloudtik local-dump --stream`` on the remote
    node. The resulting file will be saved locally in a temporary file and
    returned.

    Args:
        remote_node (Node): Remote node to gather archive from.
        script_path (str): Path to this script on the remote node.
        parameters (GetParameters): Parameters (settings) for getting data.

    Returns:
        Path to a temporary file containing the node's collected data.

    """
    cmd = [
        "ssh",
        "-o StrictHostKeyChecking=no",
        "-o UserKnownHostsFile=/dev/null",
        "-o LogLevel=ERROR",
        "-i",
        remote_node.ssh_key,
        f"{remote_node.ssh_user}@{remote_node.host}",
    ]

    if remote_node.docker_container:
        cmd += [
            "docker",
            "exec",
            remote_node.docker_container,
        ]

    collect_cmd = [script_path, "local-dump", "--verbosity=0", "--stream"]
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
            if parameters.processes_verbose else ["--no-proccesses-verbose"]

    # Specify --login and -i here to source bashrc and avoid command not found issue
    cmd += ["/bin/bash", "--login", "-c", "-i", _wrap(collect_cmd, quotes="\"")]
    cmd += ["2>/dev/null"]

    cat = "node" if not remote_node.is_head else "head"

    cli_logger.verbose(f"Collecting data from remote node: {remote_node.host}")
    tmp = tempfile.mktemp(
        prefix=f"cloudtik_{cat}_{remote_node.host}_", suffix=".tar.gz")
    with open(tmp, "wb") as fp:
        try:
            subprocess.check_call(cmd, stdout=fp, stderr=sys.stderr)
        except subprocess.CalledProcessError as exc:
            raise RemoteCommandFailed(
                f"Gathering logs from remote node failed: {' '.join(cmd)}"
            ) from exc

    return tmp


def create_and_add_remote_data_to_local_archive(
        archive: Archive, remote_node: Node, parameters: GetParameters):
    """Create and get data from remote node and add to local archive.

    Args:
        archive (Archive): Archive object to add remote data to.
        remote_node (Node): Remote node to gather archive from.
        parameters (GetParameters): Parameters (settings) for getting data.

    Returns:
        Open archive object.
    """
    tmp = create_and_get_archive_from_remote_node(remote_node, parameters)

    if not archive.is_open:
        archive.open()

    cat = "node" if not remote_node.is_head else "head"

    with archive.subdir("", root=os.path.dirname(tmp)) as sd:
        sd.add(tmp, arcname=f"cloudtik_{cat}_{remote_node.host}.tar.gz")

    return archive


def create_and_add_local_data_to_local_archive(archive: Archive,
                                               parameters: GetParameters):
    """Create and get data from this node and add to archive.

    Args:
        archive (Archive): Archive object to add remote data to.
        parameters (GetParameters): Parameters (settings) for getting data.

    Returns:
        Open archive object.
    """
    with Archive() as local_data_archive:
        get_all_local_data(local_data_archive, parameters)

    if not archive.is_open:
        archive.open()

    with archive.subdir(
            "", root=os.path.dirname(local_data_archive.file)) as sd:
        sd.add(local_data_archive.file, arcname="local_node.tar.gz")

    os.remove(local_data_archive.file)

    return archive


def create_archive_for_remote_nodes(archive: Archive,
                                    remote_nodes: Sequence[Node],
                                    parameters: GetParameters):
    """Create an archive combining data from the remote nodes.

    This will parallelize calls to get data from remote nodes.

    Args:
        archive (Archive): Archive object to add remote data to.
        remote_nodes (Sequence[Node]): Sequence of remote nodes.
        parameters (GetParameters): Parameters (settings) for getting data.

    Returns:
        Open archive object.

    """
    if not archive.is_open:
        archive.open()

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_SSH_WORKERS) as executor:
        for remote_node in remote_nodes:
            executor.submit(
                create_and_add_remote_data_to_local_archive,
                archive=archive,
                remote_node=remote_node,
                parameters=parameters)

    return archive


def create_archive_for_local_and_remote_nodes(archive: Archive,
                                              remote_nodes: Sequence[Node],
                                              parameters: GetParameters):
    """Create an archive combining data from the local and remote nodes.

    This will parallelize calls to get data from remote nodes.

    Args:
        archive (Archive): Archive object to add data to.
        remote_nodes (Sequence[Node]): Sequence of remote nodes.
        parameters (GetParameters): Parameters (settings) for getting data.

    Returns:
        Open archive object.

    """
    if not archive.is_open:
        archive.open()

    try:
        create_and_add_local_data_to_local_archive(archive, parameters)
    except CommandFailed as exc:
        cli_logger.error(exc)

    create_archive_for_remote_nodes(archive, remote_nodes, parameters)

    cli_logger.print(f"Collected data from local node and {len(remote_nodes)} "
                     f"remote nodes.")
    return archive


def create_and_get_archive_from_head_node(head_node: Node,
                                          parameters: GetParameters,
                                          script_path: str = "cloudtik"
                                          ) -> Optional[str]:
    """Create an archive containing logs on a remote node and transfer.

    This will call ``cloudtik local-dump --stream`` on the remote
    node. The resulting file will be saved locally in a temporary file and
    returned.

    Args:
        head_node (Node): Remote node to gather archive from.
        script_path (str): Path to this script on the remote node.
        parameters (GetParameters): Parameters (settings) for getting data.

    Returns:
        Path to a temporary file containing the node's collected data.

    """
    cmd = [
        "ssh",
        "-o StrictHostKeyChecking=no",
        "-o UserKnownHostsFile=/dev/null",
        "-o LogLevel=ERROR",
        "-i",
        head_node.ssh_key,
        f"{head_node.ssh_user}@{head_node.host}",
    ]

    if head_node.docker_container:
        cmd += [
            "docker",
            "exec",
            head_node.docker_container,
        ]

    collect_cmd = [script_path, "head", "cluster-dump", "--verbosity=0", "--stream"]
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
            if parameters.processes_verbose else ["--no-proccesses-verbose"]

    # Specify --login and -i here to source bashrc and avoid command not found issue
    cmd += ["/bin/bash", "--login", "-c", "-i", _wrap(collect_cmd, quotes="\"")]
    cmd += ["2>/dev/null"]

    cat = "cluster"

    cli_logger.print(f"Collecting cluster data from head node: {head_node.host}")
    tmp = tempfile.mktemp(
        prefix=f"cloudtik_{cat}_{head_node.host}_", suffix=".tar.gz")
    with open(tmp, "wb") as fp:
        try:
            cli_logger.verbose("Running `{}`", " ".join(collect_cmd))
            cli_logger.verbose("Full command is `{}`", " ".join(cmd))
            subprocess.check_call(cmd, stdout=fp, stderr=sys.stderr)
        except subprocess.CalledProcessError as exc:
            raise RemoteCommandFailed(
                f"Gathering logs from head node failed: {' '.join(cmd)}"
            ) from exc

    return tmp


def create_and_add_workers_data_to_local_archive(
        archive: Archive, head_node: Node, parameters: GetParameters):
    """Create and get data from remote node and add to local archive.

    Args:
        archive (Archive): Archive object to add remote data to.
        head_node (Node): Remote node to gather archive from.
        parameters (GetParameters): Parameters (settings) for getting data.

    Returns:
        Open archive object.
    """
    tmp = create_and_get_archive_from_head_node(head_node, parameters)

    if not archive.is_open:
        archive.open()

    cat = "workers"

    with archive.subdir("", root=os.path.dirname(tmp)) as sd:
        sd.add(tmp, arcname=f"cloudtik_{cat}.tar.gz")

    return archive


def create_archive_for_cluster_nodes(archive: Archive,
                                     head_node: Node,
                                     parameters: GetParameters,
                                     head_only: bool = False):
    """Create an archive combining data from the remote nodes.

    This will parallelize calls to get data from remote nodes.

    Args:
        archive (Archive): Archive object to add remote data to.
        head_node (Node): The head node.
        parameters (GetParameters): Parameters (settings) for getting data.

    Returns:
        Open archive object.

    """
    if not archive.is_open:
        archive.open()

    # head node dump
    create_and_add_remote_data_to_local_archive(
        archive, head_node, parameters)

    if not head_only:
        # workers dump
        create_and_add_workers_data_to_local_archive(
            archive, head_node, parameters)

    return archive


def create_archive_for_local_and_cluster_nodes(archive: Archive,
                                               head_node: Node,
                                               parameters: GetParameters):
    """Create an archive combining data from the local and remote nodes.

    This will parallelize calls to get data from remote nodes.

    Args:
        archive (Archive): Archive object to add data to.
        head_node (Node): The head node.
        parameters (GetParameters): Parameters (settings) for getting data.

    Returns:
        Open archive object.

    """
    if not archive.is_open:
        archive.open()

    try:
        create_and_add_local_data_to_local_archive(archive, parameters)
    except CommandFailed as exc:
        cli_logger.error(exc)

    create_archive_for_cluster_nodes(archive, head_node, parameters)

    cli_logger.print(f"Collected cluster data from local node and cluster nodes")
    return archive


###
# cluster info
###
def get_info_from_cluster_config(
        cluster_config: str,
        should_bootstrap: bool
) -> Tuple[str, List[str], str, str, Optional[str], Optional[str]]:
    """Get information from cluster config.

    Return head ip, list of host IPs, ssh user, ssh key file, and optional docker
    container.

    Args:
        cluster_config (str): Path to cluster config.
        should_bootstrap (bool): Specify if we need to bootstrap the config
    Returns:
        Tuple of list of host IPs, ssh user name, ssh key file path,
            optional docker container name, optional cluster name.
    """
    from cloudtik.core._private.cluster.cluster_operator import _bootstrap_config

    cli_logger.verbose(f"Retrieving cluster information from cluster file: "
                       f"{cluster_config}")

    cluster_config = os.path.expanduser(cluster_config)

    config = yaml.safe_load(open(cluster_config).read())
    if should_bootstrap:
        config = _bootstrap_config(config, no_config_cache=True)

    provider = _get_node_provider(config["provider"], config["cluster_name"])
    head_nodes = provider.non_terminated_nodes({
        CLOUDTIK_TAG_NODE_KIND: NODE_KIND_HEAD
    })
    worker_nodes = provider.non_terminated_nodes({
        CLOUDTIK_TAG_NODE_KIND: NODE_KIND_WORKER
    })

    head_node = head_nodes[0] if len(head_nodes) > 0 else None
    # TODO haifeng: check which ip address to use here
    head_node_ip = get_head_working_ip(config, provider, head_node) if head_node else None
    workers = [get_node_cluster_ip(config, provider, node) for node in worker_nodes]
    ssh_user = config["auth"]["ssh_user"]
    ssh_key = config["auth"]["ssh_private_key"]

    docker = None
    docker_config = config.get("docker", None)
    if docker_config and docker_config.get("enabled", False):
        docker = docker_config.get("container_name", None)

    cluster_name = config.get("cluster_name", None)

    return head_node_ip, workers, ssh_user, ssh_key, docker, cluster_name


def _info_from_params(
        cluster: Optional[str] = None,
        host: Optional[str] = None,
        ssh_user: Optional[str] = None,
        ssh_key: Optional[str] = None,
        docker: Optional[str] = None,
        should_bootstrap: bool = True
):
    """Parse command line arguments.

    Note: This returns a list of hosts, not a comma separated string!
    """
    # TODO haifeng: check this condition if host list is specified for a cluster (running on head)
    if not host and not cluster:
        bootstrap_config = os.path.expanduser("~/cloudtik_bootstrap_config.yaml")
        if os.path.exists(bootstrap_config):
            cluster = bootstrap_config
            cli_logger.verbose(f"Detected cluster config file at {cluster}. "
                               f"If this is incorrect, specify with "
                               f"`cloudtik cluster-dump <config>`")
    elif cluster:
        cluster = os.path.expanduser(cluster)

    cluster_name = None
    head_node_ip = None
    if cluster:
        head_node_ip, h, u, k, d, cluster_name = get_info_from_cluster_config(cluster, should_bootstrap)

        ssh_user = ssh_user or u
        ssh_key = ssh_key or k
        docker = docker or d
        workers = host.split(",") if host else h
    elif host:
        workers = host.split(",")
    else:
        raise LocalCommandFailed(
            "You need to either specify a `<cluster_config>` or `--host`.")

    if not ssh_user:
        ssh_user = DEFAULT_SSH_USER
        cli_logger.warning(
            f"Using default SSH user `{ssh_user}`. "
            f"If this is incorrect, specify with `--ssh-user <user>`")

    if not ssh_key:
        for cand_key in DEFAULT_SSH_KEYS:
            cand_key_file = os.path.expanduser(cand_key)
            if os.path.exists(cand_key_file):
                ssh_key = cand_key_file
                cli_logger.warning(
                    f"Auto detected SSH key file: {ssh_key}. "
                    f"If this is incorrect, specify with `--ssh-key <key>`")
                break

    return cluster, head_node_ip, workers, ssh_user, ssh_key, docker, cluster_name
