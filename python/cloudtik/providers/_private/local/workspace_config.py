import copy
import json
import logging
import os
import shutil
import socket
import subprocess
import tempfile
import uuid
from contextlib import closing
from functools import partial
from typing import Any, Optional
from typing import Dict

import psutil

from cloudtik.core._private.cli_logger import cli_logger, cf
from cloudtik.core._private.core_utils import kill_process_tree
from cloudtik.core.tags import CLOUDTIK_TAG_NODE_KIND, NODE_KIND_HEAD
from cloudtik.core.workspace_provider import Existence
from cloudtik.providers._private.local.config import get_cluster_name_from_node
from cloudtik.providers._private.local.local_container_scheduler import LocalContainerScheduler
from cloudtik.providers._private.local.local_host_scheduler import LocalHostScheduler

logger = logging.getLogger(__name__)

LOCAL_DOCKER_WORKSPACE_NUM_CREATION_STEPS = 2
LOCAL_DOCKER_WORKSPACE_NUM_DELETION_STEPS = 2
LOCAL_DOCKER_WORKSPACE_TARGET_RESOURCES = 2

DEFAULT_SSH_SERVER_PORT = 3371


def _get_expanded_path(path):
    return os.path.expanduser(path)


def _safe_remove_file(file_to_remove):
    if os.path.exists(file_to_remove):
        os.remove(file_to_remove)


def _get_network_name(workspace_name):
    return "cloudtik-{}".format(workspace_name)


def _get_bridge_interface_name(workspace_name):
    interface_suffix = str(uuid.uuid3(uuid.NAMESPACE_OID, workspace_name))[:8]
    return "tik-{}".format(interface_suffix)


def _get_sshd_config_file(workspace_name):
    return _get_expanded_path(
        "~/.ssh/cloudtik-{}-sshd_config".format(workspace_name))


def _get_ssh_control_key_file(workspace_name):
    return _get_expanded_path(
        "~/.ssh/cloudtik-{}-control_key".format(workspace_name))


def _get_authorized_keys_file(workspace_name):
    return _get_expanded_path(
        "~/.ssh/cloudtik-{}-authorized_keys".format(workspace_name))


def _get_host_key_file(workspace_name):
    return _get_expanded_path(
        "~/.ssh/cloudtik-{}-host_key".format(workspace_name))


def is_docker_workspace(provider_config: Dict[str, Any]) -> bool:
    return provider_config.get("docker", False)


def _create_local_scheduler(provider_config):
    if is_docker_workspace(provider_config):
        local_scheduler = LocalContainerScheduler(provider_config)
    else:
        local_scheduler = LocalHostScheduler(provider_config)
    return local_scheduler


def get_workspace_head_nodes(provider_config: Dict[str, Any]):
    tag_filters = {CLOUDTIK_TAG_NODE_KIND: NODE_KIND_HEAD}
    local_scheduler = _create_local_scheduler(provider_config)
    return local_scheduler.get_non_terminated_nodes(tag_filters)


def _get_node_info(provider_config: Dict[str, Any], node_id):
    local_scheduler = _create_local_scheduler(provider_config)
    return local_scheduler.get_node_info(node_id)


def _get_node_tags(provider_config: Dict[str, Any], node_id):
    local_scheduler = _create_local_scheduler(provider_config)
    return local_scheduler.get_node_tags(node_id)


def list_local_clusters(provider_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    head_nodes = get_workspace_head_nodes(provider_config)
    clusters = {}
    for head_node in head_nodes:
        node_info = _get_node_info(provider_config, head_node)
        cluster_name = get_cluster_name_from_node(node_info)
        if cluster_name:
            clusters[cluster_name] = node_info
    return clusters


def get_free_ssh_server_port(default_port):
    """ Get free port"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        test_port = default_port
        while True:
            result = s.connect_ex(('127.0.0.1', test_port))
            if result != 0:
                return test_port
            else:
                test_port += 1


def get_ssh_server_process_file(workspace_name: str):
    ssh_server_process_file = os.path.join(
        tempfile.gettempdir(), "cloudtik-{}-sshd".format(workspace_name))
    return ssh_server_process_file


def _get_ssh_server_process(ssh_server_process_file: str):
    if os.path.exists(ssh_server_process_file):
        process_info = json.loads(open(ssh_server_process_file).read())
        if process_info.get("process") and process_info["process"].get("pid"):
            ssh_server_process = process_info["process"]
            return (ssh_server_process["pid"],
                    ssh_server_process.get("bind_address"),
                    ssh_server_process["port"])
    return None, None, None


def _find_ssh_server_process_for_workspace(workspace_name):
    sshd_config = _get_sshd_config_file(workspace_name)
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            args = subprocess.list2cmdline(proc.cmdline())
            cmd_name = proc.name()
            if "sshd" in cmd_name and sshd_config in args:
                return proc.pid
        except psutil.Error:
            pass
    return None


def create_local_workspace(config):
    # create a copy of the input config to modify
    config = copy.deepcopy(config)

    # create workspace
    config = _create_workspace(config)
    return config


def _create_workspace(config):
    provider_config = config["provider"]
    if not is_docker_workspace(provider_config):
        return config

    workspace_name = config["workspace_name"]

    current_step = 1
    total_steps = LOCAL_DOCKER_WORKSPACE_NUM_CREATION_STEPS

    try:
        with cli_logger.group("Creating workspace: {}", workspace_name):
            with cli_logger.group(
                    "Creating docker bridge network",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _create_docker_bridge_network(config, workspace_name)

            with cli_logger.group(
                    "Creating bridge SSH server",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _create_bridge_ssh_server(config, workspace_name)
    except Exception as e:
        cli_logger.error("Failed to update workspace with the name {}. "
                         "You need to delete and try create again. {}", workspace_name, str(e))
        raise e

    cli_logger.success(
        "Successfully updated workspace: {}.",
        cf.bold(workspace_name))
    return config


def _create_docker_bridge_network(config, workspace_name):
    if _is_bridge_network_exists(workspace_name):
        cli_logger.print("Docker bridge network for the workspace already exists. Skip creation.")
        return

    network_name = _get_network_name(workspace_name)
    interface_name = _get_bridge_interface_name(workspace_name)
    # create a docker bridge network for workspace
    cmd = ("sudo docker network create -d bridge {network_name} "
           "-o \"com.docker.network.bridge.name\"=\"{interface_name}\"").format(
        network_name=network_name, interface_name=interface_name)
    try:
        cli_logger.print("Creating docker bridge network: {}.", network_name)
        subprocess.check_output(cmd, shell=True)
        cli_logger.print("Successfully created docker bridge network.")
    except subprocess.CalledProcessError as e:
        cli_logger.error("Failed to create bridge network: {}", str(e))
        raise e


def _create_bridge_ssh_server(config, workspace_name):
    current_step = 1
    total_steps = 2

    with cli_logger.group(
            "Preparing SSH server keys and configuration",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _prepare_ssh_server_keys_and_config(config, workspace_name)

    with cli_logger.group(
            "Starting bridge SSH server",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _start_bridge_ssh_server(config, workspace_name)


def _prepare_ssh_server_keys_and_config(config, workspace_name):
    # Create private key
    ssh_private_key_file = _get_ssh_control_key_file(workspace_name)
    authorized_keys_file = _get_authorized_keys_file(workspace_name)
    host_key_file = _get_host_key_file(workspace_name)
    if not os.path.exists(ssh_private_key_file):
        subprocess.check_output(
            f'ssh-keygen -b 2048 -t rsa -q -N "" '
            f"-f {ssh_private_key_file} "
            f"&& sudo chmod 600 {ssh_private_key_file}",
            shell=True,
        )
        cli_logger.print("Successfully created SSH control key.")
    if not os.path.exists(authorized_keys_file):
        subprocess.check_output(
            f"ssh-keygen -y "
            f"-f {ssh_private_key_file} "
            f"> {authorized_keys_file} "
            f"&& sudo chmod 600 {authorized_keys_file}",
            shell=True,
        )
        cli_logger.print("Successfully created SSH authorized keys.")
    if not os.path.exists(host_key_file):
        cli_logger.print("Creating SSH host key...")
        subprocess.check_output(
            f'ssh-keygen -b 2048 -t rsa -q -N "" '
            f"-f {host_key_file} "
            f"&& sudo chmod 600 {host_key_file}",
            shell=True,
        )
        cli_logger.print("Successfully created SSH host key.")

    sshd_config = _get_sshd_config_file(workspace_name)
    src_sshd_config = os.path.join(os.path.dirname(__file__), "sshd_config")
    shutil.copyfile(src_sshd_config, sshd_config)
    cli_logger.print("Successfully prepared SSH server configurations.")


def _start_bridge_ssh_server(config, workspace_name):
    network_name = _get_network_name(workspace_name)
    sshd_config = _get_sshd_config_file(workspace_name)
    authorized_keys_file = _get_authorized_keys_file(workspace_name)
    host_key_file = _get_host_key_file(workspace_name)

    bridge_address = _get_bridge_address(workspace_name)
    ssh_server_port = get_free_ssh_server_port(DEFAULT_SSH_SERVER_PORT)
    ssh_server_process_file = get_ssh_server_process_file(workspace_name)
    sshd_path = shutil.which("sshd")

    cmd = (f"{sshd_path} -f {sshd_config} "
           f"-h {host_key_file} "
           f"-p {ssh_server_port} -o ListenAddress={bridge_address} "
           f"-o AuthorizedKeysFile={authorized_keys_file}"
           )
    try:
        cli_logger.print("Starting bridge SSH server: {}.", network_name)

        p = subprocess.Popen(cmd, shell=True, stderr=subprocess.DEVNULL)
        if os.path.exists(ssh_server_process_file):
            process_info = json.loads(open(ssh_server_process_file).read())
        else:
            process_info = {}
        process_info["process"] = {"pid": p.pid, "bind_address": bridge_address, "port": ssh_server_port}
        with open(ssh_server_process_file, "w", opener=partial(os.open, mode=0o600)) as f:
            f.write(json.dumps(process_info))

        cli_logger.print("Successfully started bridge SSH server.")
    except subprocess.CalledProcessError as e:
        cli_logger.error("Failed to start bridge SSH server: {}", str(e))
        raise e


def _get_bridge_address(workspace_name):
    network_name = _get_network_name(workspace_name)
    cmd = (f"sudo docker network inspect {network_name} "
           "-f '{{ (index .IPAM.Config 0).Gateway }}'")
    bridge_address = subprocess.check_output(
        cmd,
        shell=True,
    ).decode().strip()
    return bridge_address


def _is_bridge_network_exists(workspace_name):
    network_name = _get_network_name(workspace_name)
    cmd = (f"sudo docker network ls --filter name={network_name} "
           "--format '{{.Name}}'")
    list_network_name = subprocess.check_output(
        cmd,
        shell=True,
    ).decode().strip()
    return True if list_network_name else False


def delete_local_workspace(
        config):
    provider_config = config["provider"]
    if not is_docker_workspace(provider_config):
        return config

    workspace_name = config["workspace_name"]

    current_step = 1
    total_steps = LOCAL_DOCKER_WORKSPACE_NUM_DELETION_STEPS
    try:
        with cli_logger.group("Deleting workspace: {}", workspace_name):
            with cli_logger.group(
                    "Deleting bridge SSH server",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _delete_bridge_ssh_server(config, workspace_name)

            with cli_logger.group(
                    "Deleting docker bridge network",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _delete_docker_bridge_network(config, workspace_name)
    except Exception as e:
        cli_logger.error(
            "Failed to delete workspace {}. {}", workspace_name, str(e))
        raise e

    cli_logger.success(
            "Successfully deleted workspace: {}.",
            cf.bold(workspace_name))


def _delete_docker_bridge_network(config, workspace_name):
    if not _is_bridge_network_exists(workspace_name):
        cli_logger.print("Docker bridge network for the workspace not found. Skip deletion.")
        return

    network_name = _get_network_name(workspace_name)
    # delete a docker bridge network for workspace
    cmd = "sudo docker network rm {network_name} ".format(network_name=network_name)
    try:
        cli_logger.print("Deleting docker bridge network: {}.", network_name)
        subprocess.check_output(cmd, shell=True)
        cli_logger.print("Successfully deleted docker bridge network.")
    except subprocess.CalledProcessError as e:
        cli_logger.error("Failed to delete docker bridge network: {}", str(e))
        raise e


def _delete_bridge_ssh_server(config, workspace_name):
    current_step = 1
    total_steps = 2

    with cli_logger.group(
            "Stopping bridge SSH server",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _stop_bridge_ssh_server(config, workspace_name)

    with cli_logger.group(
            "Deleting SSH server configurations",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_ssh_server_keys_and_config(config, workspace_name)


def _is_bridge_ssh_server_running(workspace_name):
    pid = _find_ssh_server_process_for_workspace(workspace_name)
    return False if pid is None else True


def _stop_bridge_ssh_server(config, workspace_name):
    network_name = _get_network_name(workspace_name)
    ssh_server_process_file = get_ssh_server_process_file(workspace_name)
    pid = _find_ssh_server_process_for_workspace(workspace_name)
    if pid is None:
        cli_logger.print("Bridge SSH server {} not started.", network_name)
        return

    try:
        cli_logger.print("Stopping bridge SSH server: {}.", network_name)
        kill_process_tree(pid)
        with open(ssh_server_process_file, "w", opener=partial(os.open, mode=0o600)) as f:
            f.write(json.dumps({"proxy": {}}))
        cli_logger.print("Successfully stopped bridge SSH server.")
    except subprocess.CalledProcessError as e:
        cli_logger.error("Failed to stop bridge SSH server: {}", str(e))
        raise e


def _delete_ssh_server_keys_and_config(config, workspace_name):
    # delete the private keys
    ssh_private_key_file = _get_ssh_control_key_file(workspace_name)
    authorized_keys_file = _get_authorized_keys_file(workspace_name)
    host_key_file = _get_host_key_file(workspace_name)
    _safe_remove_file(ssh_private_key_file)
    _safe_remove_file(authorized_keys_file)
    _safe_remove_file(host_key_file)
    cli_logger.print("Successfully deleted all the workspace SSH keys.")

    sshd_config = _get_sshd_config_file(workspace_name)
    _safe_remove_file(sshd_config)
    cli_logger.print("Successfully delete SSH server configurations.")


def check_local_workspace_integrity(config):
    existence = check_local_workspace_existence(config)
    return True if existence == Existence.COMPLETED else False


def check_local_workspace_existence(config):
    provider_config = config["provider"]
    if not is_docker_workspace(provider_config):
        return Existence.NOT_EXIST

    workspace_name = config["workspace_name"]

    skipped_resources = 0
    target_resources = LOCAL_DOCKER_WORKSPACE_TARGET_RESOURCES
    existing_resources = 0

    # check docker bridge network
    if _is_bridge_network_exists(workspace_name):
        existing_resources += 1

    if _is_bridge_ssh_server_running(workspace_name):
        existing_resources += 1

    if existing_resources <= skipped_resources:
        return Existence.NOT_EXIST
    elif existing_resources == target_resources:
        return Existence.COMPLETED
    else:
        return Existence.IN_COMPLETED
