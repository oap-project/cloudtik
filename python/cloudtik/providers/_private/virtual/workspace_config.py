import copy
import json
import logging
import os
import shutil
import subprocess
from functools import partial
from typing import Any, Optional
from typing import Dict

from cloudtik.core._private.cli_logger import cli_logger, cf
from cloudtik.core._private.core_utils import kill_process_tree
from cloudtik.core._private.utils import exec_with_output, get_host_address, get_free_port
from cloudtik.core.tags import CLOUDTIK_TAG_NODE_KIND, NODE_KIND_HEAD
from cloudtik.core.workspace_provider import Existence
from cloudtik.providers._private.virtual.config import get_cluster_name_from_node, with_sudo, _safe_remove_file, \
    _get_network_name, _get_bridge_interface_name, _get_sshd_config_file, _get_ssh_control_key_file, \
    _get_authorized_keys_file, _get_host_key_file, _make_sure_data_path, \
    get_ssh_server_process_file, _find_ssh_server_process_for_workspace, DEFAULT_SSH_SERVER_PORT, _configure_docker
from cloudtik.providers._private.virtual.virtual_container_scheduler import VirtualContainerScheduler
from cloudtik.providers._private.virtual.utils import _get_node_info

logger = logging.getLogger(__name__)

VIRTUALWORKSPACE_NUM_CREATION_STEPS = 2
VIRTUALWORKSPACE_NUM_DELETION_STEPS = 2
VIRTUALWORKSPACE_TARGET_RESOURCES = 2


def _create_virtual_scheduler(provider_config):
    return VirtualContainerScheduler(provider_config, None)


def get_workspace_head_nodes(workspace_name, provider_config: Dict[str, Any]):
    tag_filters = {CLOUDTIK_TAG_NODE_KIND: NODE_KIND_HEAD}
    # The provider config is workspace provider
    # while scheduler expect cluster provider with bootstrap
    # we need to make sure of that
    virtual_scheduler = _create_virtual_scheduler(provider_config)
    return virtual_scheduler.list_nodes(workspace_name, tag_filters)


def list_virtual_clusters(
        workspace_name,
        provider_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    head_nodes = get_workspace_head_nodes(workspace_name, provider_config)
    clusters = {}
    for head_node in head_nodes:
        node_info = _get_node_info(head_node)
        cluster_name = get_cluster_name_from_node(node_info)
        if cluster_name:
            clusters[cluster_name] = node_info
    return clusters


def create_virtual_workspace(config):
    # create a copy of the input config to modify
    config = copy.deepcopy(config)

    # create workspace
    config = _create_workspace(config)
    return config


def _create_workspace(config):
    workspace_name = config["workspace_name"]

    current_step = 1
    total_steps = VIRTUALWORKSPACE_NUM_CREATION_STEPS

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
    interface_name = _get_bridge_interface_name(network_name)
    # create a docker bridge network for workspace
    docker_cmd = ("docker network create -d bridge {network_name} "
                  "-o \"com.docker.network.bridge.name\"=\"{interface_name}\"").format(
        network_name=network_name, interface_name=interface_name)
    cmd = with_sudo(docker_cmd)
    try:
        cli_logger.print("Creating docker bridge network: {}.", network_name)
        exec_with_output(cmd)
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
    # make sure the ~/.ssh folder exists and with the right permission
    ssh_path = os.path.expanduser("~/.ssh")
    if not os.path.exists(ssh_path):
        os.makedirs(ssh_path, mode=0o700, exist_ok=True)

    _make_sure_data_path()

    # Create private key
    ssh_private_key_file = _get_ssh_control_key_file(workspace_name)
    authorized_keys_file = _get_authorized_keys_file(workspace_name)
    host_key_file = _get_host_key_file(workspace_name)
    if not os.path.exists(ssh_private_key_file):
        exec_with_output(
            f'ssh-keygen -b 2048 -t rsa -q -N "" '
            f"-f {ssh_private_key_file} "
            f"&& chmod 600 {ssh_private_key_file}"
        )
        cli_logger.print("Successfully created SSH control key.")
    if not os.path.exists(authorized_keys_file):
        exec_with_output(
            f"ssh-keygen -y "
            f"-f {ssh_private_key_file} "
            f"> {authorized_keys_file} "
            f"&& chmod 600 {authorized_keys_file}"
        )
        cli_logger.print("Successfully created SSH authorized keys.")
    if not os.path.exists(host_key_file):
        exec_with_output(
            f'ssh-keygen -b 2048 -t rsa -q -N "" '
            f"-f {host_key_file} "
            f"&& chmod 600 {host_key_file}"
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
    ssh_server_port = get_free_port(bridge_address, DEFAULT_SSH_SERVER_PORT)
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
    docker_cmd = (f"docker network inspect {network_name} "
                  "-f '{{ (index .IPAM.Config 0).Gateway }}'")
    cmd = with_sudo(docker_cmd)
    bridge_address = exec_with_output(
        cmd
    ).decode().strip()

    # check whether the bridge address is appearing the IP list of host
    # for rootless docker, there is no interface created at the host
    private_addresses = get_host_address(address_type="private")
    if bridge_address not in private_addresses:
        # choose any private ip address
        if private_addresses:
            return sorted(private_addresses)[0]
        else:
            # use public IP
            public_addresses = get_host_address(address_type="public")
            if not public_addresses:
                raise RuntimeError("No proper ip address found for the host.")
            return sorted(public_addresses)[0]
    return bridge_address


def _is_bridge_network_exists(workspace_name):
    network_name = _get_network_name(workspace_name)
    docker_cmd = (f"docker network ls --filter name={network_name} "
                  "--format '{{.Name}}'")
    cmd = with_sudo(docker_cmd)
    list_network_name = exec_with_output(
        cmd
    ).decode().strip()
    return True if list_network_name else False


def delete_virtual_workspace(config):
    workspace_name = config["workspace_name"]

    current_step = 1
    total_steps = VIRTUALWORKSPACE_NUM_DELETION_STEPS
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
    docker_cmd = "docker network rm {network_name} ".format(network_name=network_name)
    cmd = with_sudo(docker_cmd)
    try:
        cli_logger.print("Deleting docker bridge network: {}.", network_name)
        exec_with_output(cmd)
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


def check_virtual_workspace_integrity(config):
    existence = check_virtual_workspace_existence(config)
    return True if existence == Existence.COMPLETED else False


def check_virtual_workspace_existence(config):
    workspace_name = config["workspace_name"]

    skipped_resources = 0
    target_resources = VIRTUALWORKSPACE_TARGET_RESOURCES
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


def update_virtual_workspace(
        config):
    workspace_name = config["workspace_name"]
    try:
        with cli_logger.group("Updating workspace: {}", workspace_name):
            update_docker_workspace(config, workspace_name)
    except Exception as e:
        cli_logger.error("Failed to update workspace with the name {}. "
                         "You need to delete and try create again. {}", workspace_name, str(e))
        raise e

    cli_logger.success(
        "Successfully updated workspace: {}.",
        cf.bold(workspace_name))


def update_docker_workspace(
        config, workspace_name):
    current_step = 1
    total_steps = 1

    with cli_logger.group(
            "Starting bridge SSH server",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        if _is_bridge_ssh_server_running(workspace_name):
            cli_logger.print(
                "Workspace bridge SSH server is already running. Skip update.")
        else:
            _start_bridge_ssh_server(config, workspace_name)


def bootstrap_virtual_workspace_config(config):
    config["provider"]["workspace_name"] = config["workspace_name"]

    # create a copy of the input config to modify
    config = copy.deepcopy(config)
    config = _configure_docker(config)
    return config

