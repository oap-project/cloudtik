from typing import Any, Dict

from cloudtik.core._private.core_utils import generate_public_key
from cloudtik.core._private.utils import get_config_for_update

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["cloudtik-ssh-server-sshd_config", False, "SSHServer", "node"],
]

SSH_SERVER_RUNTIME_CONFIG_KEY = "sshserver"
SSH_SERVER_DEFAULT_PORT = 22022
SSH_PUBLIC_KEY_ERROR = ("For use SSH Server runtime, "
                        "you must specify ssh public key of the cluster ssh private key "
                        "using ssh_public_key under auth configuration.")


def _get_ssh_server_port(runtime_config):
    ssh_server_config = runtime_config.get(
        SSH_SERVER_RUNTIME_CONFIG_KEY, {})
    return ssh_server_config.get("port", SSH_SERVER_DEFAULT_PORT)


def _bootstrap_runtime_config(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    ssh_public_key = cluster_config["auth"].get("ssh_public_key")
    if not ssh_public_key:
        ssh_private_key = cluster_config["auth"].get("ssh_private_key")
        if not ssh_private_key:
            raise ValueError(SSH_PUBLIC_KEY_ERROR)
        # if public key not specified and private key is specified
        # we can generate the public key file from private key
        try:
            ssh_public_key = generate_public_key(ssh_private_key)
        except Exception:
            raise ValueError("Failed to generate public key from private key file: {}. ".format(
                ssh_private_key) + SSH_PUBLIC_KEY_ERROR)

    file_mounts_config = get_config_for_update(cluster_config, "file_mounts")
    file_mounts_config.update({
        "~/.ssh/cloudtik-ssh-server-authorized_keys": ssh_public_key
    })
    return cluster_config


def _with_runtime_environment_variables(runtime_config, config):
    ssh_server_port = _get_ssh_server_port(runtime_config)
    runtime_envs = {
        "SSH_SERVER_PORT": ssh_server_port,
    }
    return runtime_envs


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _get_head_service_ports(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    ssh_server_port = _get_ssh_server_port(runtime_config)
    service_ports = {
        "sshd": {
            "protocol": "TCP",
            "port": ssh_server_port,
        },
    }
    return service_ports
