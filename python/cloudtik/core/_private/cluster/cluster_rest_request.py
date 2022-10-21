from typing import Dict, Any

import sshtunnel
import urllib
import urllib.request
import urllib.error

from paramiko import ProxyCommand

from cloudtik.core._private.cluster.cluster_config import _load_cluster_config
from cloudtik.core._private.utils import get_cluster_head_ip


REST_ENDPOINT_URL_FORMAT = "http://{}:{}/{}"


def request_rest_on_head(
        cluster_config_file: str, cluster_name: str, endpoint: str, rest_api_port: int):
    config = _load_cluster_config(cluster_config_file, cluster_name)
    return _request_rest_on_head(
        config=config, endpoint=endpoint, rest_api_port=rest_api_port)


def _request_rest_on_head(
        config: Dict[str, Any], endpoint: str, rest_api_port: int):
    head_public_ip = get_cluster_head_ip(config, True)
    head_node_ip = get_cluster_head_ip(config, False)
    return request_rest_on_server(
        config=config, server_ip=head_public_ip,
        rest_api_ip=head_node_ip, rest_api_port=rest_api_port, endpoint=endpoint)


def ssh_proxy_wrapper(ssh_proxy_command, server_ip, ssh_port, ssh_user):
    if ssh_proxy_command is None:
        return None
    ssh_proxy_command = ssh_proxy_command.replace("%h", server_ip)
    ssh_proxy_command = ssh_proxy_command.replace("%p", str(ssh_port))
    ssh_proxy_command = ssh_proxy_command.replace("%r", str(ssh_user))
    ssh_proxy = ProxyCommand(ssh_proxy_command)
    return ssh_proxy


def request_rest_on_server(
        config, server_ip, rest_api_ip: str, rest_api_port: int, endpoint: str):
    auth_config = config.get("auth", {})
    ssh_proxy_command = auth_config.get("ssh_proxy_command", None)
    ssh_private_key = auth_config.get("ssh_private_key", None)
    ssh_user = auth_config["ssh_user"]
    ssh_port = auth_config.get("ssh_port", 22)
    ssh_proxy = ssh_proxy_wrapper(ssh_proxy_command, server_ip, ssh_port, ssh_user)

    with sshtunnel.open_tunnel(
            server_ip,
            ssh_username=ssh_user,
            ssh_port=ssh_port,
            ssh_pkey=ssh_private_key,
            ssh_proxy=ssh_proxy,
            remote_bind_address=(rest_api_ip, rest_api_port)
    ) as tunnel:
        endpoint_url = REST_ENDPOINT_URL_FORMAT.format(
            "127.0.0.1", tunnel.local_bind_port, endpoint)

        # sine we have use 127.0.0.1, disable all proxy on 127.0.0.1
        proxy_support = urllib.request.ProxyHandler({"no": "127.0.0.1"})
        opener = urllib.request.build_opener(proxy_support)
        urllib.request.install_opener(opener)

        response = urllib.request.urlopen(endpoint_url, timeout=10)
        return response.read()
