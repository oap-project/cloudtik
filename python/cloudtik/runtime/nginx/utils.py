from typing import Any, Dict

from cloudtik.core._private.service_discovery.utils import SERVICE_DISCOVERY_PROTOCOL, SERVICE_DISCOVERY_PORT, \
    SERVICE_DISCOVERY_NODE_KIND, SERVICE_DISCOVERY_PROTOCOL_TCP, SERVICE_DISCOVERY_NODE_KIND_WORKER, \
    get_canonical_service_name

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["nginx", True, "NGINX", "node"],
]

NGINX_RUNTIME_CONFIG_KEY = "nginx"
NGINX_SERVICE_PORT_CONFIG_KEY = "port"

NGINX_SERVICE_NAME = "nginx"
NGINX_SERVICE_PORT_DEFAULT = 80


def _get_config(runtime_config: Dict[str, Any]):
    return runtime_config.get(NGINX_RUNTIME_CONFIG_KEY, {})


def _get_service_port(runtime_config: Dict[str, Any]):
    nginx_config = _get_config(runtime_config)
    return nginx_config.get(
        NGINX_SERVICE_PORT_CONFIG_KEY, NGINX_SERVICE_PORT_DEFAULT)


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _get_head_service_urls(runtime_config: Dict[str, Any], cluster_head_ip):
    service_port = _get_service_port(runtime_config)
    services = {
        "nginx": {
            "name": "NGINX",
            "url": "http://{}:{}".format(cluster_head_ip, service_port)
        },
    }
    return services


def _get_head_service_ports(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    service_port = _get_service_port(runtime_config)
    service_ports = {
        "nginx": {
            "protocol": "TCP",
            "port": service_port,
        },
    }
    return service_ports


def _get_runtime_services(
        runtime_config: Dict[str, Any], cluster_name: str) -> Dict[str, Any]:
    nginx_config = _get_config(runtime_config)
    service_name = get_canonical_service_name(
        nginx_config, cluster_name, NGINX_SERVICE_NAME)
    service_port = _get_service_port(runtime_config)
    services = {
        service_name: {
            SERVICE_DISCOVERY_PROTOCOL: SERVICE_DISCOVERY_PROTOCOL_TCP,
            SERVICE_DISCOVERY_PORT: service_port,
        },
    }
    return services
