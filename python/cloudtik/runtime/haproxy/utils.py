from typing import Any, Dict

from cloudtik.core._private.service_discovery.utils import SERVICE_DISCOVERY_PROTOCOL, SERVICE_DISCOVERY_PORT, \
    SERVICE_DISCOVERY_PROTOCOL_TCP, get_canonical_service_name

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["haproxy", True, "HAProxy", "node"],
]

HAPROXY_RUNTIME_CONFIG_KEY = "haproxy"
HAPROXY_SERVICE_PORT_CONFIG_KEY = "port"

HAPROXY_SERVICE_NAME = "haproxy"
HAPROXY_SERVICE_PORT_DEFAULT = 80


def _get_config(runtime_config: Dict[str, Any]):
    return runtime_config.get(HAPROXY_RUNTIME_CONFIG_KEY, {})


def _get_service_port(runtime_config: Dict[str, Any]):
    haproxy_config = _get_config(runtime_config)
    return haproxy_config.get(
        HAPROXY_SERVICE_PORT_CONFIG_KEY, HAPROXY_SERVICE_PORT_DEFAULT)


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _get_head_service_urls(runtime_config: Dict[str, Any], cluster_head_ip):
    service_port = _get_service_port(runtime_config)
    services = {
        "haproxy": {
            "name": "HAProxy",
            "url": "http://{}:{}".format(cluster_head_ip, service_port)
        },
    }
    return services


def _get_head_service_ports(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    service_port = _get_service_port(runtime_config)
    service_ports = {
        "haproxy": {
            "protocol": "TCP",
            "port": service_port,
        },
    }
    return service_ports


def _get_runtime_services(
        runtime_config: Dict[str, Any], cluster_name: str) -> Dict[str, Any]:
    haproxy_config = _get_config(runtime_config)
    service_name = get_canonical_service_name(
        haproxy_config, cluster_name, HAPROXY_SERVICE_NAME)
    service_port = _get_service_port(runtime_config)
    services = {
        service_name: {
            SERVICE_DISCOVERY_PROTOCOL: SERVICE_DISCOVERY_PROTOCOL_TCP,
            SERVICE_DISCOVERY_PORT: service_port,
        },
    }
    return services
