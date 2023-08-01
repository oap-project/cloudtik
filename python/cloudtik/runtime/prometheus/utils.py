from typing import Any, Dict

from cloudtik.core._private.service_discovery.utils import SERVICE_DISCOVERY_PROTOCOL, SERVICE_DISCOVERY_PORT, \
    SERVICE_DISCOVERY_PROTOCOL_TCP, get_canonical_service_name

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["prometheus", True, "Prometheus", "node"],
]

PROMETHEUS_RUNTIME_CONFIG_KEY = "prometheus"
PROMETHEUS_SERVICE_PORT_CONFIG_KEY = "port"

PROMETHEUS_SERVICE_NAME = "prometheus"
PROMETHEUS_SERVICE_PORT_DEFAULT = 9090


def _get_config(runtime_config: Dict[str, Any]):
    return runtime_config.get(PROMETHEUS_RUNTIME_CONFIG_KEY, {})


def _get_service_port(prometheus_config: Dict[str, Any]):
    return prometheus_config.get(
        PROMETHEUS_SERVICE_PORT_CONFIG_KEY, PROMETHEUS_SERVICE_PORT_DEFAULT)


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _with_runtime_environment_variables(
        runtime_config, config,
        provider, node_id: str):
    runtime_envs = {}

    prometheus_config = _get_config(runtime_config)
    service_port = _get_service_port(prometheus_config)
    runtime_envs["PROMETHEUS_SERVICE_PORT"] = service_port

    return runtime_envs


def _get_runtime_endpoints(runtime_config: Dict[str, Any], cluster_head_ip):
    prometheus_config = _get_config(runtime_config)
    service_port = _get_service_port(prometheus_config)
    endpoints = {
        "prometheus": {
            "name": "Prometheus",
            "url": "http://{}:{}".format(cluster_head_ip, service_port)
        },
    }
    return endpoints


def _get_head_service_ports(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    prometheus_config = _get_config(runtime_config)
    service_port = _get_service_port(prometheus_config)
    service_ports = {
        "prometheus": {
            "protocol": "TCP",
            "port": service_port,
        },
    }
    return service_ports


def _get_runtime_services(
        runtime_config: Dict[str, Any], cluster_name: str) -> Dict[str, Any]:
    prometheus_config = _get_config(runtime_config)
    service_name = get_canonical_service_name(
        prometheus_config, cluster_name, PROMETHEUS_SERVICE_NAME)
    service_port = _get_service_port(prometheus_config)
    services = {
        service_name: {
            SERVICE_DISCOVERY_PROTOCOL: SERVICE_DISCOVERY_PROTOCOL_TCP,
            SERVICE_DISCOVERY_PORT: service_port,
        },
    }
    return services
