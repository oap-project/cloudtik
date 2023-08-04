import os
from typing import Any, Dict

from cloudtik.core._private.service_discovery.utils import \
    get_canonical_service_name, define_runtime_service_on_head_or_all, get_service_discovery_config

RUNTIME_PROCESSES = [
        # The first element is the substring to filter.
        # The second element, if True, is to filter ps results by command name.
        # The third element is the process name.
        # The forth element, if node, the process should on all nodes,if head, the process should on head node.
        ["grafana", True, "Grafana", "node"],
    ]


GRAFANA_RUNTIME_CONFIG_KEY = "grafana"
GRAFANA_SERVICE_PORT_CONFIG_KEY = "port"
GRAFANA_HIGH_AVAILABILITY_CONFIG_KEY = "high_availability"

GRAFANA_SERVICE_NAME = "grafana"
GRAFANA_SERVICE_PORT_DEFAULT = 3000


def _get_config(runtime_config: Dict[str, Any]):
    return runtime_config.get(GRAFANA_RUNTIME_CONFIG_KEY, {})


def _get_service_port(grafana_config: Dict[str, Any]):
    return grafana_config.get(
        GRAFANA_SERVICE_PORT_CONFIG_KEY, GRAFANA_SERVICE_PORT_DEFAULT)


def _is_high_availability(grafana_config: Dict[str, Any]):
    return grafana_config.get(
        GRAFANA_HIGH_AVAILABILITY_CONFIG_KEY, False)


def _get_home_dir():
    return os.path.join(os.getenv("HOME"), "runtime", GRAFANA_SERVICE_NAME)


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _get_runtime_logs():
    home_dir = _get_home_dir()
    logs_dir = os.path.join(home_dir, "logs")
    return {"grafana": logs_dir}


def _with_runtime_environment_variables(
        runtime_config, config,
        provider, node_id: str):
    runtime_envs = {}

    grafana_config = _get_config(runtime_config)
    service_port = _get_service_port(grafana_config)
    runtime_envs["GRAFANA_SERVICE_PORT"] = service_port

    high_availability = _is_high_availability(grafana_config)
    if high_availability:
        runtime_envs["GRAFANA_HIGH_AVAILABILITY"] = high_availability

    return runtime_envs


def _get_runtime_endpoints(runtime_config: Dict[str, Any], cluster_head_ip):
    grafana_config = _get_config(runtime_config)
    service_port = _get_service_port(grafana_config)
    endpoints = {
        "grafana": {
            "name": "Grafana",
            "url": "http://{}:{}".format(cluster_head_ip, service_port)
        },
    }
    return endpoints


def _get_head_service_ports(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    grafana_config = _get_config(runtime_config)
    service_port = _get_service_port(grafana_config)
    service_ports = {
        "grafana": {
            "protocol": "TCP",
            "port": service_port,
        },
    }
    return service_ports


def _get_runtime_services(
        runtime_config: Dict[str, Any], cluster_name: str) -> Dict[str, Any]:
    grafana_config = _get_config(runtime_config)
    service_discovery_config = get_service_discovery_config(grafana_config)
    service_name = get_canonical_service_name(
        service_discovery_config, cluster_name, GRAFANA_SERVICE_NAME)
    service_port = _get_service_port(grafana_config)
    services = {
        service_name: define_runtime_service_on_head_or_all(
            service_discovery_config, service_port,
            _is_high_availability(grafana_config)),
    }
    return services
