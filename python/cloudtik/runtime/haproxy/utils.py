from typing import Any, Dict

from cloudtik.core._private.service_discovery.utils import get_canonical_service_name, define_runtime_service, \
    get_service_discovery_config

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["haproxy", True, "HAProxy", "node"],
]

HAPROXY_RUNTIME_CONFIG_KEY = "haproxy"
HAPROXY_SERVICE_PORT_CONFIG_KEY = "port"
HAPROXY_SERVICE_PROTOCOL_CONFIG_KEY = "protocol"

HAPROXY_BACKEND_CONFIG_KEY = "backend"
HAPROXY_BACKEND_SERVICE_NAME_CONFIG_KEY = "service_name"
HAPROXY_BACKEND_SERVICE_TAG_CONFIG_KEY = "service_tag"
HAPROXY_BACKEND_SERVICE_MAX_SERVERS_CONFIG_KEY = "service_max_servers"

HAPROXY_SERVICE_NAME = "haproxy"
HAPROXY_SERVICE_PORT_DEFAULT = 80
HAPROXY_SERVICE_PROTOCOL_DEFAULT = "tcp"
HAPROXY_BACKEND_SERVICE_MAX_SERVERS_DEFAULT = 128

HAPROXY_CONFIG_MODE_CONSUL_DNS = "CONSUL-DNS"


def _get_config(runtime_config: Dict[str, Any]):
    return runtime_config.get(HAPROXY_RUNTIME_CONFIG_KEY, {})


def _get_service_port(haproxy_config: Dict[str, Any]):
    return haproxy_config.get(
        HAPROXY_SERVICE_PORT_CONFIG_KEY, HAPROXY_SERVICE_PORT_DEFAULT)


def _get_service_protocol(haproxy_config):
    return haproxy_config.get(
        HAPROXY_SERVICE_PROTOCOL_CONFIG_KEY, HAPROXY_SERVICE_PROTOCOL_DEFAULT)


def _get_backend_config(haproxy_config: Dict[str, Any]):
    return haproxy_config.get(
        HAPROXY_BACKEND_CONFIG_KEY, {})


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _with_runtime_environment_variables(
        runtime_config, config):
    runtime_envs = {}

    # export HAPROXY_LOAD_BALANCER_PORT, HAPROXY_LOAD_BALANCER_PROTOCOL
    # HAPROXY_LOAD_BALANCER_MAX_SERVER_NUMBER, HAPROXY_LOAD_BALANCER_SERVICE_NAME
    haproxy_config = _get_config(runtime_config)

    runtime_envs["HAPROXY_FRONTEND_PORT"] = _get_service_port(haproxy_config)
    runtime_envs["HAPROXY_FRONTEND_PROTOCOL"] = _get_service_protocol(haproxy_config)

    # TODO: to support more mode for:
    #  1. Consul with multiple static services or dynamic services
    #  2. other service discovery
    runtime_envs["HAPROXY_CONFIG_MODE"] = HAPROXY_CONFIG_MODE_CONSUL_DNS

    _with_runtime_envs_for_consul_dns(haproxy_config, runtime_envs)
    return runtime_envs


def _with_runtime_envs_for_consul_dns(haproxy_config, runtime_envs):
    backend_config = _get_backend_config(haproxy_config)
    service_name = backend_config.get(HAPROXY_BACKEND_SERVICE_NAME_CONFIG_KEY)
    if not service_name:
        raise ValueError("Backend service name is not configured for load balancer.")

    runtime_envs["HAPROXY_BACKEND_SERVICE_NAME"] = service_name
    runtime_envs["HAPROXY_BACKEND_SERVICE_MAX_SERVERS"] = backend_config.get(
        HAPROXY_BACKEND_SERVICE_MAX_SERVERS_CONFIG_KEY,
        HAPROXY_BACKEND_SERVICE_MAX_SERVERS_DEFAULT)

    service_tag = backend_config.get(
        HAPROXY_BACKEND_SERVICE_TAG_CONFIG_KEY)
    if service_tag:
        runtime_envs["HAPROXY_BACKEND_SERVICE_TAG"] = service_tag


def _get_runtime_endpoints(runtime_config: Dict[str, Any], cluster_head_ip):
    haproxy_config = _get_config(runtime_config)
    service_port = _get_service_port(haproxy_config)
    service_protocol = _get_service_protocol(haproxy_config)
    endpoints = {
        "haproxy": {
            "name": "HAProxy",
            "url": "{}://{}:{}".format(service_protocol, cluster_head_ip, service_port)
        },
    }
    return endpoints


def _get_head_service_ports(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    haproxy_config = _get_config(runtime_config)
    service_port = _get_service_port(haproxy_config)
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
    service_discovery_config = get_service_discovery_config(haproxy_config)
    service_name = get_canonical_service_name(
        service_discovery_config, cluster_name, HAPROXY_SERVICE_NAME)
    service_port = _get_service_port(haproxy_config)
    services = {
        service_name: define_runtime_service(
            service_discovery_config,
            service_port),
    }
    return services
