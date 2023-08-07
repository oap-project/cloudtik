import os
from typing import Any, Dict

from cloudtik.core._private.runtime_utils import get_runtime_config_from_node, get_runtime_value
from cloudtik.core._private.service_discovery.runtime_services import get_service_discovery_runtime
from cloudtik.core._private.service_discovery.utils import get_canonical_service_name, define_runtime_service, \
    get_service_discovery_config
from cloudtik.core._private.utils import RUNTIME_CONFIG_KEY

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
HAPROXY_HIGH_AVAILABILITY_CONFIG_KEY = "high_availability"
HAPROXY_FRONTEND_MODE_CONFIG_KEY = "frontend_mode"


HAPROXY_BACKEND_CONFIG_KEY = "backend"
HAPROXY_BACKEND_CONFIG_MODE_CONFIG_KEY = "config_mode"
HAPROXY_BACKEND_MAX_SERVERS_CONFIG_KEY = "max_servers"
HAPROXY_BACKEND_SERVICE_NAME_CONFIG_KEY = "service_name"
HAPROXY_BACKEND_SERVICE_TAG_CONFIG_KEY = "service_tag"
HAPROXY_BACKEND_STATIC_SERVERS_CONFIG_KEY = "static.servers"
HAPROXY_BACKEND_DYNAMIC_SERVICE_CONFIG_KEY = "dynamic.service"


HAPROXY_SERVICE_NAME = "haproxy"
HAPROXY_SERVICE_PORT_DEFAULT = 80
HAPROXY_SERVICE_PROTOCOL_DEFAULT = "tcp"
HAPROXY_BACKEND_MAX_SERVERS_DEFAULT = 128

HAPROXY_FRONTEND_MODE_LOAD_BALANCER = "load-balancer"
HAPROXY_FRONTEND_MODE_GATEWAY = "gateway"

HAPROXY_CONFIG_MODE_DNS = "dns"
HAPROXY_CONFIG_MODE_STATIC = "static"
HAPROXY_CONFIG_MODE_DYNAMIC = "dynamic"


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


def _is_high_availability(haproxy_config: Dict[str, Any]):
    return haproxy_config.get(
        HAPROXY_HIGH_AVAILABILITY_CONFIG_KEY, False)


def _get_home_dir():
    return os.path.join(
        os.getenv("HOME"), "runtime", HAPROXY_RUNTIME_CONFIG_KEY)


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _validate_config(config: Dict[str, Any]):
    runtime_config = config.get(RUNTIME_CONFIG_KEY)
    haproxy_config = _get_config(runtime_config)

    config_mode = haproxy_config.get(HAPROXY_BACKEND_CONFIG_MODE_CONFIG_KEY)
    backend_config = _get_backend_config(haproxy_config)
    if config_mode == HAPROXY_CONFIG_MODE_STATIC:
        if not backend_config.get(
                HAPROXY_BACKEND_STATIC_SERVERS_CONFIG_KEY):
            raise ValueError("Static servers must be provided with config mode: static.")
    elif config_mode == HAPROXY_CONFIG_MODE_DNS:
        service_name = backend_config.get(HAPROXY_BACKEND_SERVICE_NAME_CONFIG_KEY)
        if not service_name:
            raise ValueError("Service name must be configured for config mode: dns.")


def _with_runtime_environment_variables(
        runtime_config, config):
    runtime_envs = {}

    haproxy_config = _get_config(runtime_config)
    cluster_runtime_config = config.get(RUNTIME_CONFIG_KEY)

    runtime_envs["HAPROXY_FRONTEND_PORT"] = _get_service_port(haproxy_config)
    runtime_envs["HAPROXY_FRONTEND_PROTOCOL"] = _get_service_protocol(haproxy_config)

    high_availability = _is_high_availability(haproxy_config)
    if high_availability:
        runtime_envs["HAPROXY_HIGH_AVAILABILITY"] = high_availability

    # Backend discovery support mode for:
    # 1. DNS (given static service name and optionally service tag)
    # 2. Static: a static list of servers
    # 3. Dynamic: a dynamic discovered service (services)
    config_mode = haproxy_config.get(HAPROXY_BACKEND_CONFIG_MODE_CONFIG_KEY)
    if not config_mode:
        backend_config = _get_backend_config(haproxy_config)
        if backend_config.get(
                HAPROXY_BACKEND_STATIC_SERVERS_CONFIG_KEY):
            # if there are static servers configured
            config_mode = HAPROXY_CONFIG_MODE_STATIC
        elif get_service_discovery_runtime(cluster_runtime_config):
            config_mode = HAPROXY_CONFIG_MODE_DNS
        else:
            config_mode = HAPROXY_CONFIG_MODE_STATIC

    if config_mode == HAPROXY_CONFIG_MODE_DNS:
        _with_runtime_envs_for_dns(haproxy_config, runtime_envs)
    elif config_mode == HAPROXY_CONFIG_MODE_STATIC:
        _with_runtime_envs_for_static(haproxy_config, runtime_envs)
    else:
        _with_runtime_envs_for_dynamic(haproxy_config, runtime_envs)

    runtime_envs["HAPROXY_CONFIG_MODE"] = config_mode
    return runtime_envs


def _with_runtime_envs_for_dns(haproxy_config, runtime_envs):
    backend_config = _get_backend_config(haproxy_config)
    service_name = backend_config.get(HAPROXY_BACKEND_SERVICE_NAME_CONFIG_KEY)
    if not service_name:
        raise ValueError("Service name must be configured for config mode: dns.")

    runtime_envs["HAPROXY_BACKEND_SERVICE_NAME"] = service_name
    runtime_envs["HAPROXY_BACKEND_MAX_SERVERS"] = backend_config.get(
        HAPROXY_BACKEND_MAX_SERVERS_CONFIG_KEY,
        HAPROXY_BACKEND_MAX_SERVERS_DEFAULT)

    service_tag = backend_config.get(
        HAPROXY_BACKEND_SERVICE_TAG_CONFIG_KEY)
    if service_tag:
        runtime_envs["HAPROXY_BACKEND_SERVICE_TAG"] = service_tag


def _with_runtime_envs_for_static(haproxy_config, runtime_envs):
    # TODO
    pass


def _with_runtime_envs_for_dynamic(haproxy_config, runtime_envs):
    # TODO
    pass


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


###################################
# Calls from node when configuring
###################################


def configure_backend(head):
    runtime_config = get_runtime_config_from_node(head)
    haproxy_config = _get_config(runtime_config)

    config_mode = get_runtime_value("HAPROXY_CONFIG_MODE")
    if config_mode == HAPROXY_CONFIG_MODE_STATIC:
        _configure_backend_static(haproxy_config)


def _configure_backend_static(haproxy_config):
    backend_config = _get_backend_config(haproxy_config)
    servers = backend_config.get(
        HAPROXY_BACKEND_STATIC_SERVERS_CONFIG_KEY)
    if servers:
        home_dir = _get_home_dir()
        config_file = os.path.join(
            home_dir, "conf", "haproxy.cfg")
        with open(config_file, "a") as f:
            for index, server in enumerate(servers, start=1):
                f.write("    server server{} {}\n".format(
                    index, server))
