import os
from typing import Any, Dict

from cloudtik.core._private.constants import CLOUDTIK_RUNTIME_ENV_NODE_IP
from cloudtik.core._private.runtime_utils import get_runtime_value, get_runtime_config_from_node
from cloudtik.core._private.service_discovery.runtime_services import get_service_discovery_runtime
from cloudtik.core._private.service_discovery.utils import get_canonical_service_name, \
    get_service_discovery_config, define_runtime_service_on_head_or_all
from cloudtik.core._private.utils import RUNTIME_CONFIG_KEY
from cloudtik.runtime.common.service_discovery.consul import get_service_dns_name

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["nginx", True, "NGINX", "node"],
]

NGINX_RUNTIME_CONFIG_KEY = "nginx"
NGINX_SERVICE_PORT_CONFIG_KEY = "port"

NGINX_HIGH_AVAILABILITY_CONFIG_KEY = "high_availability"
NGINX_APP_MODE_CONFIG_KEY = "app_mode"

NGINX_BACKEND_CONFIG_KEY = "backend"
NGINX_BACKEND_CONFIG_MODE_CONFIG_KEY = "config_mode"
NGINX_BACKEND_BALANCE_CONFIG_KEY = "balance"
NGINX_BACKEND_SERVICE_NAME_CONFIG_KEY = "service_name"
NGINX_BACKEND_SERVICE_TAG_CONFIG_KEY = "service_tag"
NGINX_BACKEND_SERVICE_CLUSTER_CONFIG_KEY = "service_cluster"
NGINX_BACKEND_SERVICE_PORT_CONFIG_KEY = "service_port"
NGINX_BACKEND_SERVERS_CONFIG_KEY = "servers"
NGINX_BACKEND_SELECTOR_CONFIG_KEY = "selector"

NGINX_SERVICE_NAME = "nginx"
NGINX_SERVICE_PORT_DEFAULT = 80

NGINX_APP_MODE_WEB = "web"
NGINX_APP_MODE_LOAD_BALANCER = "load-balancer"
# gateway currently not implemented
NGINX_APP_MODE_GATEWAY = "gateway"

NGINX_CONFIG_MODE_DNS = "dns"
NGINX_CONFIG_MODE_STATIC = "static"
# dynamic currently not implemented
NGINX_CONFIG_MODE_DYNAMIC = "dynamic"

NGINX_BACKEND_BALANCE_ROUND_ROBIN = "round_robin"
NGINX_BACKEND_BALANCE_LEAST_CONN = "least_conn"
NGINX_BACKEND_BALANCE_RANDOM = "random"
NGINX_BACKEND_BALANCE_IP_HASH = "ip_hash"
NGINX_BACKEND_BALANCE_HASH = "hash"


def _get_config(runtime_config: Dict[str, Any]):
    return runtime_config.get(NGINX_RUNTIME_CONFIG_KEY, {})


def _get_service_port(runtime_config: Dict[str, Any]):
    nginx_config = _get_config(runtime_config)
    return nginx_config.get(
        NGINX_SERVICE_PORT_CONFIG_KEY, NGINX_SERVICE_PORT_DEFAULT)


def _get_app_mode(nginx_config):
    return nginx_config.get(
        NGINX_APP_MODE_CONFIG_KEY, NGINX_APP_MODE_LOAD_BALANCER)


def _get_backend_config(nginx_config: Dict[str, Any]):
    return nginx_config.get(
        NGINX_BACKEND_CONFIG_KEY, {})


def _is_high_availability(nginx_config: Dict[str, Any]):
    return nginx_config.get(
        NGINX_HIGH_AVAILABILITY_CONFIG_KEY, False)


def _get_home_dir():
    return os.path.join(
        os.getenv("HOME"), "runtime", NGINX_RUNTIME_CONFIG_KEY)


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _get_runtime_endpoints(runtime_config: Dict[str, Any], cluster_head_ip):
    service_port = _get_service_port(runtime_config)
    endpoints = {
        "nginx": {
            "name": "NGINX",
            "url": "http://{}:{}".format(cluster_head_ip, service_port)
        },
    }
    return endpoints


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
    service_discovery_config = get_service_discovery_config(nginx_config)
    service_name = get_canonical_service_name(
        service_discovery_config, cluster_name, NGINX_SERVICE_NAME)
    service_port = _get_service_port(runtime_config)
    services = {
        service_name: define_runtime_service_on_head_or_all(
            service_discovery_config, service_port,
            _is_high_availability(nginx_config))
    }
    return services


def _validate_config(config: Dict[str, Any]):
    runtime_config = config.get(RUNTIME_CONFIG_KEY)
    nginx_config = _get_config(runtime_config)
    backend_config = _get_backend_config(nginx_config)

    app_mode = _get_app_mode(nginx_config)
    config_mode = backend_config.get(NGINX_BACKEND_CONFIG_MODE_CONFIG_KEY)
    if app_mode == NGINX_APP_MODE_LOAD_BALANCER:
        if config_mode == NGINX_CONFIG_MODE_STATIC:
            if not backend_config.get(
                    NGINX_BACKEND_SERVERS_CONFIG_KEY):
                raise ValueError("Static servers must be provided with config mode: static.")
        elif config_mode == NGINX_CONFIG_MODE_DNS:
            service_name = backend_config.get(NGINX_BACKEND_SERVICE_NAME_CONFIG_KEY)
            if not service_name:
                raise ValueError("Service name must be configured for config mode: dns.")


def _with_runtime_environment_variables(
        runtime_config, config):
    runtime_envs = {}

    nginx_config = _get_config(runtime_config)

    high_availability = _is_high_availability(nginx_config)
    if high_availability:
        runtime_envs["NGINX_HIGH_AVAILABILITY"] = high_availability

    runtime_envs["NGINX_LISTEN_PORT"] = _get_service_port(nginx_config)

    app_mode = _get_app_mode(nginx_config)
    runtime_envs["NGINX_APP_MODE"] = app_mode

    backend_config = _get_backend_config(nginx_config)
    if app_mode == NGINX_APP_MODE_WEB:
        _with_runtime_envs_for_web(
            config, backend_config, runtime_envs)
    elif app_mode == NGINX_APP_MODE_LOAD_BALANCER:
        _with_runtime_envs_for_load_balancer(
            config, backend_config, runtime_envs)
    else:
        raise ValueError("Invalid application mode: {}. "
                         "Must be web or load-balancer.".format(app_mode))

    balance = backend_config.get(
        NGINX_BACKEND_BALANCE_CONFIG_KEY)
    if balance:
        runtime_envs["NGINX_BACKEND_BALANCE"] = balance
    return runtime_envs


def _get_default_load_balancer_config_mode(config, backend_config):
    cluster_runtime_config = config.get(RUNTIME_CONFIG_KEY)
    if backend_config.get(
            NGINX_BACKEND_SERVERS_CONFIG_KEY):
        # if there are static servers configured
        config_mode = NGINX_CONFIG_MODE_STATIC
    elif get_service_discovery_runtime(cluster_runtime_config):
        # if there is service selector defined
        if backend_config.get(
                NGINX_BACKEND_SELECTOR_CONFIG_KEY):
            config_mode = NGINX_CONFIG_MODE_DYNAMIC
        else:
            config_mode = NGINX_CONFIG_MODE_DNS
    else:
        config_mode = NGINX_CONFIG_MODE_STATIC
    return config_mode


def _with_runtime_envs_for_web(
        config, backend_config, runtime_envs):
    pass


def _with_runtime_envs_for_load_balancer(
        config, backend_config, runtime_envs):
    config_mode = backend_config.get(
        NGINX_BACKEND_CONFIG_MODE_CONFIG_KEY)
    if not config_mode:
        config_mode = _get_default_load_balancer_config_mode(
            config, backend_config)

    if config_mode == NGINX_CONFIG_MODE_DNS:
        _with_runtime_envs_for_dns(backend_config, runtime_envs)
    elif config_mode == NGINX_CONFIG_MODE_STATIC:
        _with_runtime_envs_for_static(backend_config, runtime_envs)
    else:
        _with_runtime_envs_for_dynamic(backend_config, runtime_envs)
    runtime_envs["NGINX_CONFIG_MODE"] = config_mode


def _get_service_dns_name(backend_config):
    service_name = backend_config.get(NGINX_BACKEND_SERVICE_NAME_CONFIG_KEY)
    if not service_name:
        raise ValueError("Service name must be configured for config mode: dns.")

    service_tag = backend_config.get(
        NGINX_BACKEND_SERVICE_TAG_CONFIG_KEY)
    service_cluster = backend_config.get(
        NGINX_BACKEND_SERVICE_CLUSTER_CONFIG_KEY)

    return get_service_dns_name(
        service_name, service_tag, service_cluster)


def _with_runtime_envs_for_dns(backend_config, runtime_envs):
    service_dns_name = _get_service_dns_name(backend_config)
    runtime_envs["NGINX_BACKEND_SERVICE_DNS_NAME"] = service_dns_name

    service_port = backend_config.get(
        NGINX_BACKEND_SERVICE_PORT_CONFIG_KEY, NGINX_SERVICE_PORT_DEFAULT)
    runtime_envs["NGINX_BACKEND_SERVICE_PORT"] = service_port


def _with_runtime_envs_for_static(backend_config, runtime_envs):
    pass


def _with_runtime_envs_for_dynamic(backend_config, runtime_envs):
    pass


###################################
# Calls from node when configuring
###################################


def configure_backend(head):
    runtime_config = get_runtime_config_from_node(head)
    nginx_config = _get_config(runtime_config)

    app_mode = get_runtime_value("NGINX_APP_MODE")
    config_mode = get_runtime_value("NGINX_CONFIG_MODE")
    if app_mode == NGINX_APP_MODE_LOAD_BALANCER:
        if config_mode == NGINX_CONFIG_MODE_STATIC:
            _configure_static_backend(nginx_config)


def _configure_static_backend(nginx_config):
    backend_config = _get_backend_config(nginx_config)
    servers = backend_config.get(
        NGINX_BACKEND_SERVERS_CONFIG_KEY)

    balance_method = get_runtime_value("NGINX_BACKEND_BALANCE")
    bind_ip = get_runtime_value(CLOUDTIK_RUNTIME_ENV_NODE_IP)
    bind_port = get_runtime_value("NGINX_LISTEN_PORT")

    server_block = f"""
    server {{
        listen {bind_ip}:{bind_port};\n
        location / {{
            proxy_pass http://backend;
        }}
    }}
"""

    home_dir = _get_home_dir()
    config_file = os.path.join(
        home_dir, "conf", "nginx.conf")
    with open(config_file, "a") as f:
        # http block
        f.write("http {\n")

        # upstream block
        f.write("    upstream backend {\n")
        if balance_method and balance_method != NGINX_BACKEND_BALANCE_ROUND_ROBIN:
            f.write(f"    {balance_method};\n")
        for server in servers:
            server_line = f"server {server} max_fails=10 fail_timeout=30s slow_start=30s;\n"
            f.write(server_line)
        # end upstream block
        f.write("    }\n")
        # server block
        f.write(server_block)
        # end http block
        f.write("}\n")
