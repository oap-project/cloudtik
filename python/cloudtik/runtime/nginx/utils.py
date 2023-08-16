import os
from shlex import quote
from typing import Any, Dict

from cloudtik.core._private.constants import CLOUDTIK_RUNTIME_ENV_CLUSTER
from cloudtik.core._private.core_utils import exec_with_call, exec_with_output, remove_files
from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_NGINX
from cloudtik.core._private.runtime_utils import get_runtime_value, get_runtime_config_from_node
from cloudtik.core._private.service_discovery.runtime_services import get_service_discovery_runtime
from cloudtik.core._private.service_discovery.utils import get_canonical_service_name, \
    get_service_discovery_config, define_runtime_service_on_head_or_all, exclude_runtime_of_cluster, \
    serialize_service_selector, SERVICE_DISCOVERY_PROTOCOL_HTTP
from cloudtik.core._private.utils import RUNTIME_CONFIG_KEY
from cloudtik.runtime.common.service_discovery.consul import get_service_dns_name, select_dns_service_tag

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["nginx", True, "NGINX", "node"],
]

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

NGINX_SERVICE_NAME = BUILT_IN_RUNTIME_NGINX
NGINX_SERVICE_PORT_DEFAULT = 80

NGINX_APP_MODE_WEB = "web"
NGINX_APP_MODE_LOAD_BALANCER = "load-balancer"
NGINX_APP_MODE_API_GATEWAY = "api-gateway"

NGINX_CONFIG_MODE_DNS = "dns"
NGINX_CONFIG_MODE_STATIC = "static"
NGINX_CONFIG_MODE_DYNAMIC = "dynamic"

NGINX_BACKEND_BALANCE_ROUND_ROBIN = "round_robin"
NGINX_BACKEND_BALANCE_LEAST_CONN = "least_conn"
NGINX_BACKEND_BALANCE_RANDOM = "random"
NGINX_BACKEND_BALANCE_IP_HASH = "ip_hash"
NGINX_BACKEND_BALANCE_HASH = "hash"

NGINX_DISCOVER_BACKEND_SERVERS_INTERVAL = 15

NGINX_LOAD_BALANCER_UPSTREAM_NAME = "backend"


def _get_config(runtime_config: Dict[str, Any]):
    return runtime_config.get(BUILT_IN_RUNTIME_NGINX, {})


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
        os.getenv("HOME"), "runtime", BUILT_IN_RUNTIME_NGINX)


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
            _is_high_availability(nginx_config),
            protocol=SERVICE_DISCOVERY_PROTOCOL_HTTP,
        )
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
    else:
        if config_mode and (
                config_mode != NGINX_CONFIG_MODE_DNS and
                config_mode != NGINX_CONFIG_MODE_DYNAMIC):
            raise ValueError("API Gateway mode support only DNS and dynamic config mode.")


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
    elif app_mode == NGINX_APP_MODE_API_GATEWAY:
        _with_runtime_envs_for_api_gateway(
            config, backend_config, runtime_envs)
    else:
        raise ValueError("Invalid application mode: {}. "
                         "Must be web, load-balancer or api-gateway.".format(app_mode))

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
        elif backend_config.get(
                NGINX_BACKEND_SERVICE_NAME_CONFIG_KEY):
            config_mode = NGINX_CONFIG_MODE_DNS
        else:
            config_mode = NGINX_CONFIG_MODE_DYNAMIC
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


def _get_default_api_gateway_config_mode(config, backend_config):
    cluster_runtime_config = config.get(RUNTIME_CONFIG_KEY)
    if not get_service_discovery_runtime(cluster_runtime_config):
        raise ValueError("Service discovery runtime is needed for API gateway mode.")

    if backend_config.get(
            NGINX_BACKEND_SELECTOR_CONFIG_KEY):
        config_mode = NGINX_CONFIG_MODE_DYNAMIC
    elif backend_config.get(
            NGINX_BACKEND_SERVICE_NAME_CONFIG_KEY):
        config_mode = NGINX_CONFIG_MODE_DNS
    else:
        config_mode = NGINX_CONFIG_MODE_DYNAMIC
    return config_mode


def _with_runtime_envs_for_api_gateway(
        config, backend_config, runtime_envs):
    config_mode = backend_config.get(NGINX_BACKEND_CONFIG_MODE_CONFIG_KEY)
    if not config_mode:
        config_mode = _get_default_api_gateway_config_mode(
            config, backend_config)

    runtime_envs["NGINX_CONFIG_MODE"] = config_mode


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


def _get_upstreams_config_dir():
    home_dir = _get_home_dir()
    return os.path.join(
        home_dir, "conf", "upstreams")


def _get_routers_config_dir():
    return os.path.join(
        _get_home_dir(), "conf", "routers")


def _configure_static_backend(nginx_config):
    backend_config = _get_backend_config(nginx_config)
    servers = backend_config.get(
        NGINX_BACKEND_SERVERS_CONFIG_KEY)
    balance_method = get_runtime_value("NGINX_BACKEND_BALANCE")
    _save_load_balancer_upstream(
        servers, balance_method)


def _save_load_balancer_upstream(servers, balance_method):
    upstreams_dir = _get_upstreams_config_dir()
    config_file = os.path.join(
        upstreams_dir, "load-balancer.conf")
    _save_upstream_config(
        config_file, NGINX_LOAD_BALANCER_UPSTREAM_NAME,
        servers, balance_method)


def _save_load_balancer_router():
    routers_dir = _get_routers_config_dir()
    config_file = os.path.join(
        routers_dir, "load-balancer.conf")
    _save_router_config(
        config_file, "", NGINX_LOAD_BALANCER_UPSTREAM_NAME)


def _save_upstream_config(
        upstream_config_file, backend_name,
        servers, balance_method):
    with open(upstream_config_file, "w") as f:
        # upstream block
        f.write("upstream " + backend_name + " {\n")
        if balance_method and balance_method != NGINX_BACKEND_BALANCE_ROUND_ROBIN:
            f.write(f"    {balance_method};\n")
        for server in servers:
            server_line = f"    server {server} max_fails=10 fail_timeout=30s;\n"
            f.write(server_line)
        # end upstream block
        f.write("}\n")


def _get_pull_identifier():
    return "{}-discovery".format(NGINX_SERVICE_NAME)


def start_pull_server(head):
    runtime_config = get_runtime_config_from_node(head)
    nginx_config = _get_config(runtime_config)

    app_mode = get_runtime_value("NGINX_APP_MODE")
    if app_mode == NGINX_APP_MODE_LOAD_BALANCER:
        discovery_class = "DiscoverBackendServers"
    else:
        config_mode = get_runtime_value("NGINX_CONFIG_MODE")
        if config_mode == NGINX_CONFIG_MODE_DNS:
            discovery_class = "DiscoverAPIGatewayBackends"
        else:
            discovery_class = "DiscoverAPIGatewayBackendServers"

    service_selector = nginx_config.get(
            NGINX_BACKEND_SELECTOR_CONFIG_KEY, {})
    cluster_name = get_runtime_value(CLOUDTIK_RUNTIME_ENV_CLUSTER)
    exclude_runtime_of_cluster(
        service_selector, BUILT_IN_RUNTIME_NGINX, cluster_name)

    service_selector_str = serialize_service_selector(service_selector)

    pull_identifier = _get_pull_identifier()

    cmd = ["cloudtik", "node", "pull", pull_identifier, "start"]
    cmd += ["--pull-class=cloudtik.runtime.nginx.discovery.{}".format(
        discovery_class)]
    cmd += ["--interval={}".format(
        NGINX_DISCOVER_BACKEND_SERVERS_INTERVAL)]
    # job parameters
    if service_selector_str:
        cmd += ["service_selector={}".format(service_selector_str)]

    balance_method = get_runtime_value("NGINX_BACKEND_BALANCE")
    if balance_method:
        cmd += ["balance_method={}".format(
            quote(balance_method))]

    cmd_str = " ".join(cmd)
    exec_with_output(cmd_str)


def stop_pull_server():
    pull_identifier = _get_pull_identifier()
    cmd = ["cloudtik", "node", "pull", pull_identifier, "stop"]
    cmd_str = " ".join(cmd)
    exec_with_output(cmd_str)


def update_load_balancer_configuration(
        backend_servers, balance_method):
    # write load balancer upstream config file
    servers = ["{}:{}".format(
        server_address[0], server_address[1]
    ) for _, server_address in backend_servers.items()]

    _save_load_balancer_upstream(servers, balance_method)
    _save_load_balancer_router()

    # the upstream config is changed, reload the service
    exec_with_call("sudo service nginx reload")


def update_api_gateway_dynamic_backends(
        api_gateway_backends, balance_method):
    # sort to make the order to the backends are always the same
    sorted_api_gateway_backends = sorted(api_gateway_backends.items())

    # write upstreams config
    _update_api_gateway_dynamic_upstreams(
        sorted_api_gateway_backends, balance_method)
    # write api-gateway config
    _update_api_gateway_dynamic_routers(
        sorted_api_gateway_backends)

    # Need reload nginx if there is new backend added
    exec_with_call("sudo service nginx reload")


def _update_api_gateway_dynamic_upstreams(
        sorted_api_gateway_backends, balance_method):
    upstreams_dir = _get_upstreams_config_dir()
    remove_files(upstreams_dir)

    for backend_name, backend_servers in sorted_api_gateway_backends:
        upstream_config_file = os.path.join(
            upstreams_dir, "{}.conf".format(backend_name))
        servers = ["{}:{}".format(
            server_address[0], server_address[1]
        ) for _, server_address in backend_servers.items()]
        _save_upstream_config(
            upstream_config_file, backend_name,
            servers, balance_method)


def _update_api_gateway_dynamic_routers(
        sorted_api_gateway_backends):
    routers_dir = _get_routers_config_dir()
    remove_files(routers_dir)

    for backend_name, backend_service in sorted_api_gateway_backends:
        router_file = os.path.join(
            routers_dir, "{}.conf".format(backend_name))
        _save_router_config(
            router_file, backend_name, backend_name)


def _save_router_config(router_file, location, backend_name):
    with open(router_file, "w") as f:
        # for each backend, we generate a location block
        f.write("location /" + location + " {\n")
        f.write(f"    proxy_pass http://{backend_name};\n")
        f.write("}\n")


def update_api_gateway_dns_backends(
        api_gateway_backends):
    routers_dir = _get_routers_config_dir()
    remove_files(routers_dir)

    # sort to make the order to the backends are always the same
    sorted_api_gateway_backends = sorted(api_gateway_backends.items())

    for backend_name, backend_service in sorted_api_gateway_backends:
        router_file = os.path.join(
            routers_dir, "{}.conf".format(backend_name))

        service_port = backend_service["service_port"]
        tags = backend_service.get("tags")
        service_tag = select_dns_service_tag(tags)
        service_dns_name = get_service_dns_name(
            backend_name, service_tag)

        variable_name = backend_name.replace('-', '_')
        with open(router_file, "w") as f:
            # for each backend, we generate a location block
            f.write("location /" + backend_name + " {\n")
            f.write(f"    set ${variable_name}_servers {service_dns_name};\n")
            f.write(f"    proxy_pass http://${variable_name}_servers:{service_port};\n")
            f.write("}\n")

    # Need reload nginx if there is new backend added
    exec_with_call("sudo service nginx reload")
