import os
import shutil
from shlex import quote
from typing import Any, Dict

from cloudtik.core._private.constants import CLOUDTIK_RUNTIME_ENV_CLUSTER, CLOUDTIK_RUNTIME_ENV_NODE_IP
from cloudtik.core._private.core_utils import exec_with_output, exec_with_call
from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_HAPROXY
from cloudtik.core._private.runtime_utils import get_runtime_config_from_node, get_runtime_value
from cloudtik.core._private.service_discovery.runtime_services import get_service_discovery_runtime
from cloudtik.core._private.service_discovery.utils import get_canonical_service_name, \
    get_service_discovery_config, serialize_service_selector, define_runtime_service_on_head_or_all, \
    exclude_runtime_of_cluster
from cloudtik.core._private.utils import RUNTIME_CONFIG_KEY
from cloudtik.runtime.common.service_discovery.consul import get_rfc2782_service_dns_name
from cloudtik.runtime.haproxy.admin_api import get_backend_server_name

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["haproxy", True, "HAProxy", "node"],
]

HAPROXY_SERVICE_PORT_CONFIG_KEY = "port"
HAPROXY_SERVICE_PROTOCOL_CONFIG_KEY = "protocol"
HAPROXY_HIGH_AVAILABILITY_CONFIG_KEY = "high_availability"
HAPROXY_APP_MODE_CONFIG_KEY = "app_mode"


HAPROXY_BACKEND_CONFIG_KEY = "backend"
HAPROXY_BACKEND_CONFIG_MODE_CONFIG_KEY = "config_mode"
HAPROXY_BACKEND_BALANCE_CONFIG_KEY = "balance"
HAPROXY_BACKEND_MAX_SERVERS_CONFIG_KEY = "max_servers"
HAPROXY_BACKEND_SERVICE_NAME_CONFIG_KEY = "service_name"
HAPROXY_BACKEND_SERVICE_TAG_CONFIG_KEY = "service_tag"
HAPROXY_BACKEND_SERVICE_CLUSTER_CONFIG_KEY = "service_cluster"
HAPROXY_BACKEND_SERVERS_CONFIG_KEY = "servers"
HAPROXY_BACKEND_SELECTOR_CONFIG_KEY = "selector"
HAPROXY_BACKEND_SESSION_PERSISTENCE_CONFIG_KEY = "session_persistence"

HAPROXY_SERVICE_NAME = BUILT_IN_RUNTIME_HAPROXY
HAPROXY_SERVICE_PORT_DEFAULT = 80
HAPROXY_SERVICE_PROTOCOL_TCP = "tcp"
HAPROXY_SERVICE_PROTOCOL_HTTP = "http"
HAPROXY_BACKEND_MAX_SERVERS_DEFAULT = 32
HAPROXY_BACKEND_DYNAMIC_FREE_SLOTS = 8

HAPROXY_APP_MODE_LOAD_BALANCER = "load-balancer"
HAPROXY_APP_MODE_API_GATEWAY = "api-gateway"

HAPROXY_CONFIG_MODE_DNS = "dns"
HAPROXY_CONFIG_MODE_STATIC = "static"
HAPROXY_CONFIG_MODE_DYNAMIC = "dynamic"

"""
NOTE:
1. For using dynamic config mode which uses HAProxy Runtime API with add server command:
The backend must be configured to use a dynamic load-balancing algorithm for the balance directive:
roundrobin, leastconn, first, or random.
So for other balance options, need to use DNS config mode to scale.

"""
HAPROXY_BACKEND_BALANCE_ROUNDROBIN = "roundrobin"
HAPROXY_BACKEND_BALANCE_LEASTCONN = "leastconn"
HAPROXY_BACKEND_BALANCE_FIRST = "first"
HAPROXY_BACKEND_BALANCE_RANDOM = "random"
# Takes a regular sample expression in argument.
# The expression is evaluated for each request and hashed according to the configured hash-type.
# The result of the hash is divided by the total weight of the running servers
# to designate which server will receive the request.
# This can be used in place of "source", "uri", "hdr()", "url_param()", "rdp-cookie"
HAPROXY_BACKEND_BALANCE_HASH = "hash"
# The source IP address is hashed and divided by the total weight of
# the running servers to designate which server will receive the request.
HAPROXY_BACKEND_BALANCE_SOURCE = "source"
# This algorithm hashes either the left part of the URI or the whole URI
# and divides the hash value by the total weight of the running servers.
HAPROXY_BACKEND_BALANCE_URI = "uri"
# The URL parameter specified in argument will be looked up
# in the query string of each HTTP GET request.
HAPROXY_BACKEND_BALANCE_URL_PARAM = "url_param"
# hdr(<name>) The HTTP header <name> will be looked up in each HTTP request for hash
HAPROXY_BACKEND_BALANCE_HDR = "hdr"
# rdp-cookie(<name>)
HAPROXY_BACKEND_BALANCE_RDP_COOKIE = "rdp-cookie"

# Persistence based on an HTTP cookie. This option is only available with mode http
HAPROXY_BACKEND_SESSION_PERSISTENCE_COOKIE = "cookie"
# Persistence based on the client's IP address. This option is available with mode http and mode tcp
HAPROXY_BACKEND_SESSION_PERSISTENCE_IP = "ip"

HAPROXY_DISCOVER_BACKEND_SERVERS_INTERVAL = 15
HAPROXY_BACKEND_NAME_DEFAULT = "servers"
HAPROXY_BACKEND_SERVER_BASE_NAME = "server"


def get_default_server_name(server_id):
    return get_backend_server_name(
        HAPROXY_BACKEND_SERVER_BASE_NAME, server_id)


def _get_config(runtime_config: Dict[str, Any]):
    return runtime_config.get(BUILT_IN_RUNTIME_HAPROXY, {})


def _get_service_port(haproxy_config: Dict[str, Any]):
    return haproxy_config.get(
        HAPROXY_SERVICE_PORT_CONFIG_KEY, HAPROXY_SERVICE_PORT_DEFAULT)


def _get_service_protocol(haproxy_config):
    app_mode = _get_app_mode(haproxy_config)
    if app_mode == HAPROXY_APP_MODE_LOAD_BALANCER:
        default_protocol = HAPROXY_SERVICE_PROTOCOL_TCP
    else:
        default_protocol = HAPROXY_SERVICE_PROTOCOL_HTTP
    return haproxy_config.get(
        HAPROXY_SERVICE_PROTOCOL_CONFIG_KEY, default_protocol)


def _get_app_mode(haproxy_config):
    return haproxy_config.get(
        HAPROXY_APP_MODE_CONFIG_KEY, HAPROXY_APP_MODE_LOAD_BALANCER)


def _get_backend_config(haproxy_config: Dict[str, Any]):
    return haproxy_config.get(
        HAPROXY_BACKEND_CONFIG_KEY, {})


def _is_high_availability(haproxy_config: Dict[str, Any]):
    return haproxy_config.get(
        HAPROXY_HIGH_AVAILABILITY_CONFIG_KEY, False)


def _get_home_dir():
    return os.path.join(
        os.getenv("HOME"), "runtime", BUILT_IN_RUNTIME_HAPROXY)


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _validate_config(config: Dict[str, Any]):
    runtime_config = config.get(RUNTIME_CONFIG_KEY)
    haproxy_config = _get_config(runtime_config)
    backend_config = _get_backend_config(haproxy_config)

    app_mode = _get_app_mode(haproxy_config)
    config_mode = backend_config.get(HAPROXY_BACKEND_CONFIG_MODE_CONFIG_KEY)
    if app_mode == HAPROXY_APP_MODE_LOAD_BALANCER:
        if config_mode == HAPROXY_CONFIG_MODE_STATIC:
            if not backend_config.get(
                    HAPROXY_BACKEND_SERVERS_CONFIG_KEY):
                raise ValueError("Static servers must be provided with config mode: static.")
        elif config_mode == HAPROXY_CONFIG_MODE_DNS:
            service_name = backend_config.get(HAPROXY_BACKEND_SERVICE_NAME_CONFIG_KEY)
            if not service_name:
                raise ValueError("Service name must be configured for config mode: dns.")
    else:
        if config_mode and config_mode != HAPROXY_CONFIG_MODE_DYNAMIC:
            raise ValueError("API Gateway mode support only dynamic config mode.")

        # API gateway should use http protocol
        service_protocol = haproxy_config.get(
            HAPROXY_SERVICE_PROTOCOL_CONFIG_KEY)
        if service_protocol and service_protocol != HAPROXY_SERVICE_PROTOCOL_HTTP:
            raise ValueError("API Gateway mode should use http protocol.")


def _with_runtime_environment_variables(
        runtime_config, config):
    runtime_envs = {}

    haproxy_config = _get_config(runtime_config)

    high_availability = _is_high_availability(haproxy_config)
    if high_availability:
        runtime_envs["HAPROXY_HIGH_AVAILABILITY"] = high_availability

    runtime_envs["HAPROXY_FRONTEND_PORT"] = _get_service_port(haproxy_config)
    runtime_envs["HAPROXY_FRONTEND_PROTOCOL"] = _get_service_protocol(haproxy_config)

    app_mode = _get_app_mode(haproxy_config)
    runtime_envs["HAPROXY_APP_MODE"] = app_mode

    # Backend discovery support mode for:
    # 1. DNS (given static service name and optionally service tag)
    # 2. Static: a static list of servers
    # 3. Dynamic: a dynamic discovered service (services)
    backend_config = _get_backend_config(haproxy_config)
    if app_mode == HAPROXY_APP_MODE_LOAD_BALANCER:
        _with_runtime_envs_for_load_balancer(
            config, backend_config, runtime_envs)
    elif app_mode == HAPROXY_APP_MODE_API_GATEWAY:
        _with_runtime_envs_for_api_gateway(
            config, backend_config, runtime_envs)
    else:
        raise ValueError("Invalid application mode: {}. "
                         "Must be load-balancer or api-gateway.".format(app_mode))

    runtime_envs["HAPROXY_BACKEND_MAX_SERVERS"] = backend_config.get(
        HAPROXY_BACKEND_MAX_SERVERS_CONFIG_KEY,
        HAPROXY_BACKEND_MAX_SERVERS_DEFAULT)

    balance = backend_config.get(HAPROXY_BACKEND_BALANCE_CONFIG_KEY)
    if not balance:
        balance = HAPROXY_BACKEND_BALANCE_ROUNDROBIN
    runtime_envs["HAPROXY_BACKEND_BALANCE"] = balance
    return runtime_envs


def _get_default_load_balancer_config_mode(config, backend_config):
    cluster_runtime_config = config.get(RUNTIME_CONFIG_KEY)
    if backend_config.get(
            HAPROXY_BACKEND_SERVERS_CONFIG_KEY):
        # if there are static servers configured
        config_mode = HAPROXY_CONFIG_MODE_STATIC
    elif get_service_discovery_runtime(cluster_runtime_config):
        # if there is service selector defined
        if backend_config.get(
                HAPROXY_BACKEND_SELECTOR_CONFIG_KEY):
            config_mode = HAPROXY_CONFIG_MODE_DYNAMIC
        elif backend_config.get(
                HAPROXY_BACKEND_SERVICE_NAME_CONFIG_KEY):
            config_mode = HAPROXY_CONFIG_MODE_DNS
        else:
            config_mode = HAPROXY_CONFIG_MODE_DYNAMIC
    else:
        config_mode = HAPROXY_CONFIG_MODE_STATIC
    return config_mode


def _with_runtime_envs_for_load_balancer(config, backend_config, runtime_envs):
    config_mode = backend_config.get(
        HAPROXY_BACKEND_CONFIG_MODE_CONFIG_KEY)
    if not config_mode:
        config_mode = _get_default_load_balancer_config_mode(
            config, backend_config)

    if config_mode == HAPROXY_CONFIG_MODE_DNS:
        _with_runtime_envs_for_dns(backend_config, runtime_envs)
    elif config_mode == HAPROXY_CONFIG_MODE_STATIC:
        _with_runtime_envs_for_static(backend_config, runtime_envs)
    else:
        _with_runtime_envs_for_dynamic(backend_config, runtime_envs)
    runtime_envs["HAPROXY_CONFIG_MODE"] = config_mode


def _with_runtime_envs_for_dns(backend_config, runtime_envs):
    service_name = backend_config.get(HAPROXY_BACKEND_SERVICE_NAME_CONFIG_KEY)
    if not service_name:
        raise ValueError("Service name must be configured for config mode: dns.")

    service_tag = backend_config.get(
        HAPROXY_BACKEND_SERVICE_TAG_CONFIG_KEY)
    service_cluster = backend_config.get(
        HAPROXY_BACKEND_SERVICE_CLUSTER_CONFIG_KEY)

    service_dns_name = get_rfc2782_service_dns_name(
        service_name, service_tag, service_cluster)
    runtime_envs["HAPROXY_BACKEND_SERVICE_DNS_NAME"] = service_dns_name


def _with_runtime_envs_for_static(backend_config, runtime_envs):
    pass


def _with_runtime_envs_for_dynamic(backend_config, runtime_envs):
    pass


def _get_default_api_gateway_config_mode(config, backend_config):
    cluster_runtime_config = config.get(RUNTIME_CONFIG_KEY)
    if not get_service_discovery_runtime(cluster_runtime_config):
        raise ValueError("Service discovery runtime is needed for API gateway mode.")

    # for simplicity, the API gateway operates with the service selector
    config_mode = HAPROXY_CONFIG_MODE_DYNAMIC
    return config_mode


def _with_runtime_envs_for_api_gateway(config, backend_config, runtime_envs):
    config_mode = backend_config.get(HAPROXY_BACKEND_CONFIG_MODE_CONFIG_KEY)
    if not config_mode:
        config_mode = _get_default_api_gateway_config_mode(
            config, backend_config)

    runtime_envs["HAPROXY_CONFIG_MODE"] = config_mode


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
        service_name: define_runtime_service_on_head_or_all(
            service_discovery_config, service_port,
            _is_high_availability(haproxy_config))
    }
    return services


###################################
# Calls from node when configuring
###################################


def configure_backend(head):
    runtime_config = get_runtime_config_from_node(head)
    haproxy_config = _get_config(runtime_config)

    app_mode = get_runtime_value("HAPROXY_APP_MODE")
    config_mode = get_runtime_value("HAPROXY_CONFIG_MODE")
    if app_mode == HAPROXY_APP_MODE_LOAD_BALANCER:
        if config_mode == HAPROXY_CONFIG_MODE_STATIC:
            _configure_static_backend(haproxy_config)


def _configure_static_backend(haproxy_config):
    backend_config = _get_backend_config(haproxy_config)
    servers = backend_config.get(
        HAPROXY_BACKEND_SERVERS_CONFIG_KEY)
    if servers:
        home_dir = _get_home_dir()
        config_file = os.path.join(
            home_dir, "conf", "haproxy.cfg")
        with open(config_file, "a") as f:
            for server_id, server in enumerate(servers, start=1):
                server_name = get_default_server_name(server_id)
                f.write("    server {} {} check\n".format(
                    server_name, server))


def _get_pull_identifier():
    return "{}-discovery".format(HAPROXY_SERVICE_NAME)


def start_pull_server(head):
    runtime_config = get_runtime_config_from_node(head)
    haproxy_config = _get_config(runtime_config)

    app_mode = get_runtime_value("HAPROXY_APP_MODE")
    if app_mode == HAPROXY_APP_MODE_LOAD_BALANCER:
        discovery_class = "DiscoverBackendServers"
    else:
        discovery_class = "DiscoverAPIGatewayBackendServers"

    service_selector = haproxy_config.get(
            HAPROXY_BACKEND_SELECTOR_CONFIG_KEY, {})
    cluster_name = get_runtime_value(CLOUDTIK_RUNTIME_ENV_CLUSTER)
    exclude_runtime_of_cluster(
        service_selector, BUILT_IN_RUNTIME_HAPROXY, cluster_name)

    service_selector_str = serialize_service_selector(service_selector)

    pull_identifier = _get_pull_identifier()

    cmd = ["cloudtik", "node", "pull", pull_identifier, "start"]
    cmd += ["--pull-class=cloudtik.runtime.haproxy.discovery.{}".format(
        discovery_class)]
    cmd += ["--interval={}".format(
        HAPROXY_DISCOVER_BACKEND_SERVERS_INTERVAL)]
    # job parameters
    if service_selector_str:
        cmd += ["service_selector={}".format(service_selector_str)]
    if app_mode == HAPROXY_APP_MODE_LOAD_BALANCER:
        cmd += ["backend_name={}".format(HAPROXY_BACKEND_NAME_DEFAULT)]
    else:
        # the bind_ip, bind_port and balance type
        bind_ip = get_runtime_value(CLOUDTIK_RUNTIME_ENV_NODE_IP)
        bind_port = get_runtime_value("HAPROXY_FRONTEND_PORT")
        balance_method = get_runtime_value("HAPROXY_BACKEND_BALANCE")
        cmd += ["bind_ip={}".format(bind_ip)]
        cmd += ["bind_port={}".format(bind_port)]
        cmd += ["balance_method={}".format(
            quote(balance_method))]

    cmd_str = " ".join(cmd)
    exec_with_output(cmd_str)


def stop_pull_server():
    pull_identifier = _get_pull_identifier()
    cmd = ["cloudtik", "node", "pull", pull_identifier, "stop"]
    cmd_str = " ".join(cmd)
    exec_with_output(cmd_str)


def _get_backend_server_block(backend_servers):
    backend_server_block = ""
    i = 0
    for backend_server in backend_servers:
        i += 1
        server_name = get_default_server_name(i)
        backend_server_block += "    server %s %s:%s check\n" % (
            server_name,
            backend_server[0], backend_server[1])
    for disabled_slot in range(0, HAPROXY_BACKEND_DYNAMIC_FREE_SLOTS):
        i += 1
        server_name = get_default_server_name(i)
        backend_server_block += "    server %s 0.0.0.0:80 check disabled\n" % (
            server_name)
    return backend_server_block


def update_configuration(backend_servers):
    # write haproxy config file
    conf_dir = os.path.join(_get_home_dir(), "conf")
    template_file = os.path.join(
        conf_dir, "haproxy-template.cfg")
    working_file = os.path.join(
        conf_dir, "haproxy-working.cfg")
    shutil.copyfile(template_file, working_file)

    backend_server_block = _get_backend_server_block(
        backend_servers)
    with open(working_file, "a") as f:
        f.write(backend_server_block)

    config_file = os.path.join(
        conf_dir, "haproxy.cfg")
    # move overwritten
    shutil.move(working_file, config_file)


def update_api_gateway_configuration(
        api_gateway_backends, new_backends,
        bind_ip, bind_port, balance_method):
    if not bind_port:
        bind_port = HAPROXY_SERVICE_PORT_DEFAULT
    service_protocol = HAPROXY_SERVICE_PROTOCOL_HTTP
    conf_dir = os.path.join(_get_home_dir(), "conf")
    template_file = os.path.join(
        conf_dir, "haproxy-template.cfg")
    working_file = os.path.join(
        conf_dir, "haproxy-working.cfg")
    shutil.copyfile(template_file, working_file)

    # sort to make the order to the backends are always the same
    sorted_api_gateway_backends = sorted(api_gateway_backends.items())

    with open(working_file, "a") as f:
        f.write("frontend api_gateway\n")
        if bind_ip:
            f.write(f"    bind {bind_ip}:{bind_port}\n")
        else:
            f.write(f"    bind :{bind_port}\n")
        f.write(f"    mode {service_protocol}\n")
        f.write(f"    option {service_protocol}log\n")
        # route to a backend based on path's prefix
        for backend_name, backend_servers in sorted_api_gateway_backends:
            f.write("    use_backend " + backend_name +
                    " if { path /" + backend_name +
                    " } || { path_beg /" + backend_name +
                    "/ }\n")

        f.write("\n")
        # write each backend
        for backend_name, backend_servers in sorted_api_gateway_backends:
            backend_server_block = _get_backend_server_block(
                backend_servers)
            f.write(f"backend {backend_name}\n")
            f.write(f"    mode {service_protocol}\n")
            if balance_method:
                f.write(f"    balance {balance_method}\n")
            f.write("    http-request replace-path /" + backend_name +
                    "(/)?(.*) /\\2\n")
            f.write(backend_server_block)

    config_file = os.path.join(
        conf_dir, "haproxy.cfg")
    # move overwritten
    shutil.move(working_file, config_file)

    if new_backends:
        # Need reload haproxy if there is new backend added
        exec_with_call("sudo service haproxy reload")
