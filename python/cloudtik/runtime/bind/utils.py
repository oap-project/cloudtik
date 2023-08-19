import os
from typing import Any, Dict

from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_BIND, BUILT_IN_RUNTIME_CONSUL
from cloudtik.core._private.service_discovery.utils import \
    get_canonical_service_name, define_runtime_service, \
    get_service_discovery_config, SERVICE_DISCOVERY_FEATURE_DNS
from cloudtik.core._private.util.resolv_conf import get_resolv_conf_name_servers
from cloudtik.core._private.utils import get_runtime_config
from cloudtik.runtime.common.service_discovery.cluster import has_runtime_in_cluster

RUNTIME_PROCESSES = [
        # The first element is the substring to filter.
        # The second element, if True, is to filter ps results by command name.
        # The third element is the process name.
        # The forth element, if node, the process should on all nodes,if head, the process should on head node.
        ["named", True, "DNS Server", "node"],
    ]

BIND_SERVICE_PORT_CONFIG_KEY = "port"
BIND_DNSSEC_VALIDATION_CONFIG_KEY = "dnssec_validation"
BIND_DEFAULT_RESOLVER_CONFIG_KEY = "default_resolver"

BIND_SERVICE_NAME = BUILT_IN_RUNTIME_BIND
BIND_SERVICE_PORT_DEFAULT = 53


def _get_config(runtime_config: Dict[str, Any]):
    return runtime_config.get(BUILT_IN_RUNTIME_BIND, {})


def _get_service_port(bind_config: Dict[str, Any]):
    return bind_config.get(
        BIND_SERVICE_PORT_CONFIG_KEY, BIND_SERVICE_PORT_DEFAULT)


def _get_home_dir():
    return os.path.join(
        os.getenv("HOME"), "runtime", BUILT_IN_RUNTIME_BIND)


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _with_runtime_environment_variables(
        runtime_config, config):
    runtime_envs = {}
    bind_config = _get_config(runtime_config)

    service_port = _get_service_port(bind_config)
    runtime_envs["BIND_SERVICE_PORT"] = service_port

    dnssec_validation = bind_config.get(BIND_DNSSEC_VALIDATION_CONFIG_KEY)
    if dnssec_validation:
        runtime_envs["BIND_DNSSEC_VALIDATION"] = dnssec_validation

    cluster_runtime_config = get_runtime_config(config)
    if has_runtime_in_cluster(
            cluster_runtime_config, BUILT_IN_RUNTIME_CONSUL):
        runtime_envs["BIND_CONSUL_RESOLVE"] = True

    default_resolver = bind_config.get(
        BIND_DEFAULT_RESOLVER_CONFIG_KEY, False)
    if default_resolver:
        runtime_envs["BIND_DEFAULT_RESOLVER"] = True

    return runtime_envs


def _get_runtime_services(
        runtime_config: Dict[str, Any], cluster_name: str) -> Dict[str, Any]:
    bind_config = _get_config(runtime_config)
    service_discovery_config = get_service_discovery_config(bind_config)
    service_name = get_canonical_service_name(
        service_discovery_config, cluster_name, BIND_SERVICE_NAME)
    service_port = _get_service_port(bind_config)
    services = {
        service_name: define_runtime_service(
            service_discovery_config, service_port,
            features=[SERVICE_DISCOVERY_FEATURE_DNS]),
    }
    return services


###################################
# Calls from node when configuring
###################################


def configure_upstream(head):
    conf_dir = os.path.join(
        _get_home_dir(), "conf")
    origin_resolv_conf = os.path.join(
        conf_dir, "resolv.conf")
    upstream_config_file = os.path.join(
        conf_dir, "named.conf.upstream")

    name_servers = get_resolv_conf_name_servers(
        origin_resolv_conf)
    with open(upstream_config_file, "w") as f:
        f.write('zone "." {\n')
        f.write('  type forward;\n')
        f.write('  forwarders {\n')
        for name_server in name_servers:
            f.write("    {};\n".format(name_server))
        f.write('  };\n')
        f.write('};\n')
