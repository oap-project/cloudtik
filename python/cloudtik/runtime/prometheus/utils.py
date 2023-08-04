import os
from typing import Any, Dict

from cloudtik.core._private.service_discovery.runtime_services import get_service_discovery_runtime
from cloudtik.core._private.service_discovery.utils import \
    get_canonical_service_name, \
    define_runtime_service_on_head_or_all, get_service_discovery_config
from cloudtik.core._private.utils import RUNTIME_CONFIG_KEY

RUNTIME_PROCESSES = [
        # The first element is the substring to filter.
        # The second element, if True, is to filter ps results by command name.
        # The third element is the process name.
        # The forth element, if node, the process should on all nodes,if head, the process should on head node.
        ["prometheus", True, "Prometheus", "node"],
    ]

PROMETHEUS_RUNTIME_CONFIG_KEY = "prometheus"
PROMETHEUS_SERVICE_PORT_CONFIG_KEY = "port"
PROMETHEUS_HIGH_AVAILABILITY_CONFIG_KEY = "high_availability"
PROMETHEUS_SCRAPE_SCOPE_CONFIG_KEY = "scrape_scope"
PROMETHEUS_SERVICE_DISCOVERY_CONFIG_KEY = "service_discovery"

PROMETHEUS_SERVICE_NAME = "prometheus"
PROMETHEUS_SERVICE_PORT_DEFAULT = 9090

PROMETHEUS_SERVICE_DISCOVERY_FILE = "FILE"
PROMETHEUS_SERVICE_DISCOVERY_CONSUL = "CONSUL"

PROMETHEUS_SCRAPE_SCOPE_LOCAL = "local"
PROMETHEUS_SCRAPE_SCOPE_WORKSPACE = "workspace"


def _get_config(runtime_config: Dict[str, Any]):
    return runtime_config.get(PROMETHEUS_RUNTIME_CONFIG_KEY, {})


def _get_service_port(prometheus_config: Dict[str, Any]):
    return prometheus_config.get(
        PROMETHEUS_SERVICE_PORT_CONFIG_KEY, PROMETHEUS_SERVICE_PORT_DEFAULT)


def _is_high_availability(prometheus_config: Dict[str, Any]):
    return prometheus_config.get(
        PROMETHEUS_HIGH_AVAILABILITY_CONFIG_KEY, False)


def _get_home_dir():
    return os.path.join(os.getenv("HOME"), "runtime", PROMETHEUS_SERVICE_NAME)


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _get_runtime_logs():
    home_dir = _get_home_dir()
    logs_dir = os.path.join(home_dir, "logs")
    return {"prometheus": logs_dir}


def _with_runtime_environment_variables(
        runtime_config, config):
    runtime_envs = {}

    prometheus_config = _get_config(runtime_config)

    service_port = _get_service_port(prometheus_config)
    runtime_envs["PROMETHEUS_SERVICE_PORT"] = service_port

    high_availability = _is_high_availability(prometheus_config)
    if high_availability:
        runtime_envs["PROMETHEUS_HIGH_AVAILABILITY"] = high_availability

    sd = prometheus_config.get(PROMETHEUS_SERVICE_DISCOVERY_CONFIG_KEY)
    if not sd:
        # auto decide
        cluster_runtime_config = config.get(RUNTIME_CONFIG_KEY)
        if get_service_discovery_runtime(cluster_runtime_config):
            sd = PROMETHEUS_SERVICE_DISCOVERY_CONSUL
        else:
            sd = PROMETHEUS_SERVICE_DISCOVERY_FILE

    if sd == PROMETHEUS_SERVICE_DISCOVERY_FILE:
        _with_file_sd_environment_variables(
            prometheus_config, config, runtime_envs)
    elif sd == PROMETHEUS_SERVICE_DISCOVERY_CONSUL:
        _with_consul_sd_environment_variables(
            prometheus_config, config, runtime_envs)
    else:
        raise RuntimeError(
            "Unsupported service discovery type: {}. "
            "Valid types are: {}, {}.".format(
                sd,
                PROMETHEUS_SERVICE_DISCOVERY_FILE,
                PROMETHEUS_SERVICE_DISCOVERY_CONSUL))

    scrape_scope = prometheus_config.get(PROMETHEUS_SCRAPE_SCOPE_CONFIG_KEY)
    if scrape_scope == PROMETHEUS_SCRAPE_SCOPE_WORKSPACE:
        # make sure
        cluster_runtime_config = config.get(RUNTIME_CONFIG_KEY)
        if not get_service_discovery_runtime(cluster_runtime_config):
            raise RuntimeError("Service discovery service is needed for workspace scoped scrape.")

        sd = PROMETHEUS_SERVICE_DISCOVERY_CONSUL
        scrape_scope = PROMETHEUS_SCRAPE_SCOPE_WORKSPACE
    elif not scrape_scope:
        scrape_scope = PROMETHEUS_SCRAPE_SCOPE_LOCAL

    runtime_envs["PROMETHEUS_SERVICE_DISCOVERY"] = sd
    runtime_envs["PROMETHEUS_SCRAPE_SCOPE"] = scrape_scope
    return runtime_envs


def _with_file_sd_environment_variables(
        prometheus_config, config, runtime_envs):
    # TODO: discovery through file periodically updated by daemon
    pass


def _with_consul_sd_environment_variables(
        prometheus_config, config, runtime_envs):
    # TODO: export variables necessary for Consul service discovery
    pass


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
    service_discovery_config = get_service_discovery_config(prometheus_config)
    service_name = get_canonical_service_name(
        service_discovery_config, cluster_name, PROMETHEUS_SERVICE_NAME)
    service_port = _get_service_port(prometheus_config)
    services = {
        service_name: define_runtime_service_on_head_or_all(
            service_discovery_config, service_port,
            _is_high_availability(prometheus_config),
            metrics=True)
    }
    return services
