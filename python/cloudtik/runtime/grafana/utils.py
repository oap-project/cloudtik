import os
from typing import Any, Dict

from cloudtik.core._private.constants import CLOUDTIK_RUNTIME_ENV_CLUSTER
from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_PROMETHEUS
from cloudtik.core._private.runtime_utils import get_runtime_config_from_node, get_runtime_value, get_runtime_head_ip, \
    save_yaml
from cloudtik.core._private.service_discovery.runtime_services import get_service_discovery_runtime, \
    get_services_of_runtime
from cloudtik.core._private.service_discovery.utils import \
    get_canonical_service_name, define_runtime_service_on_head_or_all, get_service_discovery_config, \
    SERVICE_DISCOVERY_PORT
from cloudtik.core._private.utils import RUNTIME_CONFIG_KEY

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
GRAFANA_DATA_SOURCES_SCOPE_CONFIG_KEY = "data_sources_scope"
# statically configured data sources
GRAFANA_DATA_SOURCES_CONFIG_KEY = "data_sources"

GRAFANA_SERVICE_NAME = "grafana"
GRAFANA_SERVICE_PORT_DEFAULT = 3000

GRAFANA_DATA_SOURCES_SCOPE_NONE = "none"
GRAFANA_DATA_SOURCES_SCOPE_LOCAL = "local"
GRAFANA_DATA_SOURCES_SCOPE_WORKSPACE = "workspace"


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
        runtime_config, config):
    runtime_envs = {}

    grafana_config = _get_config(runtime_config)
    cluster_runtime_config = config.get(RUNTIME_CONFIG_KEY)

    service_port = _get_service_port(grafana_config)
    runtime_envs["GRAFANA_SERVICE_PORT"] = service_port

    high_availability = _is_high_availability(grafana_config)
    if high_availability:
        runtime_envs["GRAFANA_HIGH_AVAILABILITY"] = high_availability

    data_sources_scope = grafana_config.get(GRAFANA_DATA_SOURCES_SCOPE_CONFIG_KEY)
    if data_sources_scope == GRAFANA_DATA_SOURCES_SCOPE_WORKSPACE:
        # we need service discovery service for discover workspace scope data sources
        if not get_service_discovery_runtime(cluster_runtime_config):
            raise RuntimeError(
                "Service discovery service is needed for workspace scoped data sources.")
    elif not data_sources_scope:
        data_sources_scope = GRAFANA_DATA_SOURCES_SCOPE_LOCAL

    runtime_envs["GRAFANA_DATA_SOURCES_SCOPE"] = data_sources_scope

    if data_sources_scope == GRAFANA_DATA_SOURCES_SCOPE_LOCAL:
        with_local_data_sources(grafana_config, config, runtime_envs)
    elif data_sources_scope == GRAFANA_DATA_SOURCES_SCOPE_WORKSPACE:
        with_workspace_data_sources(grafana_config, config, runtime_envs)

    return runtime_envs


def with_local_data_sources(
        grafana_config, config, runtime_envs):
    prometheus_services = get_services_of_runtime(
        config, BUILT_IN_RUNTIME_PROMETHEUS)
    if prometheus_services:
        service = next(iter(prometheus_services.values()))
        runtime_envs["GRAFANA_LOCAL_PROMETHEUS_PORT"] = service[SERVICE_DISCOVERY_PORT]


def with_workspace_data_sources(
        grafana_config, config, runtime_envs):
    # discovery through file periodically updated by daemon
    pass


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
            _is_high_availability(grafana_config),
            metrics=True),
    }
    return services


###################################
# Calls from node when configuring
###################################


def configure_data_sources(head):
    runtime_config = get_runtime_config_from_node(head)
    grafana_config = _get_config(runtime_config)

    data_sources = grafana_config.get(GRAFANA_DATA_SOURCES_CONFIG_KEY)
    if data_sources is None:
        data_sources = []

    data_sources_scope = get_runtime_value("GRAFANA_DATA_SOURCES_SCOPE")
    prometheus_port = get_runtime_value("GRAFANA_LOCAL_PROMETHEUS_PORT")
    if data_sources_scope == GRAFANA_DATA_SOURCES_SCOPE_LOCAL and prometheus_port:
        # add a local data resource for prometheus
        cluster_name = get_runtime_value(CLOUDTIK_RUNTIME_ENV_CLUSTER)
        head_ip = get_runtime_head_ip(head)
        url = "http://{}:{}".format(head_ip, prometheus_port)
        prometheus_data_source = {
            "name": cluster_name,
            "type": "prometheus",
            "access": "proxy",
            "url": url,
            "isDefault": True,
        }
        data_sources.append(prometheus_data_source)

    if data_sources:
        _save_data_sources_config(data_sources)


def _save_data_sources_config(data_sources):
    # writhe the data sources file
    home_dir = _get_home_dir()
    config_file = os.path.join(
        home_dir, "conf", "provisioning",
        "datasources", "static-data-sources.yaml")

    config_object = {
        "apiVersion": 1,
        "datasources": data_sources
    }
    save_yaml(config_file, config_object)
