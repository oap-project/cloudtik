import os
from typing import Any, Dict

from cloudtik.core._private.core_utils import get_config_for_update, get_list_for_update, get_address_string
from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_APISIX, BUILT_IN_RUNTIME_ETCD
from cloudtik.core._private.runtime_utils import get_runtime_config_from_node, load_and_save_yaml
from cloudtik.core._private.service_discovery.runtime_services import get_service_discovery_runtime
from cloudtik.core._private.service_discovery.utils import \
    get_canonical_service_name, define_runtime_service, \
    get_service_discovery_config, SERVICE_DISCOVERY_FEATURE_API_GATEWAY, SERVICE_DISCOVERY_PROTOCOL_HTTP
from cloudtik.core._private.utils import get_runtime_config
from cloudtik.runtime.common.service_discovery.runtime_discovery import discover_etcd_from_workspace, \
    discover_etcd_on_head, ETCD_URI_KEY, is_etcd_service_discovery
from cloudtik.runtime.common.service_discovery.utils import get_service_addresses_from_string

RUNTIME_PROCESSES = [
        # The first element is the substring to filter.
        # The second element, if True, is to filter ps results by command name.
        # The third element is the process name.
        # The forth element, if node, the process should on all nodes,if head, the process should on head node.
        ["apisix", True, "APISIX", "node"],
    ]

APISIX_SERVICE_PORT_CONFIG_KEY = "port"
APISIX_ADMIN_PORT_CONFIG_KEY = "admin_port"

APISIX_SERVICE_NAME = BUILT_IN_RUNTIME_APISIX
APISIX_SERVICE_PORT_DEFAULT = 9080
APISIX_ADMIN_PORT_DEFAULT = 9180


def _get_config(runtime_config: Dict[str, Any]):
    return runtime_config.get(BUILT_IN_RUNTIME_APISIX, {})


def _get_service_port(apisix_config: Dict[str, Any]):
    return apisix_config.get(
        APISIX_SERVICE_PORT_CONFIG_KEY, APISIX_SERVICE_PORT_DEFAULT)


def _get_admin_port(apisix_config: Dict[str, Any]):
    return apisix_config.get(
        APISIX_ADMIN_PORT_CONFIG_KEY, APISIX_ADMIN_PORT_DEFAULT)


def _get_home_dir():
    return os.path.join(
        os.getenv("HOME"), "runtime", BUILT_IN_RUNTIME_APISIX)


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _config_depended_services(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    cluster_config = discover_etcd_from_workspace(
        cluster_config, BUILT_IN_RUNTIME_ETCD)
    return cluster_config


def _prepare_config_on_head(cluster_config: Dict[str, Any]):
    cluster_config = discover_etcd_on_head(
        cluster_config, BUILT_IN_RUNTIME_ETCD)

    _validate_config(cluster_config, final=True)
    return cluster_config


def _validate_config(config: Dict[str, Any], final=False):
    # Check etcd configuration
    runtime_config = get_runtime_config(config)
    apisix_config = _get_config(runtime_config)
    etcd_uri = apisix_config.get(ETCD_URI_KEY)
    if not etcd_uri:
        # if there is service discovery mechanism, assume we can get from service discovery
        if (final or not is_etcd_service_discovery(apisix_config) or
                not get_service_discovery_runtime(runtime_config)):
            raise ValueError("ETCD must be configured for APISIX.")


def _with_runtime_environment_variables(
        runtime_config, config):
    runtime_envs = {}
    apisix_config = _get_config(runtime_config)

    service_port = _get_service_port(apisix_config)
    runtime_envs["APISIX_SERVICE_PORT"] = service_port

    admin_port = _get_admin_port(apisix_config)
    runtime_envs["APISIX_ADMIN_PORT"] = admin_port

    return runtime_envs


def _get_runtime_endpoints(
        runtime_config: Dict[str, Any], cluster_head_ip):
    service_port = _get_service_port(runtime_config)
    endpoints = {
        "apisix": {
            "name": "APISIX",
            "url": "http://{}".format(
                get_address_string(cluster_head_ip, service_port))
        },
    }
    return endpoints


def _get_head_service_ports(
        runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    service_port = _get_service_port(runtime_config)
    service_ports = {
        "apisix": {
            "protocol": "TCP",
            "port": service_port,
        },
    }
    return service_ports


def _get_runtime_services(
        runtime_config: Dict[str, Any], cluster_name: str) -> Dict[str, Any]:
    apisix_config = _get_config(runtime_config)
    service_discovery_config = get_service_discovery_config(apisix_config)
    service_name = get_canonical_service_name(
        service_discovery_config, cluster_name, APISIX_SERVICE_NAME)
    service_port = _get_service_port(apisix_config)
    services = {
        service_name: define_runtime_service(
            service_discovery_config, service_port,
            protocol=SERVICE_DISCOVERY_PROTOCOL_HTTP,
            features=[SERVICE_DISCOVERY_FEATURE_API_GATEWAY]),
    }
    return services


###################################
# Calls from node when configuring
###################################


def update_configurations(head):
    runtime_config = get_runtime_config_from_node(head)
    apisix_config = _get_config(runtime_config)
    etcd_uri = apisix_config.get(ETCD_URI_KEY)
    if etcd_uri:
        _update_etcd_hosts(etcd_uri)


def _update_etcd_hosts(etcd_uri):
    home_dir = _get_home_dir()
    config_file = os.path.join(home_dir, "conf", "config.yaml")
    service_addresses = get_service_addresses_from_string(etcd_uri)

    def update_etcd_hosts(config_object):
        deployment = get_config_for_update(config_object, "deployment")
        etcd = get_config_for_update(deployment, "etcd")
        hosts = get_list_for_update(etcd, "host")
        for service_address in service_addresses:
            hosts.append("http://{}".format(
                get_address_string(service_address[0], service_address[1])))

    load_and_save_yaml(config_file, update_etcd_hosts)
