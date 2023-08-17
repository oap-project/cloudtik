import os
from typing import Any, Dict

from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_COREDNS, BUILT_IN_RUNTIME_CONSUL
from cloudtik.core._private.service_discovery.utils import \
    get_canonical_service_name, define_runtime_service, \
    get_service_discovery_config
from cloudtik.core._private.utils import get_runtime_config
from cloudtik.runtime.common.service_discovery.cluster import has_runtime_in_cluster

RUNTIME_PROCESSES = [
        # The first element is the substring to filter.
        # The second element, if True, is to filter ps results by command name.
        # The third element is the process name.
        # The forth element, if node, the process should on all nodes,if head, the process should on head node.
        ["coredns", True, "Core DNS", "node"],
    ]

COREDNS_SERVICE_PORT_CONFIG_KEY = "port"

COREDNS_SERVICE_NAME = BUILT_IN_RUNTIME_COREDNS
COREDNS_SERVICE_PORT_DEFAULT = 53


def _get_config(runtime_config: Dict[str, Any]):
    return runtime_config.get(BUILT_IN_RUNTIME_COREDNS, {})


def _get_service_port(coredns_config: Dict[str, Any]):
    return coredns_config.get(
        COREDNS_SERVICE_PORT_CONFIG_KEY, COREDNS_SERVICE_PORT_DEFAULT)


def _get_home_dir():
    return os.path.join(
        os.getenv("HOME"), "runtime", BUILT_IN_RUNTIME_COREDNS)


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _with_runtime_environment_variables(
        runtime_config, config):
    runtime_envs = {}
    coredns_config = _get_config(runtime_config)

    service_port = _get_service_port(coredns_config)
    runtime_envs["COREDNS_SERVICE_PORT"] = service_port

    cluster_runtime_config = get_runtime_config(config)
    if has_runtime_in_cluster(
            cluster_runtime_config, BUILT_IN_RUNTIME_CONSUL):
        runtime_envs["COREDNS_CONSUL_RESOLVE"] = True

    return runtime_envs


def _get_runtime_services(
        runtime_config: Dict[str, Any], cluster_name: str) -> Dict[str, Any]:
    coredns_config = _get_config(runtime_config)
    service_discovery_config = get_service_discovery_config(coredns_config)
    service_name = get_canonical_service_name(
        service_discovery_config, cluster_name, COREDNS_SERVICE_NAME)
    service_port = _get_service_port(coredns_config)
    services = {
        service_name: define_runtime_service(
            service_discovery_config, service_port),
    }
    return services
