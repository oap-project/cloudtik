from enum import Enum, auto
from typing import Dict, Any

from cloudtik.core._private.service_discovery.runtime_services import get_service_discovery_runtime
from cloudtik.core._private.utils import RUNTIME_CONFIG_KEY
from cloudtik.runtime.common.service_discovery.consul import \
    query_one_service_from_consul
from cloudtik.runtime.common.service_discovery.workspace import query_one_service_from_workspace


class DiscoveryType(Enum):
    WORKSPACE = auto()
    CONSUL = auto()
    CONSUL_LOCAL = auto()
    ANY = auto()


def query_one_service(
        cluster_config: Dict[str, Any], service_selector,
        discovery_type: DiscoveryType = DiscoveryType.ANY):
    runtime_config = cluster_config.get(RUNTIME_CONFIG_KEY)
    if (discovery_type == DiscoveryType.ANY or
            discovery_type == DiscoveryType.CONSUL_LOCAL):
        if get_service_discovery_runtime(runtime_config):
            # try first use service discovery if available
            service_addresses = query_one_service_from_consul(
                service_selector)
            if service_addresses:
                return service_addresses
        if discovery_type == DiscoveryType.CONSUL_LOCAL:
            return None

    if (discovery_type == DiscoveryType.ANY or
            discovery_type == DiscoveryType.WORKSPACE):
        # try workspace discovery
        service_addresses = query_one_service_from_workspace(
            cluster_config, service_selector)
        if service_addresses:
            return service_addresses
        if discovery_type == DiscoveryType.WORKSPACE:
            return None

    return None
