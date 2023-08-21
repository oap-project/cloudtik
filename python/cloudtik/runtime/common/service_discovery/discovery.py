from enum import Enum, auto
from typing import Dict, Any

from cloudtik.core._private.service_discovery.runtime_services import get_service_discovery_runtime, \
    get_consul_server_addresses
from cloudtik.core._private.service_discovery.utils import ServiceAddressType
from cloudtik.core._private.utils import RUNTIME_CONFIG_KEY
from cloudtik.runtime.common.service_discovery.consul import \
    query_one_service_from_consul
from cloudtik.runtime.common.service_discovery.workspace import query_one_service_from_workspace


class DiscoveryType(Enum):
    WORKSPACE = auto()
    CLUSTER = auto()
    LOCAL = auto()
    ANY = auto()


def query_one_service(
        cluster_config: Dict[str, Any], service_selector,
        discovery_type: DiscoveryType = DiscoveryType.ANY,
        address_type: ServiceAddressType = ServiceAddressType.NODE_IP):
    runtime_config = cluster_config.get(RUNTIME_CONFIG_KEY)
    if (discovery_type == DiscoveryType.ANY or
            discovery_type == DiscoveryType.LOCAL):
        if get_service_discovery_runtime(runtime_config):
            # try first use service discovery if available
            service = query_one_service_from_consul(
                service_selector, address_type=address_type)
            if service:
                return service
        if discovery_type == DiscoveryType.LOCAL:
            return None

    if (discovery_type == DiscoveryType.ANY or
            discovery_type == DiscoveryType.CLUSTER):
        if get_service_discovery_runtime(runtime_config):
            # For case that the local consul is not yet available
            # If the current cluster is consul server or consul server is not available
            # the address will be None
            addresses = get_consul_server_addresses(runtime_config)
            if addresses is not None:
                # TODO: we can retry other addresses if failed
                service = query_one_service_from_consul(
                    service_selector, address_type=address_type,
                    address=addresses[0])
                if service:
                    return service
        if discovery_type == DiscoveryType.CLUSTER:
            return None

    if (discovery_type == DiscoveryType.ANY or
            discovery_type == DiscoveryType.WORKSPACE):
        # try workspace discovery
        service = query_one_service_from_workspace(
            cluster_config, service_selector)
        if service:
            return service
        if discovery_type == DiscoveryType.WORKSPACE:
            return None

    return None
