from cloudtik.core._private.utils import publish_cluster_variable, subscribe_cluster_variable
from cloudtik.runtime.common.service_discovery.utils import get_service_addresses_string, \
    get_service_addresses_from_string


def get_service_registry_name(
        runtime_type,
        service_name=None):
    # Service register in cluster in the format of:
    # 1. runtime_name
    # 2. runtime_name.service_name
    if service_name:
        registry_name = "{}.{}".format(
            runtime_type, service_name)
    else:
        registry_name = runtime_type
    return registry_name


def register_service_to_cluster(
        runtime_type,
        service_addresses,
        service_name=None):
    if not service_addresses:
        raise ValueError("Must specify service addresses when registering a service.")
    registry_name = get_service_registry_name(
        runtime_type, service_name)
    registry_addresses = get_service_addresses_string(
        service_addresses)
    publish_cluster_variable(registry_name, registry_addresses)


def query_service_from_cluster(
        runtime_type,
        service_name=None):
    registry_name = get_service_registry_name(
        runtime_type, service_name)
    registry_addresses = subscribe_cluster_variable(registry_name)
    if not registry_addresses:
        return None
    return get_service_addresses_from_string(registry_addresses)
