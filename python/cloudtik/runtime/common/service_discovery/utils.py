from cloudtik.core._private.core_utils import get_address_string


class ServiceInstance:
    """A service instance returned by discovering processes"""
    def __init__(
            self, service_name, service_addresses,
            runtime_type, cluster_name=None, tags=None):
        self.service_name = service_name
        self.service_addresses = service_addresses
        self.runtime_type = runtime_type
        self.cluster_name = cluster_name
        self.tags = tags


def get_service_addresses_string(service_addresses):
    # allow two format: host,host,host or host:port,host:port
    return ",".join([get_address_string(
        service_address[0], service_address[1])
                     if service_address[1] else service_address[0]
                     for service_address in service_addresses])


def get_service_addresses_from_string(registry_addresses):
    registry_address_list = [x.strip() for x in registry_addresses.split(',')]
    service_addresses = []
    for registry_address in registry_address_list:
        address_parts = [x.strip() for x in registry_address.split(':')]
        n = len(address_parts)
        if n == 1:
            host = address_parts[0]
            port = 0
        elif n == 2:
            host = address_parts[0]
            port = int(address_parts[1])
        else:
            raise ValueError(
                "Invalid service address find in: {}".format(registry_addresses))
        service_addresses.append((host, port))
    return service_addresses
