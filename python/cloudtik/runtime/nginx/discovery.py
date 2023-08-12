import logging

from cloudtik.core._private.core_utils import get_json_object_hash
from cloudtik.core._private.service_discovery.utils import deserialize_service_selector
from cloudtik.core._private.util.pull.pull_job import PullJob
from cloudtik.runtime.common.service_discovery.consul \
    import query_services, query_service_nodes, get_service_address
from cloudtik.runtime.nginx.utils import update_load_balancer_configuration, \
    update_api_gateway_dynamic_backends, update_api_gateway_dns_backends

logger = logging.getLogger(__name__)


class DiscoverJob(PullJob):
    def __init__(self,
                 service_selector=None,
                 balance_method=None
                 ):
        self.service_selector = deserialize_service_selector(
            service_selector)
        self.balance_method = balance_method
        self.last_config_hash = None

    def _query_services(self):
        return query_services(self.service_selector)

    def _query_service_nodes(self, service_name):
        return query_service_nodes(service_name, self.service_selector)


class DiscoverBackendServers(DiscoverJob):
    """Pulling job for discovering backend targets and
    update the config if there are new or deleted servers with reload.
    """

    def __init__(self,
                 service_selector=None,
                 balance_method=None
                 ):
        super().__init__(service_selector, balance_method)

    def pull(self):
        selected_services = self._query_services()
        backend_servers = {}

        for service_name in selected_services:
            service_nodes = self._query_service_nodes(service_name)
            # each node is a data source. if many nodes form a load balancer in a cluster
            # it should be filtered by service selector using service name ,tags or labels
            for service_node in service_nodes:
                server_address = get_service_address(service_node)
                server_key = "{}:{}".format(server_address[0], server_address[1])
                backend_servers[server_key] = server_address
        if not backend_servers:
            logger.warning("No live servers return from the service selector.")

        # Finally, rebuild the configuration for reloads
        servers_hash = get_json_object_hash(backend_servers)
        if servers_hash != self.last_config_hash:
            # save config file and reload only when data changed
            update_load_balancer_configuration(
                backend_servers, self.balance_method)
            self.last_config_hash = servers_hash


class DiscoverAPIGatewayBackendServers(DiscoverJob):
    """Pulling job for discovering backend targets for API gateway backends
    and update the config if there are new or deleted backends.
    The selectors are used to select the list of services (include a service tag or service cluster)
    The servers are discovered through DNS by service name
    and optionally service tag and service cluster
    """

    def __init__(self,
                 service_selector=None,
                 balance_method=None
                 ):
        super().__init__(service_selector, balance_method)
        # TODO: logging the job parameters

    def pull(self):
        selected_services = self._query_services()

        api_gateway_backends = {}
        for service_name in selected_services:
            service_nodes = self._query_service_nodes(service_name)
            # each node is a data source. if many nodes form a load balancer in a cluster
            # it should be filtered by service selector using service name ,tags or labels

            backend_servers = {}
            for service_node in service_nodes:
                server_address = get_service_address(service_node)
                server_key = "{}:{}".format(server_address[0], server_address[1])
                backend_servers[server_key] = server_address

            # TODO: currently use service_name as backend_name and path prefix for simplicity
            #  future to support more flexible cases
            backend_name = service_name
            if not backend_servers:
                logger.warning("No live servers return from the service selector.")

            api_gateway_backends[backend_name] = backend_servers

        backends_hash = get_json_object_hash(api_gateway_backends)
        if backends_hash != self.last_config_hash:
            # save config file and reload only when data changed
            update_api_gateway_dynamic_backends(
                api_gateway_backends, self.balance_method)
            self.last_config_hash = backends_hash


class DiscoverAPIGatewayBackends(DiscoverJob):
    def __init__(self,
                 service_selector=None,
                 balance_method=None
                 ):
        super().__init__(service_selector, balance_method)
        # TODO: logging the job parameters

    def pull(self):
        selected_services = self._query_services()

        api_gateway_backends = {}
        for service_name in selected_services:
            service_nodes = self._query_service_nodes(service_name)

            # TODO: currently use service_name as backend_name and path prefix for simplicity
            #  future to support more flexible cases
            if not service_nodes:
                logger.warning("No live servers return from the service selector.")
            else:
                backend_name = service_name
                backend_service = self.get_dns_backend_service(service_nodes)
                api_gateway_backends[backend_name] = backend_service

        backends_hash = get_json_object_hash(api_gateway_backends)
        if backends_hash != self.last_config_hash:
            # save config file and reload only when data changed
            update_api_gateway_dns_backends(
                api_gateway_backends)
            self.last_config_hash = backends_hash

    @staticmethod
    def get_dns_backend_service(service_nodes):
        # get service port in one of the servers
        service_node = service_nodes[0]
        server_address = get_service_address(service_node)

        # get a common set of tags
        tags = set(service_node.get("ServiceTags", []))
        if tags:
            for service_node in service_nodes[1:]:
                service_tags = set(service_node.get("ServiceTags", []))
                tags = tags.intersection(service_tags)

        backend_service = {
            "service_port": server_address[1],
            "tags": list(tags),
        }
        return backend_service
