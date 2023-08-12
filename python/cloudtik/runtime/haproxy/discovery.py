import logging

from cloudtik.core._private.service_discovery.utils import deserialize_service_selector
from cloudtik.core._private.util.pull.pull_job import PullJob
from cloudtik.runtime.common.service_discovery.consul \
    import query_services, query_service_nodes, get_service_address
from cloudtik.runtime.haproxy.admin_api import list_backend_servers, enable_backend_slot, disable_backend_slot, \
    add_backend_slot, get_backend_server_address, delete_backend_slot, list_backends
from cloudtik.runtime.haproxy.utils import update_configuration, get_default_server_name, \
    HAPROXY_BACKEND_DYNAMIC_FREE_SLOTS, update_api_gateway_configuration

logger = logging.getLogger(__name__)


HAPROXY_SERVERS = [("127.0.0.1", 19999)]


def _list_backends():
    return list_backends(HAPROXY_SERVERS[0])


def _update_backend(backend_name, backend_servers):
    # Now update each HAProxy server with the backends
    for haproxy_server in HAPROXY_SERVERS:
        active_servers, inactive_servers = list_backend_servers(
            haproxy_server, backend_name)
        total_server_slots = len(active_servers) + len(inactive_servers)

        missing_backend_servers = []
        for backend_server in backend_servers:
            server_address = get_backend_server_address(backend_server)
            if server_address in active_servers:
                # Ignore backends already set
                del active_servers[server_address]
            else:
                if len(inactive_servers) > 0:
                    server_name = inactive_servers.pop(0)
                    enable_backend_slot(
                        haproxy_server, backend_name,
                        server_name, backend_server)
                else:
                    # we need a reload of the configuration after
                    missing_backend_servers.append(backend_server)

        # mark inactive for remaining servers in active servers set but not appearing
        for remaining_server, server_name in active_servers.items():
            # disable
            disable_backend_slot(
                haproxy_server, backend_name, server_name)
            # if there are missing backend, use it
            if missing_backend_servers:
                backend_server = missing_backend_servers.pop(0)
                enable_backend_slot(
                    haproxy_server, backend_name,
                    server_name, backend_server)

        if missing_backend_servers:
            num_adds = len(missing_backend_servers)
            logger.info(
                "Not enough free server slots in backend. Add {} slots.".format(num_adds))
            for server_id, backend_server in enumerate(
                    missing_backend_servers, start=total_server_slots + 1):
                server_name = get_default_server_name(server_id)
                add_backend_slot(
                    haproxy_server, backend_name,
                    server_name, backend_server)
        else:
            num_deletes = len(inactive_servers) - HAPROXY_BACKEND_DYNAMIC_FREE_SLOTS
            if num_deletes > 0:
                # if there are more inactive servers, check the spare limit for last server ids
                for server_id in range(total_server_slots,
                                       total_server_slots - num_deletes, -1):
                    server_name = get_default_server_name(server_id)
                    if server_name in inactive_servers:
                        delete_backend_slot(
                            haproxy_server, backend_name,
                            server_name)
                    else:
                        # this make sure only delete the last ones so that
                        break


class DiscoverBackendServers(PullJob):
    """Pulling job for discovering backend targets and update HAProxy using Runtime API"""

    def __init__(self,
                 service_selector=None,
                 backend_name=None,
                 ):
        self.service_selector = deserialize_service_selector(
            service_selector)
        self.backend_name = backend_name

    def pull(self):
        selected_services = self._query_services()
        backend_servers = []
        for service_name in selected_services:
            service_nodes = self._query_service_nodes(service_name)
            # each node is a data source. if many nodes form a load balancer in a cluster
            # it should be filtered by service selector using service name ,tags or labels
            for service_node in service_nodes:
                server_address = get_service_address(service_node)
                backend_servers.append(server_address)
        if not backend_servers:
            logger.warning("No live servers return from the service selector.")
        else:
            _update_backend(self.backend_name, backend_servers)

        # Finally, rebuild the HAProxy configuration for restarts/reloads
        update_configuration(backend_servers)

    def _query_services(self):
        return query_services(self.service_selector)

    def _query_service_nodes(self, service_name):
        return query_service_nodes(service_name, self.service_selector)


class DiscoverAPIGatewayBackendServers(PullJob):
    """Pulling job for discovering backend targets for API gateway backends
    and update HAProxy using Runtime API"""

    def __init__(self,
                 service_selector=None,
                 bind_ip=None,
                 bind_port=None,
                 balance_method=None,
                 ):
        self.service_selector = deserialize_service_selector(
            service_selector)
        self.bind_ip = bind_ip
        self.bind_port = bind_port
        self.balance_method = balance_method
        # TODO: logging the job parameters

    def pull(self):
        selected_services = self._query_services()
        active_backends = _list_backends()
        new_backends = set()
        api_gateway_backends = {}
        for service_name in selected_services:
            service_nodes = self._query_service_nodes(service_name)
            # each node is a data source. if many nodes form a load balancer in a cluster
            # it should be filtered by service selector using service name ,tags or labels

            backend_servers = []
            for service_node in service_nodes:
                server_address = get_service_address(service_node)
                backend_servers.append(server_address)

            # TODO: currently use service_name as backend_name and path prefix for simplicity
            #  future to support more flexible cases
            backend_name = service_name
            if not backend_servers:
                logger.warning("No live servers return from the service selector.")

            if backend_name in active_backends:
                # update only backend servers for active backend
                _update_backend(backend_name, backend_servers)
            else:
                new_backends.add(backend_name)
            api_gateway_backends[service_name] = backend_servers

        # Finally, rebuild the HAProxy configuration for restarts/reloads
        update_api_gateway_configuration(
            api_gateway_backends, new_backends,
            bind_ip=self.bind_ip,
            bind_port=self.bind_port,
            balance_method=self.balance_method)

    def _query_services(self):
        return query_services(self.service_selector)

    def _query_service_nodes(self, service_name):
        return query_service_nodes(service_name, self.service_selector)
