import logging
import socket

from cloudtik.core._private.service_discovery.utils import deserialize_service_selector
from cloudtik.core._private.util.pull.pull_job import PullJob
from cloudtik.runtime.common.service_discovery.consul \
    import query_services, query_service_nodes, get_service_address
from cloudtik.runtime.haproxy.utils import update_configuration, get_backend_server_name

logger = logging.getLogger(__name__)


HAPROXY_SERVERS = [("127.0.0.1", 19999)]


def send_haproxy_command(haproxy_server, command):
    if haproxy_server[0] == "/":
        haproxy_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    else:
        haproxy_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    haproxy_sock.settimeout(10)
    try:
        haproxy_sock.connect(haproxy_server)
        haproxy_sock.send(command)
        retval = ""
        while True:
            buf = haproxy_sock.recv(16)
            if buf:
                retval += buf
            else:
                break
        haproxy_sock.close()
    except:
        retval = ""
    finally:
        haproxy_sock.close()
    return retval


def _parse_server_slots(backend_name, stat_string):
    server_slots = stat_string.split('\n')
    active_servers = {}
    inactive_servers = []
    for server_slot in server_slots:
        server_slot_values = server_slot.split(",")
        if len(server_slot_values) > 80 and server_slot_values[0] == backend_name:
            server_name = server_slot_values[1]
            if server_name == "BACKEND":
                continue
            server_state = server_slot_values[17]
            server_addr = server_slot_values[73]
            if server_state == "MAINT":
                # Any server in MAINT is assumed to be unconfigured and
                # free to use (to stop a server for your own work try 'DRAIN' for the script to just skip it)
                inactive_servers.append(server_name)
            else:
                active_servers[server_addr] = server_name
    return active_servers, inactive_servers


def _enable_backend_slot(
        haproxy_server, backend_name, server_name, backend_server):
    send_haproxy_command(
        haproxy_server, "set server %s/%s addr %s port %s\n" % (
            backend_name, server_name,
            backend_server[0], backend_server[1]))
    send_haproxy_command(
        haproxy_server,
        "set server %s/%s state ready\n" % (
            backend_name, server_name))


def _disable_backend_slot(
        haproxy_server, backend_name, server_name):
    send_haproxy_command(
        haproxy_server,
        "set server %s/%s state maint\n" % (
            backend_name, server_name))


def _add_backend_slot(
        haproxy_server, backend_name, server_name, backend_server):
    # add a new dynamic server with address and enable it
    send_haproxy_command(
        haproxy_server,
        "add server %s/%s %s:%s check enabled\n" % (
            backend_name, server_name,
            backend_server[0], backend_server[1]))


def _add_disabled_backend_slot(
        haproxy_server, backend_name, server_name):
    # add a new dynamic server without address and disable it
    send_haproxy_command(
        haproxy_server,
        "add server %s/%s 0.0.0.0:80 check disabled\n" % (
            backend_name, server_name))


def _update_backend(backend_name, backend_servers):
    # Now update each HAProxy server with the backends
    for haproxy_server in HAPROXY_SERVERS:
        stat_string = send_haproxy_command(haproxy_server, "show stat\n")
        if not stat_string:
            raise RuntimeError("Failed to get current backend list from HAProxy socket.")

        active_servers, inactive_servers = _parse_server_slots(
            backend_name, stat_string)
        total_server_slots = len(active_servers) + len(inactive_servers)

        missing_backend_servers = []
        for backend_server in backend_servers:
            server_address = "%s:%s" % (backend_server[0], backend_server[1])
            if server_address in active_servers:
                # Ignore backends already set
                server_name = active_servers[server_address]
                del active_servers[server_address]
            else:
                if len(inactive_servers) > 0:
                    server_name = inactive_servers.pop(0)
                    _enable_backend_slot(
                        haproxy_server, backend_name,
                        server_name, backend_server)
                else:
                    # we need a reload of the configuration after
                    missing_backend_servers.append(server_address)

        # mark inactive for remaining servers in active servers set but not appearing
        for remaining_server, server_name in active_servers.items():
            # disable
            _disable_backend_slot(
                haproxy_server, backend_name, server_name)
            # if there are missing backend, use it
            if missing_backend_servers:
                backend_server = missing_backend_servers.pop(0)
                _enable_backend_slot(
                    haproxy_server, backend_name,
                    server_name, backend_server)

        if missing_backend_servers:
            num = len(missing_backend_servers)
            logger.info(
                "Not enough free server slots in backend. Adding {} slots.".format(num))
            for slot_id, backend_server in enumerate(
                    missing_backend_servers, start=total_server_slots + 1):
                server_name = get_backend_server_name(slot_id)
                _add_backend_slot(
                    haproxy_server, backend_name,
                    server_name, backend_server)


class DiscoverBackendServers(PullJob):
    """Pulling job for discovering backend targets and update HAProxy using Runtime API"""

    def __init__(self,
                 service_selector=None,
                 backend_name=None):
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
