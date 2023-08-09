import socket


def get_backend_server_address(backend_server):
    return "{}:{}".format(backend_server[0], backend_server[1])


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
                # Any server in MAINT is assumed to be un-configured and
                # free to use (to stop a server for your own work try 'DRAIN'
                # for the script to just skip it)
                inactive_servers.append(server_name)
            else:
                active_servers[server_addr] = server_name
    return active_servers, inactive_servers


def send_haproxy_command(haproxy_server, command):
    if haproxy_server[0] == "/":
        haproxy_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    else:
        haproxy_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    haproxy_sock.settimeout(10)
    try:
        haproxy_sock.connect(haproxy_server)
        haproxy_sock.send(command.encode("utf-8"))
        retval = ""
        while True:
            buf = haproxy_sock.recv(16)
            if buf:
                retval += buf.decode("utf-8")
            else:
                break
    finally:
        haproxy_sock.close()
    return retval


def list_backend_servers(haproxy_server, backend_name):
    stat_string = send_haproxy_command(haproxy_server, "show stat\n")
    if not stat_string:
        raise RuntimeError("Failed to get current backend list from HAProxy socket.")

    return _parse_server_slots(
        backend_name, stat_string)


def add_backend_slot(
        haproxy_server, backend_name, server_name, backend_server):
    # add a new dynamic server with address and enable it
    send_haproxy_command(
        haproxy_server,
        "add server %s/%s %s check enabled\n" % (
            backend_name, server_name,
            get_backend_server_address(backend_server)))


def add_disabled_backend_slot(
        haproxy_server, backend_name, server_name):
    # add a new dynamic server without address and disable it
    send_haproxy_command(
        haproxy_server,
        "add server %s/%s 0.0.0.0:80 check disabled\n" % (
            backend_name, server_name))


def del_backend_slot(
        haproxy_server, backend_name, server_name):
    # delete a dynamic server (the server need to be in maintenance mode)
    send_haproxy_command(
        haproxy_server,
        "del server %s/%s\n" % (
            backend_name, server_name))


def enable_backend_slot(
        haproxy_server, backend_name, server_name, backend_server):
    send_haproxy_command(
        haproxy_server, "set server %s/%s addr %s port %s\n" % (
            backend_name, server_name,
            backend_server[0], backend_server[1]))
    send_haproxy_command(
        haproxy_server,
        "set server %s/%s state ready\n" % (
            backend_name, server_name))


def disable_backend_slot(
        haproxy_server, backend_name, server_name):
    send_haproxy_command(
        haproxy_server,
        "set server %s/%s state maint\n" % (
            backend_name, server_name))
