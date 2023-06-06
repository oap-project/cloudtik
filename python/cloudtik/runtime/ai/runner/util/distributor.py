import re
import socket

VALID_HOST_REGEXES = {
    'valid_ip': r"^((25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\.){3}(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])$",  # noqa: E501
    'valid_ip_with_slot': r"^((25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\.){3}(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9]) slots=[0-9]{1,2}$",  # noqa: E501
    'valid_ip_with_colon': r"^((25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\.){3}(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9]):\d{1,2}$",  # noqa: E501
    'valid_hostname': r"^(([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9])\.)*([A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9\-]*[A-Za-z0-9])$",  # noqa: E501
    'valid_hostname_with_slot': r"^(([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9])\.)*([A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9\-]*[A-Za-z0-9]) slots=[0-9]{1,2}$",  # noqa: E501
    'valid_hostname_with_colon': r"^(([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9])\.)*([A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9\-]*[A-Za-z0-9]):\d{1,2}$"  # noqa: E501
}


def parse_hostfile(hostfile):
    """
        Parses the hostfile and returns the required command. Contents of hostfile must contain IP addresses
        (or) hostnames in any of the following forms. Note that all lines must be of the same form:
            "127.0.0.1"
            "127.0.0.1 slots=2"
            "127.0.0.1:2"
            "hostname-example.com"
            "hostname-example.com slots=2"
            "hostname-example.com:2"

        Args:
            hostfile (str): File path of the hostfile

        Returns:
            hostfile_info dict containing ip_addresses and slots

    """
    with open(hostfile, 'r') as f:
        lines = [line.rstrip() for line in f]

    normalized_hosts = []
    for line in lines:
        normalized_host = normalize_host(line)
        normalized_hosts.append(normalized_host)
    return normalized_hosts


def parse_hosts(hosts):
    """
        Parse comma separated hosts list
    """
    lines = [x.strip() for x in hosts.split(",")]
    normalized_hosts = []
    for line in lines:
        normalized_host = normalize_host(line)
        normalized_hosts.append(normalized_host)
    return normalized_hosts


def match_host(host):
    for k, v in VALID_HOST_REGEXES.items():
        if re.match(v, host):
            return k

    raise ValueError("Invalid host in the hosts/hostfile: {}".format(host))


def normalize_host(host):
    match = match_host(host)

    if match == 'valid_ip':
        ip = host
        slots = None
    elif match == 'valid_ip_with_slot':
        parts = host.split(' slots=')
        ip = parts[0]
        slots = int(parts[1])
    elif match == 'valid_ip_with_colon':
        parts = host.split(':')
        ip = parts[0]
        slots = int(parts[1])
    elif match == 'valid_hostname':
        ip = socket.gethostbyname(host)
        slots = None
    elif match == 'valid_hostname_with_slot':
        parts = host.split(' slots=')
        ip = socket.gethostbyname(parts[0])
        slots = int(parts[1])
    elif match == 'valid_hostname_with_colon':
        parts = host.split(':')
        ip = socket.gethostbyname(parts[0])
        slots = int(parts[1])
    else:
        raise ValueError("Invalid host: {}".format(host))

    return {"ip": ip, "slots": slots}


def _sum_of_host_slots(normalized_hosts, validate = True):
    total_slots = 0
    for normalized_host in normalized_hosts:
        slots = normalized_host["slots"]
        if slots is None:
            if validate:
                raise ValueError("No slots defined for host: {}".format(
                    normalized_host["ip"]))
            continue
        total_slots += slots
    return total_slots


def _normalize_host_slots(normalized_hosts, num_proc_per_node, force=False):
    if force:
        for normalized_host in normalized_hosts:
            normalized_host["slots"] = num_proc_per_node
    else:
        # fill in the slots if not specified
        # replace with num_proc_per_node if it is greater
        for normalized_host in normalized_hosts:
            slots = normalized_host["slots"]
            if slots is None:
                normalized_host["slots"] = num_proc_per_node
            elif slots < num_proc_per_node:
                raise ValueError("Not enough host slots.{} {} defined, {} needed.".format(
                    normalized_host["ip"], slots, num_proc_per_node))


def _normalize_num_proc_per_node(num_proc, num_nodes):
    num_proc_per_node = num_proc // num_nodes
    if (num_proc % num_nodes) > 0:
        num_proc_per_node += 1
    return num_proc_per_node


class Distributor:
    def __init__(
            self,
            num_proc=None,
            num_nodes=None,
            num_proc_per_node=None,
            hosts=None,
            hostfile=None,
    ):
        # nodes and processes
        # If num_nodes and num_proc_per_node is specified
        # hosts/hostfile can be host address only without slots
        # Or you can specify hosts with slots and specify the num_proc
        self._num_proc = num_proc
        self._num_nodes = num_nodes
        self._num_proc_per_node = num_proc_per_node

        # host arguments
        self._hosts = hosts
        self._hostfile = hostfile

        self.normalized_num_proc = self._num_proc
        self.normalized_num_nodes = self._num_nodes
        self.normalized_num_proc_per_node = self._num_proc_per_node
        self.normalized_hosts = None

        self.normalize()

    def normalize(self):
        if self._hostfile and self._hosts:
            raise ValueError("Can only specify one of the options: hosts and hostfile.")

        if self._hostfile:
            self.normalized_hosts = parse_hostfile(self._hostfile)
        elif self._hosts:
            self.normalized_hosts = parse_hosts(self._hosts)

        if self._num_nodes and self._num_nodes > 1 and not self.normalized_hosts:
            raise ValueError("Must specify hosts and hostfile for distributed.")

        if not self.normalized_hosts:
            # local single node
            if self._num_proc and self._num_proc_per_node and self._num_proc != self._num_proc_per_node:
                raise ValueError(
                    "Inconsistent parameter value. num_proc: {}, num_proc_per_node: {}".format(
                        self._num_proc, self._num_proc_per_node))
            if self._num_proc_per_node:
                self.normalized_num_proc = self._num_proc_per_node
            elif self._num_proc:
                self.normalized_num_proc = self._num_proc
            else:
                # This is the case to resolve later with num_proc_per_node
                self.normalized_num_proc = 0

            self.normalized_num_nodes = 1
        else:
            # normalize num_nodes
            num_nodes_defined = len(self.normalized_hosts)
            if not self._num_nodes:
                self.normalized_num_nodes = num_nodes_defined
            elif self._num_nodes > num_nodes_defined:
                raise ValueError("{} nodes defined in hosts/hostfile, but {} nodes needed.".format(
                    num_nodes_defined, self._num_nodes))
            else:
                self.normalized_num_nodes = self._num_nodes

            if self._num_proc:
                # if num_proc defined, use it
                self.normalized_num_proc = self._num_proc
                if self._num_proc_per_node:
                    _normalize_host_slots(
                        self.normalized_hosts, self._num_proc_per_node)
                else:
                    num_proc_per_node = _normalize_num_proc_per_node(
                        self.normalized_num_proc, self.normalized_num_nodes)
                    _normalize_host_slots(
                        self.normalized_hosts, num_proc_per_node)

                # make sure the slots is enough
                all_slots = _sum_of_host_slots(
                    self.normalized_hosts)
                if all_slots < self._num_proc:
                    raise ValueError("No enough slots. {} defined. {} needed.".format(
                        all_slots, self.normalized_num_proc))
            else:
                # if num_proc not defined, calculate from
                # 1. if num_proc_per_node is given
                # 2. if hosts defined slots, calculate from the sum of slots
                if self._num_proc_per_node:
                    self.normalized_num_proc = self._num_proc_per_node * self.normalized_num_nodes
                    _normalize_host_slots(self.normalized_hosts, self._num_proc_per_node)
                else:
                    # try get from host slots
                    all_slots = _sum_of_host_slots(
                        self.normalized_hosts, False)
                    # if all_slots == 0
                    # This is the case that we need to resolve later
                    self.normalized_num_proc = all_slots
                    # for this case, num_proc may not dividable by num_nodes

        self.normalized_num_proc_per_node = _normalize_num_proc_per_node(
            self.normalized_num_proc, self.normalized_num_nodes)

    @property
    def resolved(self):
        return True if self.normalized_num_proc else False

    def resolve(self, num_proc_per_node=1, force=False):
        if force or not self.normalized_num_proc:
            self.normalized_num_proc = self.normalized_num_nodes * num_proc_per_node
            self.normalized_num_proc_per_node = num_proc_per_node
            _normalize_host_slots(
                self.normalized_hosts, num_proc_per_node, force=force)

    def _check_resolved(self):
        if not self.resolved:
            raise ValueError(
                "There are unresolved parameters. Call resolve with num_proc_per_node.")

    def validate_same_slots(self):
        if not self.hosts:
            return
        slots = self.hosts[0]["slots"]
        for host in self.hosts:
            if slots != host["slots"]:
                raise ValueError("Host with different slots found.")

    @property
    def distributed(self):
        return self.normalized_num_nodes > 1 or self.normalized_hosts

    @property
    def num_proc(self):
        self._check_resolved()
        return self.normalized_num_proc

    @property
    def num_nodes(self):
        return self.normalized_num_nodes

    @property
    def num_proc_per_node(self):
        self._check_resolved()
        return self.normalized_num_proc_per_node

    @property
    def hosts(self):
        self._check_resolved()
        return self.normalized_hosts

    @property
    def hosts_slots_str(self):
        _hosts_slots = ["{}:{}".format(
            host["ip"], host["slots"]) for host in self.hosts]
        return ",".join(_hosts_slots)

    @property
    def hosts_str(self):
        _hosts = [host["ip"] for host in self.hosts]
        return ",".join(_hosts)
