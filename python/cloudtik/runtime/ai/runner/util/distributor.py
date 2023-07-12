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


def _normalize_host_slots(normalized_hosts, nproc_per_node, force=False):
    if not normalized_hosts:
        return

    if force:
        for normalized_host in normalized_hosts:
            normalized_host["slots"] = nproc_per_node
    else:
        # fill in the slots if not specified
        # replace with nproc_per_node if it is greater
        for normalized_host in normalized_hosts:
            slots = normalized_host["slots"]
            if slots is None:
                normalized_host["slots"] = nproc_per_node
            elif slots < nproc_per_node:
                raise ValueError("Not enough host slots.{} {} defined, {} needed.".format(
                    normalized_host["ip"], slots, nproc_per_node))


def _normalize_nproc_per_node(num_proc, nnodes):
    nproc_per_node = num_proc // nnodes
    if (num_proc % nnodes) > 0:
        nproc_per_node += 1
    return nproc_per_node


class Distributor:
    def __init__(
            self,
            num_proc=None,
            nnodes=None,
            nproc_per_node=None,
            hosts=None,
            hostfile=None,
    ):
        # nodes and processes
        # If nnodes and nproc_per_node is specified
        # hosts/hostfile can be host address only without slots
        # Or you can specify hosts with slots and specify the num_proc
        self._num_proc = num_proc
        self._nnodes = nnodes
        self._nproc_per_node = nproc_per_node

        # host arguments
        self._hosts = hosts
        self._hostfile = hostfile

        self.normalized_num_proc = self._num_proc
        self.normalized_nnodes = self._nnodes
        self.normalized_nproc_per_node = self._nproc_per_node
        self.normalized_hosts = None

        self.normalize()

    def normalize(self):
        if self._hostfile and self._hosts:
            raise ValueError("Can only specify one of the options: hosts and hostfile.")

        if self._hostfile:
            self.normalized_hosts = parse_hostfile(self._hostfile)
        elif self._hosts:
            self.normalized_hosts = parse_hosts(self._hosts)

        if self._nnodes and self._nnodes > 1 and not self.normalized_hosts:
            raise ValueError("Must specify hosts and hostfile for distributed.")

        if not self.normalized_hosts:
            # local single node
            if self._num_proc and self._nproc_per_node and self._num_proc != self._nproc_per_node:
                raise ValueError(
                    "Inconsistent parameter value. num_proc: {}, nproc_per_node: {}".format(
                        self._num_proc, self._nproc_per_node))
            if self._nproc_per_node:
                self.normalized_num_proc = self._nproc_per_node
            elif self._num_proc:
                self.normalized_num_proc = self._num_proc
            else:
                # This is the case to resolve later with nproc_per_node
                self.normalized_num_proc = 0

            self.normalized_nnodes = 1
        else:
            # normalize nnodes
            nnodes_defined = len(self.normalized_hosts)
            if not self._nnodes:
                self.normalized_nnodes = nnodes_defined
            elif self._nnodes > nnodes_defined:
                raise ValueError("{} nodes defined in hosts/hostfile, but {} nodes needed.".format(
                    nnodes_defined, self._nnodes))
            else:
                self.normalized_nnodes = self._nnodes

            if self._num_proc:
                # if num_proc defined, use it
                self.normalized_num_proc = self._num_proc
                if self._nproc_per_node:
                    _normalize_host_slots(
                        self.normalized_hosts, self._nproc_per_node)
                else:
                    nproc_per_node = _normalize_nproc_per_node(
                        self.normalized_num_proc, self.normalized_nnodes)
                    _normalize_host_slots(
                        self.normalized_hosts, nproc_per_node)

                # make sure the slots is enough
                all_slots = _sum_of_host_slots(
                    self.normalized_hosts)
                if all_slots < self._num_proc:
                    raise ValueError("No enough slots. {} defined. {} needed.".format(
                        all_slots, self.normalized_num_proc))
            else:
                # if num_proc not defined, calculate from
                # 1. if nproc_per_node is given
                # 2. if hosts defined slots, calculate from the sum of slots
                if self._nproc_per_node:
                    self.normalized_num_proc = self._nproc_per_node * self.normalized_nnodes
                    _normalize_host_slots(self.normalized_hosts, self._nproc_per_node)
                else:
                    # try get from host slots
                    all_slots = _sum_of_host_slots(
                        self.normalized_hosts, False)
                    # if all_slots == 0
                    # This is the case that we need to resolve later
                    self.normalized_num_proc = all_slots
                    # for this case, num_proc may not dividable by nnodes

        self.normalized_nproc_per_node = _normalize_nproc_per_node(
            self.normalized_num_proc, self.normalized_nnodes)

    @property
    def resolved(self):
        return True if self.normalized_num_proc else False

    def resolve(self, nproc_per_node=1, force=False):
        if force or not self.normalized_num_proc:
            self.normalized_num_proc = self.normalized_nnodes * nproc_per_node
            self.normalized_nproc_per_node = nproc_per_node
            _normalize_host_slots(
                self.normalized_hosts, nproc_per_node, force=force)

    def _check_resolved(self):
        if not self.resolved:
            raise ValueError(
                "There are unresolved parameters. "
                "Pass proper parameters or call resolve with nproc_per_node.")

    def validate_same_slots(self):
        if not self.hosts:
            return
        slots = self.hosts[0]["slots"]
        for host in self.hosts:
            if slots != host["slots"]:
                raise ValueError("Host with different slots found.")

    def export_host_file(self, host_file, with_slots=True):
        if self.hosts is None:
            raise ValueError("No hosts/hostfile parameter passed.")
        # write file to contain the IP addresses and slots
        with open(host_file, 'w') as f:
            for host in self.hosts:
                addr = "{}:{}".format(
                    host["ip"], host["slots"]) if with_slots else host["ip"]
                f.write(addr + '\n')

    @property
    def distributed(self):
        return self.normalized_nnodes > 1 or self.normalized_hosts

    @property
    def num_proc(self):
        self._check_resolved()
        return self.normalized_num_proc

    @property
    def nnodes(self):
        return self.normalized_nnodes

    @property
    def nproc_per_node(self):
        self._check_resolved()
        return self.normalized_nproc_per_node

    @property
    def hosts(self):
        self._check_resolved()
        return self.normalized_hosts

    @property
    def hosts_slots_str(self):
        if self.hosts is None:
            raise ValueError("No hosts/hostfile parameter passed.")

        _hosts_slots = ["{}:{}".format(
            host["ip"], host["slots"]) for host in self.hosts]
        return ",".join(_hosts_slots)

    @property
    def hosts_str(self):
        if self.hosts is None:
            raise ValueError("No hosts/hostfile parameter passed.")
        _hosts = [host["ip"] for host in self.hosts]
        return ",".join(_hosts)
