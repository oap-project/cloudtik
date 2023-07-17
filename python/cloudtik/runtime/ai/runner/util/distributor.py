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
                raise ValueError("Not enough host slots: {} {} defined, {} needed.".format(
                    normalized_host["ip"], slots, nproc_per_node))


def _normalize_nproc_per_node(num_proc, nnodes):
    if not num_proc or not nnodes:
        return 0
    nproc_per_node = num_proc // nnodes
    if (num_proc % nnodes) > 0:
        nproc_per_node += 1
    return nproc_per_node


def _normalize_nnodes(num_proc, nproc_per_node):
    nnodes = num_proc // nproc_per_node
    if (num_proc % nproc_per_node) > 0:
        nnodes += 1
    return nnodes


class Distributor:
    def __init__(
            self,
            num_proc=0,
            nnodes=0,
            nproc_per_node=0,
            hosts=None,
            hostfile=None,
    ):
        """
        The distributor class handles the process numbers and distribution.

        If nnodes and nproc_per_node is specified,
        hosts/hostfile can be host address only without slots.
        Or you can specify hosts with slots and specify the num_proc.

        For the cases that there is resource scheduler such as Spark or Ray,
        hosts/hostfile is not needed for a distributed run.
        """
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

        if not self.normalized_hosts:
            # local or resource scheduler case: nothing is a compulsory
            # just do best effort resolving

            # for the cases there is a resource scheduler,
            # num_proc or nnodes + nproc_per_node is enough
            if self._num_proc:
                if self._nnodes and self._nproc_per_node and (
                        self._num_proc != self._nnodes * self._nproc_per_node):
                    raise ValueError(
                        "Inconsistent values. num_proc: {}, nnodes: {}, nproc_per_node: {}".format(
                            self._num_proc, self._nnodes, self._nproc_per_node))
                self.normalized_num_proc = self._num_proc  # this line already done in init
                if self._nnodes:
                    self.normalized_nnodes = self._nnodes
                elif self._nproc_per_node:
                    self.normalized_nnodes = _normalize_nnodes(
                        self._num_proc, self._nproc_per_node)
                # normalized_nnodes is still possible to be None or 0
            elif self._nnodes and self._nproc_per_node:
                self.normalized_num_proc = self._nnodes * self._nproc_per_node
                self.normalized_nnodes = self._nnodes
            # cases remaining such as self._nnodes or self._nproc_per_node or nothing
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
        return True if (self.normalized_num_proc and self.normalized_nnodes) else False

    def resolve(self, nproc_per_node=1, nnodes=None, force=False, check=False):
        if force or not self.resolved:
            if self.normalized_nnodes:
                self.normalized_num_proc = self.normalized_nnodes * nproc_per_node
            elif self.normalized_num_proc:
                self.normalized_nnodes = _normalize_nnodes(
                    self.normalized_num_proc, nproc_per_node)
            else:
                # for the case that both num_proc and nnodes are not specified
                if nnodes:
                    self.normalized_nnodes = nnodes
                    self.normalized_num_proc = self.normalized_nnodes * nproc_per_node
                elif check:
                    self.check_resolved()
                # may still leave not resolved
            self.normalized_nproc_per_node = nproc_per_node
            _normalize_host_slots(
                self.normalized_hosts, nproc_per_node, force=force)

    def check_resolved(self):
        if not self.resolved:
            raise ValueError(
                "There are unresolved parameters. "
                "Pass proper parameters or call resolve with nproc_per_node and/or nnodes.")

    def check_distributed_with_hosts(self):
        # This checks the hosts based distributed
        if not self.distributed_with_hosts and self.normalized_nnodes and self.normalized_nnodes > 1:
            raise ValueError(
                "No hosts/hostfile specified for running with multiple nodes.")

    def check_same_slots(self):
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
    def distributed_with_hosts(self):
        return True if self.normalized_hosts else False

    @property
    def num_proc(self):
        return self.normalized_num_proc

    @property
    def nnodes(self):
        return self.normalized_nnodes

    @property
    def nproc_per_node(self):
        return self.normalized_nproc_per_node

    @property
    def hosts(self):
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

    @property
    def origin_num_proc(self):
        return self._num_proc

    @property
    def origin_nnodes(self):
        return self._nnodes

    @property
    def origin_nproc_per_node(self):
        return self._nproc_per_node
