import re
import shutil
import tempfile

SYSTEM_RESOLV_CONF = "/etc/resolv.conf"


def update_resolv_conf(name_servers, resolv_conf=None):
    if not resolv_conf:
        resolv_conf = SYSTEM_RESOLV_CONF

    old_name_servers, search, sort_list, options = parse_resolv_conf(
        resolv_conf)

    # write to temp file first
    resolv_conf_temp = tempfile.mktemp(prefix=f"resolv.conf_")
    with open(resolv_conf_temp, "w") as f:
        for name_server in name_servers:
            f.write(f'nameserver {name_server}\n')
        if search:
            search_list_string = " ".join(search)
            f.write(f'search {search_list_string}\n')
        if sort_list:
            sort_list_string = " ".join(sort_list)
            f.write(f'sortlist {sort_list_string}\n')
        if options:
            options_string = " ".join(options)
            f.write(f'options {options_string}\n')

    # move overwritten
    shutil.move(resolv_conf_temp, resolv_conf)


def get_resolv_conf_name_servers(resolv_conf):
    if not resolv_conf:
        resolv_conf = SYSTEM_RESOLV_CONF

    name_servers, _, _, _ = parse_resolv_conf(resolv_conf)
    return name_servers


def parse_resolv_conf(resolv_conf):
    name_servers = []
    search = None
    sort_list = None
    options = None
    with open(resolv_conf, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("#"):
                continue
            line = line.rstrip()
            if not line:
                continue

            tokens = re.split(' |\t', line)
            if not tokens:
                continue
            if tokens[0] == "nameserver":
                if len(tokens) != 2:
                    continue
                name_servers.append(tokens[1])
            elif tokens[0] == "search" or tokens[0] == "domain":
                search = tokens[1:]
            elif tokens[0] == "sortlist":
                sort_list = tokens[1:]
            elif tokens[0] == "options":
                options = tokens[1:]
    return name_servers, search, sort_list, options
