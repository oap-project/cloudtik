from typing import Any, Dict

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["gmetad", True, "GangliaMeta", "head"],
    ["gmond", True, "GangliaMonitor", "node"],
]


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _get_head_service_urls(cluster_head_ip):
    services = {
        "ganglia-web": {
            "name": "Ganglia Web UI",
            "url": "http://{}/ganglia".format(cluster_head_ip)
        },
    }
    return services


def _get_head_service_ports(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    service_ports = {
        "ganglia-web": {
            "protocol": "TCP",
            "port": 80,
        },
    }
    return service_ports
