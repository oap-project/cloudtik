import json
import logging
import os
import time
from typing import Dict, List

from cloudtik.core._private.constants import CLOUDTIK_HEARTBEAT_TIMEOUT_S
from cloudtik.core._private.runtime_utils import save_yaml
from cloudtik.core._private.state.control_state import ControlState
from cloudtik.core._private.state.state_utils import NODE_STATE_NODE_IP, \
    NODE_STATE_HEARTBEAT_TIME, NODE_STATE_NODE_TYPE
from cloudtik.core._private.util.pull.pull_job import PullJob
from cloudtik.runtime.prometheus.utils import _get_home_dir

logger = logging.getLogger(__name__)


def _parse_services(service_list_str) -> Dict[str, List[str]]:
    services = {}
    if not service_list_str:
        return services

    service_list = [x.strip() for x in service_list_str.split(",")]
    for service_str in service_list:
        service_parts = [x.strip() for x in service_str.split(":")]
        if len(service_parts) < 2:
            raise ValueError(
                "Invalid service specification. "
                "Format: service_name:node_type_1:node_type_2")
        service_name = service_parts[0]
        service_node_types = service_parts[1:]
        services[service_name] = service_node_types
    return services


def _get_service_targets(
        service_name, service_node_types, live_nodes_by_node_type):
    targets = _get_targets_of_node_types(
        live_nodes_by_node_type, service_node_types)
    if not targets:
        return None

    service_targets = {
        "labels": {
            "service": service_name
        },
        "targets": targets
    }
    return service_targets


def _get_targets_of_node_types(live_nodes_by_node_type, node_types):
    if len(node_types) == 1:
        return live_nodes_by_node_type.get(node_types[0])
    else:
        # more than one node types
        nodes = []
        for node_type in node_types:
            nodes += live_nodes_by_node_type.get(node_type, [])
        return nodes


class PullLocalTargets(PullJob):
    """Pulling job for local cluster nodes if service discovery is not available"""

    def __init__(self,
                 services=None,
                 redis_address=None,
                 redis_password=None):
        if not redis_address:
            raise RuntimeError("Radis address is needed for pulling local targets.")

        (redis_ip, redis_port) = redis_address.split(":")

        self.services = _parse_services(services)
        self.redis_address = redis_address
        self.redis_password = redis_password

        self.control_state = ControlState()
        self.control_state.initialize_control_state(
            redis_ip, redis_port, redis_password)
        self.node_table = self.control_state.get_node_table()

        home_dir = _get_home_dir()
        self.config_file = os.path.join(home_dir, "conf", "local-targets.yaml")

    def _get_live_nodes(self):
        live_nodes_by_node_type = {}
        now = time.time()
        nodes_state_as_json = self.node_table.get_all().values()
        for node_state_as_json in nodes_state_as_json:
            node_state = json.loads(node_state_as_json)
            # Filter out the stale record in the node table
            delta = now - node_state.get(NODE_STATE_HEARTBEAT_TIME, 0)
            if delta < CLOUDTIK_HEARTBEAT_TIMEOUT_S:
                node_type = node_state[NODE_STATE_NODE_TYPE]
                if node_type not in live_nodes_by_node_type:
                    live_nodes_by_node_type[node_type] = []
                nodes_of_node_type = live_nodes_by_node_type[node_type]
                nodes_of_node_type.append(node_state[NODE_STATE_NODE_IP])
        return live_nodes_by_node_type

    def pull(self):
        live_nodes_by_node_type = self._get_live_nodes()

        local_targets = []
        for service_name, service_node_types in self.services.items():
            service_targets = _get_service_targets(
                service_name, service_node_types, live_nodes_by_node_type)
            if service_targets:
                local_targets.append(service_targets)
        save_yaml(self.config_file, local_targets)
