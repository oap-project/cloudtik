import logging
from typing import Any, Dict, Tuple, Optional

from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_NONE
from cloudtik.core.node_provider import NodeProvider
from cloudtik.runtime.common.runtime_base import RuntimeBase
from cloudtik.runtime.consul.utils import _with_runtime_environment_variables, \
    _get_runtime_processes, _get_runtime_logs, _get_runtime_endpoints, _handle_node_constraints_reached, \
    _is_agent_server_mode, _get_head_service_ports, _bootstrap_join_list, _bootstrap_runtime_services

logger = logging.getLogger(__name__)


class ConsulRuntime(RuntimeBase):
    """Implementation for Consul Runtime"""

    def __init__(self, runtime_config: Dict[str, Any]) -> None:
        super().__init__(runtime_config)
        self.server_mode = _is_agent_server_mode(runtime_config)

    def bootstrap_config(self, cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        if not self.server_mode:
            # for client mode, we bootstrap the consul server cluster
            cluster_config = _bootstrap_join_list(cluster_config)

        # collect the runtime services information
        cluster_config = _bootstrap_runtime_services(cluster_config)
        return cluster_config

    def with_environment_variables(
            self, config: Dict[str, Any], provider: NodeProvider,
            node_id: str) -> Dict[str, Any]:
        """Export necessary runtime environment variables for running node commands.
        For example: {"ENV_NAME": value}
        """
        return _with_runtime_environment_variables(
            self.server_mode, self.runtime_config,
            config=config)

    def get_runtime_endpoints(self, cluster_head_ip: str):
        return _get_runtime_endpoints(
            self.server_mode, cluster_head_ip)

    def get_head_service_ports(self) -> Dict[str, Any]:
        return _get_head_service_ports(
            self.server_mode, self.runtime_config)

    def get_node_constraints(
            self, cluster_config: Dict[str, Any]) -> Tuple[bool, bool, bool]:
        """Whether the runtime nodes need minimal nodes launch before going to setup.
        Usually this is because the setup of the nodes need to know each other.
        """
        if self.server_mode:
            return True, True, True
        else:
            return False, False, False

    def node_constraints_reached(
            self, cluster_config: Dict[str, Any], node_type: str,
            head_info: Dict[str, Any], nodes_info: Dict[str, Any],
            quorum_id: Optional[str] = None):
        """If the get_node_constraints method returns True and runtime will be notified on head
        When the minimal nodes are reached. Please note this may call multiple times (for example server down and up)
        """
        if self.server_mode:
            _handle_node_constraints_reached(
                self.runtime_config, cluster_config,
                node_type, head_info, nodes_info)

    @staticmethod
    def get_logs() -> Dict[str, str]:
        """Return a dictionary of name to log paths.
        For example {"server-a": "/tmp/server-a/logs"}
        """
        return _get_runtime_logs()

    @staticmethod
    def get_processes():
        """Return a list of processes for this runtime.
        Format:
        #1 Keyword to filter,
        #2 filter by command (True)/filter by args (False)
        #3 The third element is the process name.
        #4 The forth element, if node, the process should on all nodes, if head, the process should on head node.
        """
        return _get_runtime_processes()

    @staticmethod
    def get_dependencies():
        return [BUILT_IN_RUNTIME_NONE]
