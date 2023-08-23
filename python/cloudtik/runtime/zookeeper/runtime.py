import logging
from typing import Any, Dict, Tuple, Optional

from cloudtik.core.node_provider import NodeProvider
from cloudtik.runtime.common.runtime_base import RuntimeBase
from cloudtik.runtime.zookeeper.utils import _with_runtime_environment_variables, \
    _get_runtime_processes, _get_runtime_logs, _get_runtime_endpoints, _handle_node_constraints_reached, \
    _get_runtime_services

logger = logging.getLogger(__name__)


class ZooKeeperRuntime(RuntimeBase):
    """Implementation for ZooKeeper Runtime"""

    def __init__(self, runtime_config: Dict[str, Any]) -> None:
        super().__init__(runtime_config)

    def with_environment_variables(
            self, config: Dict[str, Any], provider: NodeProvider,
            node_id: str) -> Dict[str, Any]:
        """Export necessary runtime environment variables for running node commands.
        For example: {"ENV_NAME": value}
        """
        return _with_runtime_environment_variables(
            self.runtime_config, config=config, provider=provider, node_id=node_id)

    def get_runtime_endpoints(self, cluster_head_ip: str):
        return _get_runtime_endpoints(cluster_head_ip)

    def get_runtime_services(self, cluster_name: str):
        return _get_runtime_services(self.runtime_config, cluster_name)

    def get_node_constraints(
            self, cluster_config: Dict[str, Any]) -> Tuple[bool, bool, bool]:
        """Whether the runtime nodes need minimal nodes launch before going to setup.
        Usually this is because the setup of the nodes need to know each other.
        """
        return True, True, True

    def node_constraints_reached(
            self, cluster_config: Dict[str, Any], node_type: str,
            head_info: Dict[str, Any], nodes_info: Dict[str, Any],
            quorum_id: Optional[str] = None):
        """If the get_node_constraints method returns True and runtime will be notified on head
        When the minimal nodes are reached. Please note this may call multiple times (
        for example server down and up)
        """
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
        #4 The forth element, if node, the process should on all nodes,
        if head, the process should on head node.
        """
        return _get_runtime_processes()
