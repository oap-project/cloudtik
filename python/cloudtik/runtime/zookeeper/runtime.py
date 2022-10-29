import logging
from typing import Any, Dict, Optional, List

from cloudtik.core.node_provider import NodeProvider
from cloudtik.core.runtime import Runtime
from cloudtik.runtime.zookeeper.utils import _config_runtime_resources, _with_runtime_environment_variables, \
    _is_runtime_scripts, _get_runnable_command, _get_runtime_processes, _validate_config, \
    _verify_config, _get_runtime_logs, _get_runtime_commands, \
    _get_defaults_config, _get_runtime_services, _handle_minimal_nodes_reached

logger = logging.getLogger(__name__)


class ZooKeeperRuntime(Runtime):
    """Implementation for ZooKeeper Runtime"""

    def __init__(self, runtime_config: Dict[str, Any]) -> None:
        Runtime.__init__(self, runtime_config)

    def prepare_config(self, cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare runtime specific configurations"""
        return _config_runtime_resources(cluster_config)

    def validate_config(self, cluster_config: Dict[str, Any], provider: NodeProvider):
        """Validate cluster configuration from runtime perspective."""
        _validate_config(cluster_config, provider)

    def verify_config(self, cluster_config: Dict[str, Any], provider: NodeProvider):
        """Verify cluster configuration at the last stage of bootstrap.
        The verification may mean a slow process to check with a server"""
        _verify_config(cluster_config, provider)

    def with_environment_variables(
            self, config: Dict[str, Any], provider: NodeProvider,
            node_id: str) -> Dict[str, Any]:
        """Export necessary runtime environment variables for running node commands.
        For example: {"ENV_NAME": value}
        """
        return _with_runtime_environment_variables(
            self.runtime_config, config=config, provider=provider, node_id=node_id)

    def cluster_booting_completed(
            self, cluster_config: Dict[str, Any], head_node_id: str) -> None:
        # Do nothing here. service uri will only be published when minimal node reached.
        pass

    def get_runnable_command(self, target: str, runtime_options: Optional[List[str]]):
        """Return the runnable command for the target script.
        For example: ["bash", target]
        """
        if not _is_runtime_scripts(target):
            return None

        return _get_runnable_command(target)

    def get_runtime_commands(self, cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a copy of runtime commands to run at different stages"""
        return _get_runtime_commands(self.runtime_config, cluster_config)

    def get_defaults_config(self, cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a copy of runtime config"""
        return _get_defaults_config(self.runtime_config, cluster_config)

    def get_runtime_services(self, cluster_head_ip: str):
        return _get_runtime_services(cluster_head_ip)

    def require_minimal_nodes(self, cluster_config: Dict[str, Any]) -> bool:
        """Whether the runtime nodes need minimal nodes launch before going to setup.
        Usually this is because the setup of the nodes need to know each other.
        """
        return True

    def minimal_nodes_reached(self, cluster_config: Dict[str, Any], nodes_info: Dict[str, Any]):
        """If the require_minimal_nodes method returns True and runtime will be notified on head
        When the minimal nodes are reached. Please note this may call multiple times (for example server down and up)
        """
        _handle_minimal_nodes_reached(self.runtime_config, cluster_config, nodes_info)

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
        For example
        ["cloudtik_cluster_controller.py", False, "ClusterController", "head"],
        """
        return _get_runtime_processes()
