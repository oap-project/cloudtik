import logging
from typing import Any, Dict

from cloudtik.core.node_provider import NodeProvider

logger = logging.getLogger(__name__)


class Runtime:
    """Interface for runtime abstraction.

    **Important**: This is an INTERNAL API that is only exposed for the purpose
    of implementing custom runtime. It is not allowed to call into
    RuntimeProvider methods from any package outside, only to
    define new implementations of RuntimeProvider for use with the "external" runtime
    provider option.
    """

    def __init__(self, runtime_config: Dict[str, Any]) -> None:
        self.runtime_config = runtime_config

    def prepare_config(self, cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare runtime specific configurations"""
        return cluster_config

    def validate_config(self, cluster_config: Dict[str, Any], provider: NodeProvider):
        """Validate cluster configuration from runtime perspective."""
        pass

    def verify_config(self, cluster_config: Dict[str, Any], provider: NodeProvider):
        """Verify cluster configuration at the last stage of bootstrap.
        The verification may mean a slow process to check with a server"""
        pass

    def with_environment_variables(
            self, runtime_config: Dict[str, Any], provider: NodeProvider) -> Dict[str, Any]:
        """Export necessary runtime environment variables for running node commands.
        For example: {"ENV_NAME": value}
        """
        return {}

    def get_runnable_command(self, target: str):
        """Return the runnable command for the target script.
        For example: ["bash", target]
        """
        return None

    def get_runtime_commands(self, cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a copy of runtime commands to run at different stages"""
        return None

    def get_defaults_config(self, cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a copy of runtime config"""
        return None

    def get_useful_urls(self, cluster_head_ip: str):
        """Return the useful urls to show when cluster started.
        It's an array of dictionary
        For example:
        [
            {"name": "app web", "url": "http://localhost/app"},
        ]
        """
        return None


    @staticmethod
    def get_logs(self) -> Dict[str, str]:
        """Return a dictionary of name to log paths.
        For example {"server-a": "/tmp/server-a/logs"}
        """
        return {}

    @staticmethod
    def get_processes(self):
        """Return a list of processes for this runtime.
        Format:
        #1 Keyword to filter,
        #2 filter by command (True)/filter by args (False)
        #3 The third element is the process name.
        #4 The forth element, if node, the process should on all nodes, if head, the process should on head node.
        For example
        ["cloudtik_cluster_controller.py", False, "ClusterController", "head"],
        """
        return []
