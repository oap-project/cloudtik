import logging
from typing import Any, Dict

from cloudtik.core.node_provider import NodeProvider

logger = logging.getLogger(__name__)


class RuntimeProvider:
    """Interface for runtime abstraction.

    **Important**: This is an INTERNAL API that is only exposed for the purpose
    of implementing custom runtime. It is not allowed to call into
    RuntimeProvider methods from any package outside, only to
    define new implementations of RuntimeProvider for use with the "external" runtime
    provider option.
    """

    def __init__(self, runtime_config: Dict[str, Any]) -> None:
        self.runtime_config = runtime_config

    def with_environment_variables(
            self, runtime_config: Dict[str, Any], provider: NodeProvider) -> Dict[str, Any]:
        """Export necessary runtime environment variables for running node commands.
        For example: {"ENV_NAME": value}
        """
        return {}

    def validate_config(self, cluster_config: Dict[str, Any], provider: NodeProvider):
        """Validate cluster configuration from runtime perspective."""
        pass

    def get_logs(self) -> Dict[str, str]:
        """Return a dictionary of name to log paths.
        For example {"server-a": "/tmp/server-a/logs"}
        """
        return {}

    def prepare_config(self, cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare runtime specific configurations"""
        return cluster_config

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

    def is_runnable_scripts(self, script_file: str) -> bool:
        """Returns whether the script file is runnable by this runtime"""
        return False

    def get_runnable_command(self, target: str):
        """Return the runnable command for the target script.
        For example: ["bash", target]
        """
        return []
