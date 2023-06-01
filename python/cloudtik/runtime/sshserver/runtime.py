import logging
from typing import Any, Dict

from cloudtik.core.node_provider import NodeProvider
from cloudtik.runtime.common.runtime_base import RuntimeBase
from cloudtik.runtime.sshserver.utils import _get_runtime_processes, \
    _get_runtime_service_ports, _with_runtime_environment_variables, _bootstrap_runtime_config

logger = logging.getLogger(__name__)


class SSHServerRuntime(RuntimeBase):
    """Implementation for SSH Server Runtime which provides SSH service on each host"""

    def __init__(self, runtime_config: Dict[str, Any]) -> None:
        super().__init__(runtime_config)

    def bootstrap_config(self, cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Final chance to update the config with runtime specific configurations
        This happens after provider bootstrap_config is done.
        """
        cluster_config = _bootstrap_runtime_config(cluster_config)
        return cluster_config

    def with_environment_variables(
            self, config: Dict[str, Any], provider: NodeProvider,
            node_id: str) -> Dict[str, Any]:
        """Export necessary runtime environment variables for running node commands.
        """
        return _with_runtime_environment_variables(
            self.runtime_config, config=config)

    def get_runtime_service_ports(self) -> Dict[str, Any]:
        return _get_runtime_service_ports(self.runtime_config)

    @staticmethod
    def get_processes():
        """Return a list of processes for this runtime."""
        return _get_runtime_processes()
