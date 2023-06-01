import logging
from typing import Any, Dict

from cloudtik.core.node_provider import NodeProvider
from cloudtik.core.runtime import Runtime
from cloudtik.runtime.ai.utils import _with_runtime_environment_variables, \
    _get_runtime_processes, _get_runtime_logs, _get_runtime_commands, \
    _get_defaults_config, _get_runtime_services, publish_service_uri, _get_runtime_service_ports

logger = logging.getLogger(__name__)


class AIRuntime(Runtime):
    """Implementation for AI Runtime"""

    def __init__(self, runtime_config: Dict[str, Any]) -> None:
        Runtime.__init__(self, runtime_config)

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
        publish_service_uri(cluster_config, head_node_id)

    def get_runtime_commands(self, cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a copy of runtime commands to run at different stages"""
        return _get_runtime_commands(self.runtime_config, cluster_config)

    def get_defaults_config(self, cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a copy of runtime config"""
        return _get_defaults_config(self.runtime_config, cluster_config)

    def get_runtime_services(self, cluster_head_ip: str):
        return _get_runtime_services(cluster_head_ip)

    def get_runtime_service_ports(self) -> Dict[str, Any]:
        return _get_runtime_service_ports(self.runtime_config)

    @staticmethod
    def get_logs() -> Dict[str, str]:
        return _get_runtime_logs()

    @staticmethod
    def get_processes():
        return _get_runtime_processes()
