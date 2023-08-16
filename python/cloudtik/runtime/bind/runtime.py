import logging
from typing import Any, Dict

from cloudtik.core.node_provider import NodeProvider
from cloudtik.runtime.common.runtime_base import RuntimeBase
from cloudtik.runtime.bind.utils import _get_runtime_processes, \
    _get_runtime_services, _with_runtime_environment_variables

logger = logging.getLogger(__name__)


class BindRuntime(RuntimeBase):
    """Implementation for Bind Runtime for a DNS Server
    which resolves domain names for both local and upstream"""

    def __init__(self, runtime_config: Dict[str, Any]) -> None:
        super().__init__(runtime_config)

    def with_environment_variables(
            self, config: Dict[str, Any], provider: NodeProvider,
            node_id: str) -> Dict[str, Any]:
        """Export necessary runtime environment variables for running node commands.
        For example: {"ENV_NAME": value}
        """
        return _with_runtime_environment_variables(
            self.runtime_config, config=config)

    def get_runtime_services(self, cluster_name: str):
        return _get_runtime_services(self.runtime_config, cluster_name)

    @staticmethod
    def get_processes():
        return _get_runtime_processes()
