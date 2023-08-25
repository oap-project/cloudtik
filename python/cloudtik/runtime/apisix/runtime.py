import logging
from typing import Any, Dict

from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_ETCD
from cloudtik.core.node_provider import NodeProvider
from cloudtik.runtime.common.runtime_base import RuntimeBase
from cloudtik.runtime.apisix.utils import _get_runtime_processes, \
    _get_runtime_services, _with_runtime_environment_variables, _config_depended_services, _prepare_config_on_head, \
    _validate_config

logger = logging.getLogger(__name__)


class APISIXRuntime(RuntimeBase):
    """Implementation for APISIX Runtime for API Gateway"""

    def __init__(self, runtime_config: Dict[str, Any]) -> None:
        super().__init__(runtime_config)

    def prepare_config(self, cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare runtime specific configurations"""
        cluster_config = _config_depended_services(cluster_config)
        return cluster_config

    def prepare_config_on_head(
            self, cluster_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Configure runtime such as using service discovery to configure
        internal service addresses the runtime depends.
        The head configuration will be updated and saved with the returned configuration.
        """
        return _prepare_config_on_head(cluster_config)

    def validate_config(self, cluster_config: Dict[str, Any]):
        """Validate cluster configuration from runtime perspective."""
        _validate_config(cluster_config)

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

    @staticmethod
    def get_dependencies():
        return [BUILT_IN_RUNTIME_ETCD]
