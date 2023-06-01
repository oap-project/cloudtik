import logging
from typing import Any, Dict, Optional, List

from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_HDFS, BUILT_IN_RUNTIME_METASTORE
from cloudtik.core.node_provider import NodeProvider
from cloudtik.runtime.common.runtime_base import RuntimeBase
from cloudtik.runtime.trino.utils import _with_runtime_environment_variables, \
    _is_runtime_scripts, _get_runnable_command, _get_runtime_processes, \
    _get_runtime_logs, _get_runtime_services, _config_depended_services, _get_runtime_service_ports

logger = logging.getLogger(__name__)


class TrinoRuntime(RuntimeBase):
    """Implementation for Trino Runtime"""

    def __init__(self, runtime_config: Dict[str, Any]) -> None:
        super().__init__(runtime_config)

    def prepare_config(self, cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare runtime specific configurations"""
        cluster_config = _config_depended_services(cluster_config)
        return cluster_config

    def with_environment_variables(
            self, config: Dict[str, Any], provider: NodeProvider,
            node_id: str) -> Dict[str, Any]:
        """Export necessary runtime environment variables for running node commands.
        For example: {"ENV_NAME": value}
        """
        return _with_runtime_environment_variables(
            self.runtime_config, config=config, provider=provider, node_id=node_id)

    def get_runnable_command(self, target: str, runtime_options: Optional[List[str]]):
        """Return the runnable command for the target script.
        For example: ["bash", target]
        """
        if not _is_runtime_scripts(target):
            return None

        return _get_runnable_command(target)

    def get_runtime_services(self, cluster_head_ip: str):
        return _get_runtime_services(cluster_head_ip)

    def get_runtime_service_ports(self) -> Dict[str, Any]:
        return _get_runtime_service_ports(self.runtime_config)

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

    @staticmethod
    def get_dependencies():
        return [BUILT_IN_RUNTIME_HDFS, BUILT_IN_RUNTIME_METASTORE]
