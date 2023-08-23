import logging
from typing import Any, Dict, Optional, List

from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_METASTORE, BUILT_IN_RUNTIME_HDFS
from cloudtik.core.job_waiter import JobWaiter
from cloudtik.core.node_provider import NodeProvider
from cloudtik.core.scaling_policy import ScalingPolicy
from cloudtik.runtime.common.runtime_base import RuntimeBase
from cloudtik.runtime.spark.job_waiter import SparkJobWaiter
from cloudtik.runtime.spark.utils import _config_runtime_resources, _with_runtime_environment_variables, \
    _is_runtime_scripts, _get_runnable_command, get_runtime_processes, _validate_config, \
    get_runtime_logs, _get_runtime_endpoints, _config_depended_services, \
    _get_head_service_ports, _get_scaling_policy, _get_runtime_services, _prepare_config_on_head, _configure

logger = logging.getLogger(__name__)


class SparkRuntime(RuntimeBase):
    """Implementation for Spark Runtime"""

    def __init__(self, runtime_config: Dict[str, Any]) -> None:
        super().__init__(runtime_config)

    def prepare_config(self, cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare runtime specific configurations"""
        cluster_config = _config_runtime_resources(cluster_config)
        cluster_config = _config_depended_services(cluster_config)
        return cluster_config

    def validate_config(self, cluster_config: Dict[str, Any]):
        """Validate cluster configuration from runtime perspective."""
        _validate_config(cluster_config)

    def prepare_config_on_head(
            self, cluster_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Configure runtime such as using service discovery to configure
        internal service addresses the runtime depends.
        The head configuration will be updated and saved with the returned configuration.
        """
        return _prepare_config_on_head(cluster_config)

    def with_environment_variables(
            self, config: Dict[str, Any], provider: NodeProvider,
            node_id: str) -> Dict[str, Any]:
        """Export necessary runtime environment variables for running node commands.
        For example: {"ENV_NAME": value}
        """
        return _with_runtime_environment_variables(
            self.runtime_config, config=config, provider=provider, node_id=node_id)

    def configure(self, head: bool):
        """ This method is called on every node as the first step of executing runtime
        configure command.
        """
        _configure(self.runtime_config, head)

    def get_runnable_command(self, target: str, runtime_options: Optional[List[str]]):
        """Return the runnable command for the target script.
        For example: ["bash", target]
        """
        if not _is_runtime_scripts(target):
            return None

        return _get_runnable_command(target, runtime_options)

    def get_runtime_endpoints(self, cluster_head_ip: str):
        return _get_runtime_endpoints(cluster_head_ip)

    def get_head_service_ports(self) -> Dict[str, Any]:
        return _get_head_service_ports(self.runtime_config)

    def get_runtime_services(self, cluster_name: str):
        return _get_runtime_services(self.runtime_config, cluster_name)

    def get_scaling_policy(
            self, cluster_config: Dict[str, Any], head_ip: str
    ) -> Optional[ScalingPolicy]:
        return _get_scaling_policy(self.runtime_config, cluster_config, head_ip)

    def get_job_waiter(self, cluster_config: Dict[str, Any]) -> Optional[JobWaiter]:
        return SparkJobWaiter(cluster_config)

    @staticmethod
    def get_logs() -> Dict[str, str]:
        """Return a dictionary of name to log paths.
        For example {"server-a": "/tmp/server-a/logs"}
        """
        return get_runtime_logs()

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
        return get_runtime_processes()

    @staticmethod
    def get_dependencies():
        return [BUILT_IN_RUNTIME_HDFS, BUILT_IN_RUNTIME_METASTORE]
