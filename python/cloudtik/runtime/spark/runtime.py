import logging
from typing import Any, Dict

from cloudtik.core.node_provider import NodeProvider
from cloudtik.runtime.spark.utils import config_spark_runtime_resources, with_spark_runtime_environment_variables, \
    is_spark_runtime_scripts, get_spark_runtime_command, get_spark_runtime_processes, spark_runtime_validate_config, \
    spark_runtime_verify_config, get_spark_runtime_logs

logger = logging.getLogger(__name__)


class SparkRuntime:
    """Implementation for Spark Runtime"""

    def __init__(self, runtime_config: Dict[str, Any]) -> None:
        self.runtime_config = runtime_config

    def prepare_config(self, cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare runtime specific configurations"""
        return config_spark_runtime_resources(cluster_config)

    def validate_config(self, cluster_config: Dict[str, Any], provider: NodeProvider):
        """Validate cluster configuration from runtime perspective."""
        spark_runtime_validate_config(cluster_config, provider)

    def verify_config(self, cluster_config: Dict[str, Any], provider: NodeProvider):
        """Verify cluster configuration at the last stage of bootstrap.
        The verification may mean a slow process to check with a server"""
        spark_runtime_verify_config(cluster_config, provider)

    def with_environment_variables(
            self, runtime_config: Dict[str, Any], provider: NodeProvider) -> Dict[str, Any]:
        """Export necessary runtime environment variables for running node commands.
        For example: {"ENV_NAME": value}
        """
        return with_spark_runtime_environment_variables(runtime_config, provider)

    def get_logs(self) -> Dict[str, str]:
        """Return a dictionary of name to log paths.
        For example {"server-a": "/tmp/server-a/logs"}
        """
        return get_spark_runtime_logs()

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
        return get_spark_runtime_processes()

    def is_runnable_scripts(self, script_file: str) -> bool:
        """Returns whether the script file is runnable by this runtime"""
        return is_spark_runtime_scripts(script_file)

    def get_runnable_command(self, target: str):
        """Return the runnable command for the target script.
        For example: ["bash", target]
        """
        return get_spark_runtime_command(target)
