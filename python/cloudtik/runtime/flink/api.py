"""IMPORTANT: this is an experimental interface and not currently stable."""

from typing import Union

from cloudtik.core.api import Cluster, ThisCluster
from cloudtik.runtime.flink.utils import request_rest_jobs, request_rest_yarn, get_runtime_default_storage


class FlinkCluster(Cluster):
    def __init__(self, cluster_config: Union[dict, str], should_bootstrap: bool = True) -> None:
        """Create a Flink cluster object to operate on with this API.

        Args:
            cluster_config (Union[str, dict]): Either the config dict of the
                cluster, or a path pointing to a file containing the config.
        """
        Cluster.__init__(self, cluster_config, should_bootstrap)

    def jobs(self, endpoint: str):
        """Make a rest request to Flink History Server

        Args:
            endpoint (str): The Spark history server rest endpoint to request
        """
        return request_rest_jobs(self.config, endpoint)

    def yarn(self, endpoint: str):
        """Make a rest request to YARN Resource Manager

        Args:
            endpoint (str): The Spark history server rest endpoint to request
        """
        return request_rest_yarn(self.config, endpoint)

    def get_default_storage(self):
        return get_runtime_default_storage(self.config)


class ThisFlinkCluster(ThisCluster):
    def __init__(self) -> None:
        """Create a Flink cluster object to operate on with this API on head."""
        ThisCluster.__init__(self)

    def jobs(self, endpoint: str):
        """Make a rest request to Flink History Server

        Args:
            endpoint (str): The Spark history server rest endpoint to request
        """
        return request_rest_jobs(self.config, endpoint, on_head=True)

    def yarn(self, endpoint: str):
        """Make a rest request to YARN Resource Manager

        Args:
            endpoint (str): The Spark history server rest endpoint to request
        """
        return request_rest_yarn(self.config, endpoint, on_head=True)

    def get_default_storage(self):
        return get_runtime_default_storage(self.config)
