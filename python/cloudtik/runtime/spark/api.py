"""IMPORTANT: this is an experimental interface and not currently stable."""

import os
from typing import Union

from cloudtik.core._private.call_context import CallContext
from cloudtik.core._private.cluster.cluster_config import _bootstrap_config, _load_cluster_config
from cloudtik.runtime.spark.utils import request_rest_applications, request_rest_yarn


class SparkCluster:
    def __init__(self, cluster_config: Union[dict, str], should_bootstrap: bool = True) -> None:
        """Create a Spark cluster object to operate on with this API.

        Args:
            cluster_config (Union[str, dict]): Either the config dict of the
                cluster, or a path pointing to a file containing the config.
        """
        self.cluster_config = cluster_config
        if isinstance(cluster_config, dict):
            if should_bootstrap:
                self.config = _bootstrap_config(
                    cluster_config, no_config_cache=True)
            else:
                self.config = cluster_config
        else:
            if not os.path.exists(cluster_config):
                raise ValueError("Cluster config file not found: {}".format(cluster_config))
            self.config = _load_cluster_config(
                cluster_config, should_bootstrap=should_bootstrap, no_config_cache=True)

        # TODO: Each call may need its own call context
        self.call_context = CallContext()
        self.call_context.set_call_from_api(True)

    def applications(self, endpoint: str):
        """Make a rest request to Spark History Server

        Args:
            endpoint (str): The Spark history server rest endpoint to request
        """
        return request_rest_applications(self.config, endpoint)

    def yarn(self, endpoint: str):
        """Make a rest request to YARN Resource Manager

        Args:
            endpoint (str): The Spark history server rest endpoint to request
        """
        return request_rest_yarn(self.config, endpoint)
