"""IMPORTANT: this is an experimental interface and not currently stable."""

from typing import Union

from cloudtik.core.api import Cluster, ThisCluster
from cloudtik.runtime.ml.utils import get_runtime_services


class MLCluster(Cluster):
    def __init__(self, cluster_config: Union[dict, str], should_bootstrap: bool = True) -> None:
        """Create a Spark cluster object to operate on with this API.

        Args:
            cluster_config (Union[str, dict]): Either the config dict of the
                cluster, or a path pointing to a file containing the config.
        """
        Cluster.__init__(self, cluster_config, should_bootstrap)

    def get_services(self):
        return get_runtime_services(self.config)


class ThisMLCluster(ThisCluster):
    def __init__(self) -> None:
        """Create a Spark cluster object to operate on with this API on head."""
        ThisCluster.__init__(self)

    def get_services(self):
        return get_runtime_services(self.config)
