"""IMPORTANT: this is an experimental interface and not currently stable."""

from typing import Union, Optional

from cloudtik.core.api import Cluster, ThisCluster
from cloudtik.runtime.ai.utils import get_runtime_endpoints


class AICluster(Cluster):
    def __init__(
            self, cluster_config: Union[dict, str],
            should_bootstrap: bool = True,
            no_config_cache: bool = True,
            verbosity: Optional[int] = None) -> None:
        """Create a Spark cluster object to operate on with this API.

        Args:
            cluster_config (Union[str, dict]): Either the config dict of the
                cluster, or a path pointing to a file containing the config.
        """
        super().__init__(
            cluster_config, should_bootstrap,
            no_config_cache, verbosity)

    def get_endpoints(self):
        return get_runtime_endpoints(self.config)


class ThisAICluster(ThisCluster):
    def __init__(self, verbosity: Optional[int] = None) -> None:
        """Create a Spark cluster object to operate on with this API on head."""
        super().__init__(verbosity)

    def get_endpoints(self):
        return get_runtime_endpoints(self.config)
