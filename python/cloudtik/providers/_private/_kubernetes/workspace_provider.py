import logging
from typing import Any, Dict

from cloudtik.providers._private._kubernetes.config import bootstrap_workspace_kubernetes
from cloudtik.core.workspace_provider import WorkspaceProvider

logger = logging.getLogger(__name__)


class KubernetesWorkspaceProvider(WorkspaceProvider):
    def __init__(self, provider_config, workspace_name):
        WorkspaceProvider.__init__(self, provider_config, workspace_name)

    def validate_config(self, provider_config: Dict[str, Any]):
        pass

    @staticmethod
    def bootstrap_workspace_config(cluster_config):
        return bootstrap_workspace_kubernetes(cluster_config)
