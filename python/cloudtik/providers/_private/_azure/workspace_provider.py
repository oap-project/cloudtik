import logging
from typing import Any, Dict

from cloudtik.providers._private._azure.config import create_azure_workspace, \
    delete_workspace_azure
from cloudtik.core.workspace_provider import WorkspaceProvider

logger = logging.getLogger(__name__)


class AzureWorkspaceProvider(WorkspaceProvider):
    def __init__(self, provider_config, workspace_name):
        WorkspaceProvider.__init__(self, provider_config, workspace_name)

    def create_workspace(self, cluster_config):
        return create_azure_workspace(cluster_config)

    def delete_workspace(self, cluster_config):
        return delete_workspace_azure(cluster_config)
    #
    # def check_workspace_resource(self, cluster_config):
    #     return check_azure_workspace_resource(cluster_config)

    @staticmethod
    def validate_config(
            provider_config: Dict[str, Any]) -> None:
        return None

    @staticmethod
    def bootstrap_workspace_config(cluster_config):
        return cluster_config

