import logging
from typing import Any, Dict

from cloudtik.providers._private.gcp.config import create_gcp_workspace, \
    delete_workspace_gcp, check_gcp_workspace_resource, update_gcp_workspace_firewalls
from cloudtik.core.workspace_provider import WorkspaceProvider

logger = logging.getLogger(__name__)


class GCPWorkspaceProvider(WorkspaceProvider):
    def __init__(self, provider_config, workspace_name):
        WorkspaceProvider.__init__(self, provider_config, workspace_name)

    def create_workspace(self, config):
        create_gcp_workspace(config)

    def delete_workspace(self, config):
        delete_workspace_gcp(config)
    
    def update_workspace_firewalls(self, config):
        update_gcp_workspace_firewalls(config)
    
    def check_workspace_resource(self, config):
        return check_gcp_workspace_resource(config)

    @staticmethod
    def validate_config(
            provider_config: Dict[str, Any]):
        pass

    @staticmethod
    def bootstrap_workspace_config(cluster_config):
        return cluster_config
