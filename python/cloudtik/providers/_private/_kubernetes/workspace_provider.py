import logging
from typing import Any, Dict, Optional

from cloudtik.providers._private._kubernetes.config import bootstrap_kubernetes_workspace, create_kubernetes_workspace, \
    delete_kubernetes_workspace, check_kubernetes_workspace_existence, check_kubernetes_workspace_integrity, \
    list_kubernetes_clusters, get_kubernetes_workspace_info, \
    publish_kubernetes_global_variables, subscribe_kubernetes_global_variables, \
    validate_kubernetes_workspace_config, update_kubernetes_workspace
from cloudtik.core.workspace_provider import WorkspaceProvider

logger = logging.getLogger(__name__)


class KubernetesWorkspaceProvider(WorkspaceProvider):
    def __init__(self, provider_config, workspace_name):
        WorkspaceProvider.__init__(self, provider_config, workspace_name)

    def create_workspace(self, config):
        create_kubernetes_workspace(config)

    def delete_workspace(self, config,
                         delete_managed_storage: bool = False,
                         delete_managed_database: bool = False):
        delete_kubernetes_workspace(
            config, delete_managed_storage, delete_managed_database)

    def update_workspace(self, config: Dict[str, Any],
                         delete_managed_storage: bool = False,
                         delete_managed_database: bool = False):
        update_kubernetes_workspace(
            config, delete_managed_storage, delete_managed_database)

    def check_workspace_existence(self, config: Dict[str, Any]):
        return check_kubernetes_workspace_existence(config)

    def check_workspace_integrity(self, config):
        return check_kubernetes_workspace_integrity(config)

    def list_clusters(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return list_kubernetes_clusters(config)

    def publish_global_variables(self, cluster_config: Dict[str, Any],
                                 global_variables: Dict[str, Any]):
        publish_kubernetes_global_variables(cluster_config, global_variables)

    def subscribe_global_variables(self, cluster_config: Dict[str, Any]):
        return subscribe_kubernetes_global_variables(self.provider_config, self.workspace_name, cluster_config)

    def validate_config(self, provider_config: Dict[str, Any]):
        validate_kubernetes_workspace_config(provider_config, self.workspace_name)

    def get_workspace_info(self, config: Dict[str, Any]):
        return get_kubernetes_workspace_info(config)

    @staticmethod
    def bootstrap_workspace_config(config):
        return bootstrap_kubernetes_workspace(config)
