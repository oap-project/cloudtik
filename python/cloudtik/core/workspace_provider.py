import logging
from typing import Any, Dict


logger = logging.getLogger(__name__)


class WorkspaceProvider:
    """Interface for preparing a workspace from a Cloud.

    **Important**: This is an INTERNAL API that is only exposed for the purpose
    of implementing custom workspace providers. It is not allowed to call into
    WorkspaceProvider methods from any package outside, only to
    define new implementations of WorkspaceProvider for use with the "external" node
    provider option.

    WorkspaceProvider are namespaced by the `workspace_name` parameter; they only
    operate on resources within that namespace.
    """

    def __init__(self, provider_config: Dict[str, Any],
                 workspace_name: str) -> None:
        self.provider_config = provider_config
        self.workspace_name = workspace_name

    def create_workspace(self, cluster_config: Dict[str, Any]):
        pass

    def delete_workspace(self, cluster_config: Dict[str, Any]):
        pass

    def update_workspace_firewalls(self, cluster_config: Dict[str, Any]):
        pass

    def check_workspace_resource(self, cluster_config: Dict[str, Any]) -> bool:
        """Bootstraps the cluster config by adding env defaults if needed."""
        return True

    def publish_global_variables(self, cluster_config: Dict[str, Any],
                                 head_node_id: str, runtime_tags: Dict[str, Any]):
        """Workspace provide a way to publish global variables and can be subscribed anybody"""
        pass

    def subscribe_global_variables(self, cluster_config: Dict[str, Any]):
        """Workspace provide a way to subscribe global variables and can be subscribed anybody"""
        return None

    @staticmethod
    def validate_config(provider_config: Dict[str, Any]):
        pass

    @staticmethod
    def bootstrap_workspace_config(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Bootstraps the cluster config by adding env defaults if needed."""
        return cluster_config
