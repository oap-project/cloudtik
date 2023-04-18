import logging
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)

CLOUDTIK_MANAGED_CLOUD_STORAGE = "managed.cloud.storage"
CLOUDTIK_MANAGED_CLOUD_STORAGE_URI = "cloud.storage.uri"


class Existence(Enum):
    NOT_EXIST = 1
    COMPLETED = 2
    IN_COMPLETED = 3
    STORAGE_ONLY = 4
    DATABASE_ONLY = 5
    STORAGE_AND_DATABASE_ONLY = 6


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

    def create_workspace(self, config: Dict[str, Any]):
        """Create a workspace and all the resources needed for the workspace based on the config."""
        pass

    def delete_workspace(self, config: Dict[str, Any],
                         delete_managed_storage: bool = False,
                         delete_managed_database:bool = False):
        """Delete all the resources created for the workspace.
        Managed cloud storage is not deleted by default unless delete_managed_storage is specified.
        """
        pass

    def update_workspace_firewalls(self, config: Dict[str, Any]):
        """Update the workspace firewalls based on the latest information in config"""
        pass

    def check_workspace_existence(self, config: Dict[str, Any]) -> Existence:
        """Check whether the workspace with the same name exists.
        The existing workspace may be in incomplete state.
        """
        return Existence.NOT_EXIST

    def check_workspace_integrity(self, config: Dict[str, Any]) -> bool:
        """Check whether the workspace is correctly configured"""
        return False

    def list_clusters(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get a list of clusters belongs to the workspace specified by the workspace config
        The mapping is keyed by the cluster name with the value of node info of head node.
        """
        return None

    def publish_global_variables(self, cluster_config: Dict[str, Any],
                                 global_variables: Dict[str, Any]):
        """Workspace provide a way to publish global variables and can be subscribed anybody"""
        pass

    def subscribe_global_variables(self, cluster_config: Dict[str, Any]):
        """Workspace provide a way to subscribe global variables and can be subscribed anybody"""
        return {}

    def get_workspace_info(self, config: Dict[str, Any]):
        """Get a dictionary of key value pairs shown for workspace info command."""
        return {}

    def validate_config(self, provider_config: Dict[str, Any]):
        """Check the workspace configuration validation.
        This happens before bootstrap_workspace_config
        """
        pass

    @staticmethod
    def bootstrap_workspace_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Bootstraps the workspace config by adding env defaults if needed."""
        return config
