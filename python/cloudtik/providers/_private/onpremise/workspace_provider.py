import logging
from typing import Any, Dict, Optional

from cloudtik.core._private.providers import _get_node_provider
from cloudtik.core._private.utils import get_running_head_node
from cloudtik.core.tags import CLOUDTIK_GLOBAL_VARIABLE_KEY, CLOUDTIK_GLOBAL_VARIABLE_KEY_PREFIX
from cloudtik.core.workspace_provider import WorkspaceProvider, Existence
from cloudtik.providers._private.onpremise.config import _get_node_tags
from cloudtik.providers._private.onpremise.workspace_config \
    import get_workspace_head_nodes, list_onpremise_clusters, check_onpremise_workspace_integrity, \
    check_onpremise_workspace_existence, create_onpremise_workspace, delete_onpremise_workspace

logger = logging.getLogger(__name__)


class OnPremiseWorkspaceProvider(WorkspaceProvider):
    def __init__(self, provider_config, workspace_name):
        WorkspaceProvider.__init__(self, provider_config, workspace_name)

    def create_workspace(self, config: Dict[str, Any]):
        """Create a workspace and all the resources needed for the workspace based on the config."""
        create_onpremise_workspace(config)

    def delete_workspace(self, config: Dict[str, Any],
                         delete_managed_storage: bool = False,
                         delete_managed_database:bool = False):
        """Delete all the resources created for the workspace.
        Managed cloud storage is not deleted by default unless delete_managed_storage is specified.
        """
        delete_onpremise_workspace(config)

    def check_workspace_integrity(self, config: Dict[str, Any]) -> bool:
        """Check whether the workspace is correctly configured"""
        return check_onpremise_workspace_integrity(config)

    def check_workspace_existence(self, config: Dict[str, Any]) -> Existence:
        """Check whether the workspace with the same name exists.
        The existing workspace may be in incomplete state.
        """
        return check_onpremise_workspace_existence(config)

    def list_clusters(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return list_onpremise_clusters(self.workspace_name, self.provider_config)

    def publish_global_variables(self, cluster_config: Dict[str, Any],
                                 global_variables: Dict[str, Any]):
        # Add prefix to the variables
        global_variables_prefixed = {}
        for name in global_variables:
            prefixed_name = CLOUDTIK_GLOBAL_VARIABLE_KEY.format(name)
            global_variables_prefixed[prefixed_name] = global_variables[name]

        provider = _get_node_provider(cluster_config["provider"], cluster_config["cluster_name"])
        head_node_id = get_running_head_node(cluster_config, provider)
        provider.set_node_tags(head_node_id, global_variables_prefixed)

    def subscribe_global_variables(self, cluster_config: Dict[str, Any]):
        global_variables = {}
        head_nodes = get_workspace_head_nodes(self.workspace_name, self.provider_config)
        for head_node in head_nodes:
            node_tags = _get_node_tags(head_node)
            for key, value in node_tags.items():
                if key.startswith(CLOUDTIK_GLOBAL_VARIABLE_KEY_PREFIX):
                    global_variable_name = key[len(CLOUDTIK_GLOBAL_VARIABLE_KEY_PREFIX):]
                    global_variables[global_variable_name] = value

        return global_variables
