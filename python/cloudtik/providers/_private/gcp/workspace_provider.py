import logging
from typing import Any, Dict, Optional

from cloudtik.providers._private.gcp.config import create_gcp_workspace, \
    delete_workspace_gcp, check_gcp_workspace_resource, update_gcp_workspace_firewalls, \
    get_workspace_head_nodes, list_gcp_clusters
from cloudtik.core._private.providers import _get_node_provider
from cloudtik.core._private.utils import binary_to_hex, hex_to_binary
from cloudtik.core.tags import CLOUDTIK_GLOBAL_VARIABLE_KEY_PREFIX, CLOUDTIK_GLOBAL_VARIABLE_KEY
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

    def list_clusters(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return list_gcp_clusters(config)

    def publish_global_variables(self, cluster_config: Dict[str, Any],
                                 head_node_id: str, global_variables: Dict[str, Any]):
        # Add prefix to the variables
        global_variables_prefixed = {}
        for name in global_variables:
            # For gcp labels, only hyphens (-), underscores (_), lowercase characters, and numbers are allowed.
            # Keys must start with a lowercase character. International characters are allowed.
            hex_name = binary_to_hex(name.encode())
            prefixed_name = CLOUDTIK_GLOBAL_VARIABLE_KEY.format(hex_name)
            global_variables_prefixed[prefixed_name] = binary_to_hex(global_variables[name].encode())

        provider = _get_node_provider(cluster_config["provider"], cluster_config["cluster_name"])
        provider.set_node_tags(head_node_id, global_variables_prefixed)

    def subscribe_global_variables(self, cluster_config: Dict[str, Any]):
        global_variables = {}
        head_nodes = get_workspace_head_nodes(cluster_config)
        for head in head_nodes:
            for key, value in head.get("labels", {}).items():
                if key.startswith(CLOUDTIK_GLOBAL_VARIABLE_KEY_PREFIX):
                    # For gcp labels, only hyphens (-), underscores (_), lowercase characters, and numbers are allowed.
                    # Keys must start with a lowercase character. International characters are allowed.
                    global_variable_name = hex_to_binary(key[len(CLOUDTIK_GLOBAL_VARIABLE_KEY_PREFIX):]).decode()
                    global_variables[global_variable_name] = hex_to_binary(value).decode()

        return global_variables

    @staticmethod
    def validate_config(
            provider_config: Dict[str, Any]):
        pass

    @staticmethod
    def bootstrap_workspace_config(cluster_config):
        return cluster_config
