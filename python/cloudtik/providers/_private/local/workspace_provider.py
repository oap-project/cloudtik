import logging
from typing import Any, Dict, Optional

from cloudtik.core._private.providers import _get_node_provider
from cloudtik.core._private.utils import get_running_head_node
from cloudtik.core.tags import CLOUDTIK_GLOBAL_VARIABLE_KEY, CLOUDTIK_GLOBAL_VARIABLE_KEY_PREFIX
from cloudtik.core.workspace_provider import WorkspaceProvider
from cloudtik.providers._private.local.config import get_workspace_head_nodes, list_local_clusters, _get_node_tags

logger = logging.getLogger(__name__)


class LocalWorkspaceProvider(WorkspaceProvider):
    def __init__(self, provider_config, workspace_name):
        WorkspaceProvider.__init__(self, provider_config, workspace_name)

    def check_workspace_integrity(self, config: Dict[str, Any]) -> bool:
        """Check whether the workspace is correctly configured"""
        return True

    def list_clusters(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return list_local_clusters(self.provider_config)

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
        head_nodes = get_workspace_head_nodes(self.provider_config)
        for head in head_nodes:
            node_tags = _get_node_tags(self.provider_config, head)
            for key, value in node_tags.items():
                if key.startswith(CLOUDTIK_GLOBAL_VARIABLE_KEY_PREFIX):
                    global_variable_name = key[len(CLOUDTIK_GLOBAL_VARIABLE_KEY_PREFIX):]
                    global_variables[global_variable_name] = value

        return global_variables
