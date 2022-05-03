import logging
from typing import Any, Dict, List, Optional
from cloudtik.providers._private.aws.config import create_aws_workspace, \
    delete_workspace_aws, check_aws_workspace_resource, update_aws_workspace_firewalls, \
    get_workspace_head_nodes, list_aws_clusters
from cloudtik.core._private.providers import _get_node_provider
from cloudtik.core.tags import CLOUDTIK_GLOBAL_VARIABLE_KEY_PREFIX, CLOUDTIK_GLOBAL_VARIABLE_KEY
from cloudtik.core.workspace_provider import WorkspaceProvider

logger = logging.getLogger(__name__)


class AWSWorkspaceProvider(WorkspaceProvider):
    def __init__(self, provider_config, workspace_name):
        WorkspaceProvider.__init__(self, provider_config, workspace_name)

    def create_workspace(self, config):
        create_aws_workspace(config)

    def delete_workspace(self, config):
        delete_workspace_aws(config)

    def update_workspace_firewalls(self, config):
        update_aws_workspace_firewalls(config)

    def check_workspace_resource(self, config):
        return check_aws_workspace_resource(config)

    def list_clusters(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return list_aws_clusters(config)

    def publish_global_variables(self, cluster_config: Dict[str, Any],
                                 head_node_id: str, global_variables: Dict[str, Any]):
        # Add prefix to the variables
        global_variables_prefixed = {}
        for name in global_variables:
            prefixed_name = CLOUDTIK_GLOBAL_VARIABLE_KEY.format(name)
            global_variables_prefixed[prefixed_name] = global_variables[name]

        provider = _get_node_provider(cluster_config["provider"], cluster_config["cluster_name"])
        provider.set_node_tags(head_node_id, global_variables_prefixed)

    def subscribe_global_variables(self, cluster_config: Dict[str, Any]):
        global_variables = {}
        head_nodes = get_workspace_head_nodes(cluster_config)
        for head in head_nodes:
            for tag in head.tags:
                tag_key = tag.get("Key")
                if tag_key.startswith(CLOUDTIK_GLOBAL_VARIABLE_KEY_PREFIX):
                    global_variable_name = tag_key[len(CLOUDTIK_GLOBAL_VARIABLE_KEY_PREFIX):]
                    global_variables[global_variable_name] = tag.get("Value")

        return global_variables

    @staticmethod
    def validate_config(
            provider_config: Dict[str, Any]):
        pass

    @staticmethod
    def bootstrap_workspace_config(cluster_config):
        return cluster_config
