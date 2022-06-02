import logging
from typing import Any, Dict, Optional

from cloudtik.core._private.utils import get_running_head_node, check_workspace_name_format
from cloudtik.providers._private.aws.config import create_aws_workspace, \
    delete_aws_workspace, check_aws_workspace_integrity, \
    update_aws_workspace_firewalls, list_aws_clusters, _get_workspace_head_nodes, bootstrap_aws_workspace, \
    check_aws_workspace_existence
from cloudtik.core._private.providers import _get_node_provider
from cloudtik.core.tags import CLOUDTIK_GLOBAL_VARIABLE_KEY_PREFIX, CLOUDTIK_GLOBAL_VARIABLE_KEY
from cloudtik.core.workspace_provider import WorkspaceProvider

logger = logging.getLogger(__name__)


class AWSWorkspaceProvider(WorkspaceProvider):
    def __init__(self, provider_config, workspace_name):
        WorkspaceProvider.__init__(self, provider_config, workspace_name)

    def create_workspace(self, config):
        create_aws_workspace(config)

    def delete_workspace(self, config,
                         delete_managed_storage: bool = False):
        delete_aws_workspace(config, delete_managed_storage)

    def update_workspace_firewalls(self, config):
        update_aws_workspace_firewalls(config)

    def check_workspace_existence(self, config: Dict[str, Any]):
        return check_aws_workspace_existence(config)

    def check_workspace_integrity(self, config):
        return check_aws_workspace_integrity(config)

    def list_clusters(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return list_aws_clusters(config)

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
        head_nodes = _get_workspace_head_nodes(
            self.provider_config, self.workspace_name)
        for head in head_nodes:
            for tag in head.tags:
                tag_key = tag.get("Key")
                if tag_key.startswith(CLOUDTIK_GLOBAL_VARIABLE_KEY_PREFIX):
                    global_variable_name = tag_key[len(CLOUDTIK_GLOBAL_VARIABLE_KEY_PREFIX):]
                    global_variables[global_variable_name] = tag.get("Value")

        return global_variables

    def validate_config(self, provider_config: Dict[str, Any]):
        if len(self.workspace_name) > 45 or not check_workspace_name_format(self.workspace_name):
            raise RuntimeError("{} workspace name is between 1 and 45 characters, "
                               "and can only contain lowercase alphanumeric "
                               "characters and dashes".format(provider_config["type"]))

    @staticmethod
    def bootstrap_workspace_config(config):
        return bootstrap_aws_workspace(config)
