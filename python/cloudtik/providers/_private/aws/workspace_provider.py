import logging
from typing import Any, Dict, Optional

from cloudtik.core._private.utils import get_running_head_node, check_workspace_name_format
from cloudtik.providers._private.aws.config import create_aws_workspace, \
    delete_aws_workspace, check_aws_workspace_integrity, \
    list_aws_clusters, _get_workspace_head_nodes, bootstrap_aws_workspace, \
    check_aws_workspace_existence, get_aws_workspace_info, update_aws_workspace
from cloudtik.core._private.providers import _get_node_provider
from cloudtik.core.tags import CLOUDTIK_GLOBAL_VARIABLE_KEY_PREFIX, CLOUDTIK_GLOBAL_VARIABLE_KEY
from cloudtik.core.workspace_provider import WorkspaceProvider

AWS_WORKSPACE_NAME_MAX_LEN = 31

logger = logging.getLogger(__name__)


class AWSWorkspaceProvider(WorkspaceProvider):
    def __init__(self, provider_config, workspace_name):
        WorkspaceProvider.__init__(self, provider_config, workspace_name)

    def create_workspace(self, config):
        create_aws_workspace(config)

    def delete_workspace(self, config,
                         delete_managed_storage: bool = False,
                         delete_managed_database: bool = False):
        delete_aws_workspace(
            config, delete_managed_storage, delete_managed_database)

    def update_workspace(self, config: Dict[str, Any],
                         delete_managed_storage: bool = False,
                         delete_managed_database: bool = False):
        update_aws_workspace(
            config, delete_managed_storage, delete_managed_database)

    def check_workspace_existence(self, config: Dict[str, Any]):
        return check_aws_workspace_existence(config)

    def check_workspace_integrity(self, config):
        return check_aws_workspace_integrity(config)

    def list_clusters(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return list_aws_clusters(config)

    def publish_global_variables(self, cluster_config: Dict[str, Any],
                                 global_variables: Dict[str, Any]):
        """
        The global variables implements as tags. The following basic restrictions apply to tags:
        Maximum number of tags per resource – 50
        For each resource, each tag key must be unique, and each tag key can have only one value.
        Maximum key length – 128 Unicode characters in UTF-8
        Maximum value length – 256 Unicode characters in UTF-8
        The allowed characters across services are: letters (a-z, A-Z), numbers (0-9),
        and spaces representable in UTF-8, and the following characters: + - = . _ : / @
        """
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
        if len(self.workspace_name) > AWS_WORKSPACE_NAME_MAX_LEN or \
                not check_workspace_name_format(self.workspace_name):
            raise RuntimeError("{} workspace name is between 1 and {} characters, "
                               "and can only contain lowercase alphanumeric "
                               "characters and dashes".format(provider_config["type"], AWS_WORKSPACE_NAME_MAX_LEN))

    def get_workspace_info(self, config: Dict[str, Any]):
        return get_aws_workspace_info(config)

    @staticmethod
    def bootstrap_workspace_config(config):
        return bootstrap_aws_workspace(config)
