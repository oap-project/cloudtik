from typing import Any, Dict, Optional

from cloudtik.core._private.providers import _get_node_provider
from cloudtik.core._private.utils import check_workspace_name_format, \
    get_running_head_node
from cloudtik.core.tags import CLOUDTIK_GLOBAL_VARIABLE_KEY, \
    CLOUDTIK_GLOBAL_VARIABLE_KEY_PREFIX
from cloudtik.core.workspace_provider import Existence, WorkspaceProvider
from cloudtik.providers._private.huaweicloud.config import \
    _get_workspace_head_nodes, bootstrap_huaweicloud_workspace, \
    check_huaweicloud_workspace_existence, \
    check_huaweicloud_workspace_integrity, create_huaweicloud_workspace, \
    delete_huaweicloud_workspace, get_huaweicloud_workspace_info, \
    list_huaweicloud_clusters, \
    update_huaweicloud_workspace
from cloudtik.providers._private.huaweicloud.utils import tags_list_to_dict

HUAWEICLOUD_WORKSPACE_NAME_MAX_LEN = 32


class HUAWEICLOUDWorkspaceProvider(WorkspaceProvider):

    def __init__(self, provider_config: Dict[str, Any],
                 workspace_name: str) -> None:
        WorkspaceProvider.__init__(self, provider_config, workspace_name)

    def create_workspace(self, config: Dict[str, Any]):
        create_huaweicloud_workspace(config)

    def delete_workspace(self, config: Dict[str, Any],
                         delete_managed_storage: bool = False,
                         delete_managed_database:bool = False):
        delete_huaweicloud_workspace(config, delete_managed_storage)

    def update_workspace(self, config: Dict[str, Any],
                         delete_managed_storage: bool = False,
                         delete_managed_database: bool = False):
        update_huaweicloud_workspace(
            config, delete_managed_storage, delete_managed_database)

    def check_workspace_existence(self, config: Dict[str, Any]) -> Existence:
        return check_huaweicloud_workspace_existence(config)

    def check_workspace_integrity(self, config: Dict[str, Any]) -> bool:
        return check_huaweicloud_workspace_integrity(config)

    def list_clusters(self, config: Dict[str, Any]) -> Optional[
        Dict[str, Any]]:
        return list_huaweicloud_clusters(config)

    def publish_global_variables(self, cluster_config: Dict[str, Any],
                                 global_variables: Dict[str, Any]):
        """
        The global variables implements as tags. The following basic restrictions apply to tags:
        Each resource supports up to 10 key-value pairs.
        Key: Only letters, digits, underscores (_), and hyphens (-) are allowed.
        Enter a maximum of 36 characters.
        Value: Only letters, digits, underscores (_), periods (.), and hyphens (-) are allowed.
        Enter a maximum of 43 characters.
        """
        # Add prefix to the variables
        global_variables_prefixed = {}
        for name in global_variables:
            prefixed_name = CLOUDTIK_GLOBAL_VARIABLE_KEY.format(name)
            global_variables_prefixed[prefixed_name] = global_variables[name]

        provider = _get_node_provider(cluster_config['provider'],
                                      cluster_config['cluster_name'])
        head_node_id = get_running_head_node(cluster_config, provider)
        provider.set_node_tags(head_node_id, global_variables_prefixed)

    def subscribe_global_variables(self, cluster_config: Dict[str, Any]):
        global_variables = {}
        head_nodes = _get_workspace_head_nodes(self.provider_config,
                                               self.workspace_name)
        for head in head_nodes:
            # Huawei Cloud server tags format:
            # ['key1=value1', 'key2=value2', 'key3=value3']
            tags_dict = tags_list_to_dict(head.tags)
            for tag_key, tag_value in tags_dict.items():
                _prefix = CLOUDTIK_GLOBAL_VARIABLE_KEY_PREFIX
                if tag_key.startswith(_prefix):
                    global_variable_name = tag_key[len(_prefix):]
                    global_variables[global_variable_name] = tag_value

        return global_variables

    def validate_config(self, provider_config: Dict[str, Any]):
        if len(self.workspace_name) > HUAWEICLOUD_WORKSPACE_NAME_MAX_LEN or \
                not check_workspace_name_format(self.workspace_name):
            raise RuntimeError(
                "{} workspace name is between 1 and {} characters, "
                "and can only contain lowercase alphanumeric "
                "characters and dashes".format(
                    provider_config["type"],
                    HUAWEICLOUD_WORKSPACE_NAME_MAX_LEN)
            )

    def get_workspace_info(self, config: Dict[str, Any]):
        return get_huaweicloud_workspace_info(config)

    @staticmethod
    def bootstrap_workspace_config(config: Dict[str, Any]) -> Dict[str, Any]:
        return bootstrap_huaweicloud_workspace(config)
