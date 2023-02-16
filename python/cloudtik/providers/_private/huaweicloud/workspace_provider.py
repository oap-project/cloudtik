from typing import Any, Dict, Optional

from cloudtik.core._private.utils import check_workspace_name_format
from cloudtik.core.workspace_provider import Existence, WorkspaceProvider
from cloudtik.providers._private.huaweicloud.config import \
    bootstrap_huaweicloud_workspace, check_huaweicloud_workspace_existence, \
    create_huaweicloud_workspace, \
    delete_huaweicloud_workspace, get_huaweicloud_workspace_info, \
    list_huaweicloud_clusters, update_huaweicloud_workspace_firewalls, check_huaweicloud_workspace_integrity

HUAWEICLOUD_WORKSPACE_NAME_MAX_LEN = 32


class HUAWEICLOUDWorkspaceProvider(WorkspaceProvider):

    def __init__(self, provider_config: Dict[str, Any],
                 workspace_name: str) -> None:
        WorkspaceProvider.__init__(self, provider_config, workspace_name)

    def create_workspace(self, config: Dict[str, Any]):
        create_huaweicloud_workspace(config)

    def delete_workspace(self, config: Dict[str, Any],
                         delete_managed_storage: bool = False):
        delete_huaweicloud_workspace(config, delete_managed_storage)

    def update_workspace_firewalls(self, config: Dict[str, Any]):
        update_huaweicloud_workspace_firewalls(config)

    def check_workspace_existence(self, config: Dict[str, Any]) -> Existence:
        return check_huaweicloud_workspace_existence(config)

    def check_workspace_integrity(self, config: Dict[str, Any]) -> bool:
        return check_huaweicloud_workspace_integrity(config)

    def list_clusters(self, config: Dict[str, Any]) -> Optional[
        Dict[str, Any]]:
        return list_huaweicloud_clusters(config)

    def publish_global_variables(self, cluster_config: Dict[str, Any],
                                 global_variables: Dict[str, Any]):
        # TODO(ChenRui): implement node provider
        pass

    def subscribe_global_variables(self, cluster_config: Dict[str, Any]):
        # TODO(ChenRui): implement node provider
        pass

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
