import logging
from typing import Any, Dict
from cloudtik.providers._private.aws.config import  create_aws_workspace, \
    delete_workspace_aws, check_aws_workspace_resource, update_aws_workspace_firewalls, \
    get_workspace_head_nodes
from cloudtik.core._private.providers import _get_node_provider

from cloudtik.core.workspace_provider import WorkspaceProvider

logger = logging.getLogger(__name__)


class AWSWorkspaceProvider(WorkspaceProvider):
    def __init__(self, provider_config, workspace_name):
        WorkspaceProvider.__init__(self, provider_config, workspace_name)

    def create_workspace(self, cluster_config):
        create_aws_workspace(cluster_config)

    def delete_workspace(self, cluster_config):
        delete_workspace_aws(cluster_config)

    def update_workspace_firewalls(self, cluster_config):
        update_aws_workspace_firewalls(cluster_config)

    def check_workspace_resource(self, cluster_config):
        return check_aws_workspace_resource(cluster_config)

    def publish_global_variables(self, cluster_config: Dict[str, Any],
                                 head_node_id: str, runtime_tags: Dict[str, Any]):

        provider = _get_node_provider(cluster_config["provider"], cluster_config["cluster_name"])
        provider.set_node_tags(head_node_id, runtime_tags)

    def subscribe_global_variables(self, cluster_config: Dict[str, Any],
                                 runtime_tags: Dict[str, Any]):
        nodes = get_workspace_head_nodes(cluster_config, runtime_tags)
        if len(nodes) == 0:
            return None
        else:
            node = nodes[0]
            for key in runtime_tags.keys():
                for tag in node.tags:
                    if tag.get("Key") == key:
                        runtime_tags[key] = tag.get("Value")

            return runtime_tags

    @staticmethod
    def validate_config(
            provider_config: Dict[str, Any]):
        pass

    @staticmethod
    def bootstrap_workspace_config(cluster_config):
        return cluster_config
