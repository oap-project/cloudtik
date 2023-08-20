import logging
from types import ModuleType
from typing import Any, Dict, Optional

from cloudtik.core._private.call_context import CallContext
from cloudtik.core.command_executor import CommandExecutor
from cloudtik.core.node_provider import NodeProvider
from cloudtik.core.tags import CLOUDTIK_TAG_CLUSTER_NAME, CLOUDTIK_TAG_WORKSPACE_NAME
from cloudtik.providers._private.local.config import prepare_local, post_prepare_local, bootstrap_local, \
    bootstrap_local_for_api
from cloudtik.providers._private.local.local_scheduler import LocalScheduler

logger = logging.getLogger(__name__)


class LocalNodeProvider(NodeProvider):
    """NodeProvider for automatically managed nodes locally
    """

    def __init__(self, provider_config, cluster_name):
        NodeProvider.__init__(self, provider_config, cluster_name)
        self.local_scheduler = LocalScheduler(
            provider_config, cluster_name)

    def non_terminated_nodes(self, tag_filters):
        # Only get the non terminated nodes associated with this cluster name.
        tag_filters[CLOUDTIK_TAG_CLUSTER_NAME] = self.cluster_name
        tag_filters[CLOUDTIK_TAG_WORKSPACE_NAME] = self.provider_config["workspace_name"]
        return self.local_scheduler.non_terminated_nodes(tag_filters)

    def is_running(self, node_id):
        return self.local_scheduler.is_running(node_id)

    def is_terminated(self, node_id):
        return self.local_scheduler.is_terminated(node_id)

    def node_tags(self, node_id):
        return self.local_scheduler.node_tags(node_id)

    def external_ip(self, node_id):
        return None

    def internal_ip(self, node_id):
        return self.local_scheduler.internal_ip(node_id)

    def create_node(self, node_config, tags, count):
        # Tag the newly created node with this cluster name. Helps to get
        # the right nodes when calling non_terminated_nodes.
        tags[CLOUDTIK_TAG_CLUSTER_NAME] = self.cluster_name
        tags[CLOUDTIK_TAG_WORKSPACE_NAME] = self.provider_config["workspace_name"]
        self.local_scheduler.create_node(
            node_config, tags, count)

    def set_node_tags(self, node_id, tags):
        self.local_scheduler.set_node_tags(node_id, tags)

    def terminate_node(self, node_id):
        self.local_scheduler.terminate_node(node_id)

    def get_node_info(self, node_id):
        return self.local_scheduler.get_node_info(node_id)

    def with_environment_variables(
            self, node_type_config: Dict[str, Any], node_id: str):
        return {}

    def get_command_executor(self,
                             call_context: CallContext,
                             log_prefix: str,
                             node_id: str,
                             auth_config: Dict[str, Any],
                             cluster_name: str,
                             process_runner: ModuleType,
                             use_internal_ip: bool,
                             docker_config: Optional[Dict[str, Any]] = None
                             ) -> CommandExecutor:

        return self.local_scheduler.get_command_executor(
            call_context=call_context,
            log_prefix=log_prefix,
            node_id=node_id,
            auth_config=auth_config,
            cluster_name=cluster_name,
            process_runner=process_runner,
            use_internal_ip=use_internal_ip,
            docker_config=docker_config
        )

    def prepare_config_for_head(
            self, cluster_config: Dict[str, Any],
            remote_config: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a new cluster config with custom configs for head node."""
        # Set in cluster flag
        remote_config["provider"]["local_in_cluster"] = True
        return remote_config

    @staticmethod
    def bootstrap_config(cluster_config):
        return bootstrap_local(cluster_config)

    @staticmethod
    def bootstrap_config_for_api(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        return bootstrap_local_for_api(cluster_config)

    @staticmethod
    def prepare_config(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        return prepare_local(cluster_config)

    @staticmethod
    def post_prepare(
            cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Fills out missing fields after the user config is merged
        with defaults and before validate"""
        return post_prepare_local(cluster_config)
