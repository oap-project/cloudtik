import logging
from types import ModuleType
from typing import Dict, Optional, Any

from cloudtik.core._private.call_context import CallContext
from cloudtik.core.command_executor import CommandExecutor

logger = logging.getLogger(__name__)


class LocalScheduler:
    def __init__(self, provider_config):
        self.provider_config = provider_config

    def create_node(self, cluster_name, node_config, tags, count):
        raise NotImplementedError

    def get_non_terminated_nodes(self, tag_filters):
        raise NotImplementedError

    def is_running(self, node_id):
        raise NotImplementedError

    def is_terminated(self, node_id):
        raise NotImplementedError

    def get_node_tags(self, node_id):
        raise NotImplementedError

    def get_internal_ip(self, node_id):
        raise NotImplementedError

    def set_node_tags(self, node_id, tags):
        raise NotImplementedError

    def terminate_node(self, node_id):
        raise NotImplementedError

    def get_node_info(self, node_id):
        raise NotImplementedError

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
        raise NotImplementedError
