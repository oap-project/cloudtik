import logging
from types import ModuleType
from typing import Dict, Optional, Any

from cloudtik.core._private.call_context import CallContext
from cloudtik.core._private.command_executor.docker_command_executor import DockerCommandExecutor
from cloudtik.core.command_executor import CommandExecutor
from cloudtik.providers._private.local.config import \
    _get_bridge_address
from cloudtik.providers._private.local.local_scheduler import LocalScheduler

logger = logging.getLogger(__name__)


class LocalContainerScheduler(LocalScheduler):
    def __init__(self, provider_config):
        LocalScheduler.__init__(self, provider_config)
        self.bridge_address = _get_bridge_address(provider_config)

    def create_node(self, cluster_name, node_config, tags, count):
        # TODO
        pass

    def get_non_terminated_nodes(self, tag_filters):
        # TODO
        pass

    def is_running(self, node_id):
        # TODO
        pass

    def is_terminated(self, node_id):
        # TODO
        pass

    def get_node_tags(self, node_id):
        # TODO
        pass

    def get_internal_ip(self, node_id):
        # TODO
        pass

    def set_node_tags(self, node_id, tags):
        # TODO
        pass

    def terminate_node(self, node_id):
        # TODO
        pass

    def get_node_info(self, node_id):
        # TODO
        pass

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
        common_args = {
            "log_prefix": log_prefix,
            "node_id": node_id,
            "provider": self,
            "auth_config": auth_config,
            "cluster_name": cluster_name,
            "process_runner": process_runner,
            "use_internal_ip": use_internal_ip
        }
        return DockerCommandExecutor(
            call_context, docker_config, True, **common_args)
