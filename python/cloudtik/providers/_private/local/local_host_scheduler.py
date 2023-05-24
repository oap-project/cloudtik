import logging
from types import ModuleType
from typing import Dict, Optional, Any

from cloudtik.core._private.call_context import CallContext
from cloudtik.core._private.command_executor.local_command_executor import LocalCommandExecutor
from cloudtik.core.command_executor import CommandExecutor
from cloudtik.core.tags import CLOUDTIK_TAG_CLUSTER_NAME
from cloudtik.providers._private.local.config \
    import get_host_scheduler_lock_path, get_host_scheduler_state_path, _get_request_instance_type, TAG_WORKSPACE_NAME
from cloudtik.providers._private.local.local_scheduler import LocalScheduler
from cloudtik.providers._private.local.state_store import LocalHostStateStore
from cloudtik.providers._private.local.utils import _get_node_info

logger = logging.getLogger(__name__)


class LocalHostScheduler(LocalScheduler):
    def __init__(self, provider_config, cluster_name):
        LocalScheduler.__init__(self, provider_config, cluster_name)
        self.state = LocalHostStateStore(
            get_host_scheduler_lock_path(),
            get_host_scheduler_state_path())

    def create_node(self, node_config, tags, count):
        launched = 0
        instance_type = _get_request_instance_type(node_config)
        with self.state.ctx:
            nodes = self.state.get_nodes_safe()
            for node_id, node in nodes.items():
                if node["state"] != "terminated":
                    continue

                node["tags"] = tags
                node["state"] = "running"
                node["instance_type"] = instance_type
                self.state.put_node_safe(node_id, node)
                launched = launched + 1
                if count == launched:
                    return
        if launched < count:
            raise RuntimeError(
                "No enough free nodes. {} nodes requested / {} launched.".format(
                    count, launched))

    def _list_nodes(self, tag_filters):
        # List nodes that are not cluster specific, ignoring the cluster name
        nodes = self.state.get_nodes()
        matching_nodes = []
        for node_id, node in nodes.items():
            if node["state"] == "terminated":
                continue
            ok = True
            for k, v in tag_filters.items():
                if node["tags"].get(k) != v:
                    ok = False
                    break
            if ok:
                matching_nodes.append(node)
        return matching_nodes

    def non_terminated_nodes(self, tag_filters):
        tag_filters = {} if tag_filters is None else tag_filters
        if self.cluster_name:
            tag_filters[CLOUDTIK_TAG_CLUSTER_NAME] = self.cluster_name
        matching_nodes = self._list_nodes(tag_filters)
        return [node["name"] for node in matching_nodes]

    def is_running(self, node_id):
        node = self.state.get_node(node_id)
        return node["state"] == "running" if node else False

    def is_terminated(self, node_id):
        return not self.is_running(node_id)

    def node_tags(self, node_id):
        node = self.state.get_node(node_id)
        return node["tags"] if node else None

    def internal_ip(self, node_id):
        node = self.state.get_node(node_id)
        return node.get("ip")

    def set_node_tags(self, node_id, tags):
        self.state.set_node_tags(node_id, tags, False)

    def terminate_node(self, node_id):
        node = self.state.get_node(node_id)
        if node is None:
            raise RuntimeError("Node with id {} doesn't exist.".format(node_id))
        if node["state"] != "running":
            raise RuntimeError("Node with id {} is not running.".format(node_id))

        node["state"] = "terminated"
        self.state.put_node(node_id, node)

    def get_node_info(self, node_id):
        node = self.state.get_node(node_id)
        if node is None:
            raise RuntimeError("Node with id {} doesn't exist.".format(node_id))
        return _get_node_info(node)

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
            "auth_config": auth_config,
            "cluster_name": cluster_name,
            "process_runner": process_runner,
            "use_internal_ip": use_internal_ip,
            "provider": self,
            "node_id": node_id,
        }
        return LocalCommandExecutor(call_context, **common_args)

    def create_workspace(self, workspace_name):
        self.state.create_workspace(workspace_name)
        return {"name": workspace_name}

    def delete_workspace(self, workspace_name):
        self.state.delete_workspace(workspace_name)
        return {"name": workspace_name}

    def get_workspace(self, workspace_name):
        return self.state.get_workspace(workspace_name)

    def list_nodes(self, workspace_name, tag_filters):
        # List nodes that are not cluster specific, ignoring the cluster name
        tag_filters = {} if tag_filters is None else tag_filters
        tag_filters[TAG_WORKSPACE_NAME] = workspace_name
        return self._list_nodes(tag_filters)
