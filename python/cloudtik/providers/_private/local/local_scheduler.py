import copy
import logging
import os
from threading import RLock
from types import ModuleType
from typing import Dict, Optional, Any

from cloudtik.core._private.call_context import CallContext
from cloudtik.core._private.command_executor.docker_command_executor import DockerCommandExecutor
from cloudtik.core._private.command_executor.local_command_executor import LocalCommandExecutor
from cloudtik.core._private.command_executor.ssh_command_executor import SSHCommandExecutor
from cloudtik.core._private.core_utils import get_ip_by_name
from cloudtik.core._private.state.file_state_store import FileStateStore
from cloudtik.core._private.utils import is_head_node_by_tags
from cloudtik.core.command_executor import CommandExecutor
from cloudtik.core.tags import CLOUDTIK_TAG_CLUSTER_NAME, CLOUDTIK_TAG_WORKSPACE_NAME
from cloudtik.providers._private.local.config \
    import get_local_scheduler_lock_path, get_local_scheduler_state_path, _get_request_instance_type, \
    _get_node_instance_type, _get_node_id_mapping, get_local_scheduler_state_file_name, get_state_path
from cloudtik.providers._private.local.local_docker_command_executor import LocalDockerCommandExecutor
from cloudtik.providers._private.local.state_store import LocalStateStore
from cloudtik.providers._private.local.utils import _get_node_info

logger = logging.getLogger(__name__)


STATE_MOUNT_PATH = "/cloudtik/state"


class LocalScheduler:
    def __init__(self, provider_config, cluster_name):
        self.provider_config = provider_config
        self.cluster_name = cluster_name

        self.lock = RLock()
        # Cache of node objects from the last nodes() call. This avoids
        # excessive read from state file
        self.cached_nodes: Dict[str, Any] = {}

        # workspace_name must be set in the provider config
        # when cluster name exists
        if self._is_in_cluster():
            self.state = self._get_state_store_in_cluster()
        else:
            self.state = self._get_state_store()

        # This is needed by create node, which is not needed for other cases
        if not self._is_from_workspace():
            self.node_id_mapping = _get_node_id_mapping(provider_config)
        else:
            self.node_id_mapping = None

    def create_node(self, node_config, tags, count):
        launched = 0
        instance_type = _get_request_instance_type(node_config)
        with self.lock:
            with self.state.ctx:
                nodes = self.state.get_nodes_safe()
                if is_head_node_by_tags(tags):
                    # for local provider, we do special handling for head
                    launched = self._launch_node(
                        nodes, tags, count, launched, instance_type,
                        as_head=True)
                    if count == launched:
                        return
                else:
                    launched = self._launch_node(
                        nodes, tags, count, launched, instance_type)
                    if count == launched:
                        return
        if launched < count:
            raise RuntimeError(
                "No enough free nodes. {} nodes requested / {} launched.".format(
                    count, launched))

    def _launch_node(
            self, nodes, tags, count, launched,
            instance_type, as_head=False):
        for node_id, node in nodes.items():
            if node["state"] != "terminated":
                continue

            if as_head:
                # we don't need to check instance_type
                if node_id != self._get_local_node_id():
                    continue
            else:
                node_instance_type = self.get_node_instance_type(node_id)
                if instance_type != node_instance_type:
                    continue

            node["tags"] = tags
            node["state"] = "running"
            node["instance_type"] = instance_type
            self.state.put_node_safe(node_id, node)
            launched = launched + 1
            if count == launched:
                return launched
        return launched

    def get_node_instance_type(self, node_id):
        return _get_node_instance_type(self.node_id_mapping, node_id)

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
        # list nodes is thread safe, put it out of the lock block
        matching_nodes = self._list_nodes(tag_filters)
        with self.lock:
            self.cached_nodes = {
                n["name"]: n for n in matching_nodes
            }
            return [node["name"] for node in matching_nodes]

    def is_running(self, node_id):
        # always get current status
        node = self._get_node(node_id=node_id)
        return node["state"] == "running" if node else False

    def is_terminated(self, node_id):
        # always get current status
        return not self.is_running(node_id)

    def node_tags(self, node_id):
        with self.lock:
            node = self._get_cached_node(node_id)
            return node.get("tags", {}) if node else {}

    def internal_ip(self, node_id):
        return get_ip_by_name(node_id)

    def set_node_tags(self, node_id, tags):
        with self.lock:
            node = self._get_cached_node(node_id)
            self._set_node_tags(node_id, tags)
            # update the cached node tags, although it will refresh at next non_terminated_nodes
            FileStateStore.update_node_tags(node, tags)

    def terminate_node(self, node_id):
        with self.lock:
            node = self._get_cached_node(node_id)
            if node["state"] != "running":
                raise RuntimeError("Node with id {} is not running.".format(node_id))

            node["state"] = "terminated"
            self.state.put_node(node_id, node)

    def get_node_info(self, node_id):
        with self.lock:
            node = self._get_cached_node(node_id)
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

        if self._is_in_cluster() or (
                node_id and node_id != self._get_local_node_id()):
            if self._is_docker_enabled(docker_config):
                # local node may have special with sudo docker settings
                if node_id == self._get_local_node_id():
                    # local node, handle local node docker sudo
                    docker_config = copy.deepcopy(docker_config)
                    docker_config["docker_with_sudo"] = self.provider_config.get(
                        "docker_with_sudo", False)
                return DockerCommandExecutor(
                    call_context, docker_config, True, **common_args)
            else:
                return SSHCommandExecutor(call_context, **common_args)
        else:
            # not in cluster and to local node
            # local node, handle local node docker sudo
            if self._is_docker_enabled(docker_config):
                docker_config = copy.deepcopy(docker_config)
                docker_config["docker_with_sudo"] = self.provider_config.get(
                    "docker_with_sudo", False)
                # on local host, we don't need mounts mapping too
                docker_config["mounts_mapping"] = False
                local_file_mounts = self._get_local_file_mounts()
                return LocalDockerCommandExecutor(
                    call_context, docker_config, False,
                    local_file_mounts=local_file_mounts,
                    **common_args)
            else:
                return LocalCommandExecutor(call_context, **common_args)

    def list_nodes(self, workspace_name, tag_filters):
        # List nodes that are not cluster specific, ignoring the cluster name
        tag_filters = {} if tag_filters is None else tag_filters
        tag_filters[CLOUDTIK_TAG_WORKSPACE_NAME] = workspace_name
        return self._list_nodes(tag_filters)

    def _is_in_cluster(self):
        # flag set at boostrap cluster config
        return self.provider_config.get("local_in_cluster", False)

    def _is_from_workspace(self):
        return self.provider_config.get(
            "from_workspace", False)

    @staticmethod
    def _is_docker_enabled(docker_config):
        return True if docker_config and docker_config.get(
            "enabled", False) else False

    def _get_state_store(self):
        init_and_validate = False if self._is_from_workspace() else True
        return LocalStateStore(
            get_local_scheduler_lock_path(),
            get_local_scheduler_state_path(),
            self.provider_config,
            init_and_validate=init_and_validate
        )

    def _get_state_store_in_cluster(self):
        local_scheduler_state_path = os.path.join(
            STATE_MOUNT_PATH, get_local_scheduler_state_file_name())
        return LocalStateStore(
            get_local_scheduler_lock_path(),
            local_scheduler_state_path,
            self.provider_config,
            init_and_validate=False)

    def _set_node_tags(self, node_id, tags):
        # Will use state: set labels to state
        self.state.set_node_tags(node_id, tags, False)

    def _get_cached_node(self, node_id: str):
        if node_id in self.cached_nodes:
            return self.cached_nodes[node_id]

        return self._get_node(node_id)

    def _get_node(self, node_id: str):
        self.non_terminated_nodes({})  # Side effect: updates cache
        with self.lock:
            if node_id in self.cached_nodes:
                return self.cached_nodes[node_id]

            node = self.state.get_node(node_id)
            if node is None:
                raise RuntimeError("No node found with id: {}.")
            return node

    def _get_local_node_id(self):
        return self.provider_config.get("local_node_id")

    @staticmethod
    def _get_local_file_mounts():
        # Handling mounts
        # We need add mount by default the cluster state path: to STATE_MOUNT_PATH

        file_mounts = {
            STATE_MOUNT_PATH: get_state_path()
        }

        return file_mounts
