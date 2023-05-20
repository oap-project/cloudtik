import socket
from types import ModuleType
from typing import Dict, Optional, Any
from filelock import FileLock
from threading import RLock
import json
import os
import logging

from cloudtik.core._private.call_context import CallContext
from cloudtik.core._private.command_executor.local_command_executor import LocalCommandExecutor
from cloudtik.core.command_executor import CommandExecutor
from cloudtik.providers._private.local.config \
    import get_host_scheduler_lock_path, get_host_scheduler_state_path, _get_request_instance_type, \
    get_instance_type_name
from cloudtik.providers._private.local.local_scheduler import LocalScheduler

logger = logging.getLogger(__name__)

LOCAL_HOST_ID = "localhost"


class LocalHostState:
    def __init__(self, lock_path, save_path, provider_config):
        self.lock = RLock()
        self.file_lock = FileLock(lock_path)
        self.state_path = save_path
        self._init_state()

    def _init_state(self):
        with self.lock:
            with self.file_lock:
                list_of_node_ids = [LOCAL_HOST_ID]
                if os.path.exists(self.state_path):
                    nodes = json.loads(open(self.state_path).read())
                else:
                    nodes = {}
                logger.info("Loaded cluster state: {}".format(nodes))

                # Filter removed node ips.
                for node_id in list(nodes):
                    if node_id not in list_of_node_ids:
                        del nodes[node_id]

                for node_id in list_of_node_ids:
                    if node_id not in nodes:
                        nodes[node_id] = {
                            "tags": {},
                            "state": "terminated",
                        }
                assert len(nodes) == len(list_of_node_ids)
                with open(self.state_path, "w") as f:
                    logger.info(
                        "Writing cluster state: {}".format(nodes))
                    f.write(json.dumps(nodes))

    def _load(self):
        return json.loads(open(self.state_path).read())

    def get(self):
        with self.lock:
            with self.file_lock:
                return self._load()

    def get_node(self, node_id):
        nodes = self.get()
        if node_id not in nodes:
            return None
        return nodes[node_id]

    def put(self, node_id, node):
        assert "tags" in node
        assert "state" in node
        with self.lock:
            with self.file_lock:
                nodes = self._load()
                nodes[node_id] = node
                with open(self.state_path, "w") as f:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Writing cluster state: {}".format(list(nodes)))
                    f.write(json.dumps(nodes))


class LocalHostScheduler(LocalScheduler):
    def __init__(self, provider_config):
        LocalScheduler.__init__(self, provider_config)
        self.state = LocalHostState(
            get_host_scheduler_lock_path(), get_host_scheduler_state_path(),
            provider_config)

    def create_node(self, cluster_name, node_config, tags, count):
        launched = 0
        instance_type = _get_request_instance_type(node_config)
        with self.state.file_lock:
            nodes = self.state.get()
            for node_id, node in nodes.items():
                if node["state"] != "terminated":
                    continue

                node["tags"] = tags
                node["state"] = "running"
                node["instance_type"] = instance_type
                self.state.put(node_id, node)
                launched = launched + 1
                if count == launched:
                    return
        if launched < count:
            raise RuntimeError(
                "No enough free nodes. {} nodes requested / {} launched.".format(
                    count, launched))

    def get_non_terminated_nodes(self, tag_filters):
        nodes = self.state.get()
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
                matching_nodes.append(node_id)
        return matching_nodes

    def is_running(self, node_id):
        node = self.state.get_node(node_id)
        return node["state"] == "running" if node else False

    def is_terminated(self, node_id):
        return not self.is_running(node_id)

    def get_node_tags(self, node_id):
        node = self.state.get_node(node_id)
        return node["tags"] if node else None

    def get_internal_ip(self, node_id):
        return socket.gethostbyname(socket.gethostname())

    def set_node_tags(self, node_id, tags):
        node = self.state.get_node(node_id)
        if node is None:
            raise RuntimeError("Node with id {} doesn't exist.".format(node_id))

        node["tags"].update(tags)
        self.state.put(node_id, node)

    def terminate_node(self, node_id):
        node = self.state.get_node(node_id)
        if node is None:
            raise RuntimeError("Node with id {} doesn't exist.".format(node_id))
        if node["state"] != "running":
            raise RuntimeError("Node with id {} is not running.".format(node_id))

        node["state"] = "terminated"
        self.state.put(node_id, node)

    def get_node_info(self, node_id):
        node = self.state.get_node(node_id)
        if node is None:
            raise RuntimeError("Node with id {} doesn't exist.".format(node_id))
        instance_type = node.get("instance_type", {})
        node_instance_type = get_instance_type_name(instance_type)
        node_info = {"node_id": node_id,
                     "instance_type": node_instance_type,
                     "private_ip": self.get_internal_ip(node_id),
                     "public_ip": self.get_internal_ip(node_id),
                     "instance_status": node["state"]}
        node_info.update(node["tags"])
        return node_info

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
        return LocalCommandExecutor(call_context, **common_args)
