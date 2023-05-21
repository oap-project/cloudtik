import copy
import json
import logging
import os
from threading import RLock

from filelock import FileLock

from cloudtik.providers._private.onprem.config import get_list_of_node_ips

logger = logging.getLogger(__name__)


class StateStore:
    def __init__(self, provider_config):
        self.provider_config = provider_config

    def get(self):
        raise NotImplementedError

    def get_node(self, node_id):
        raise NotImplementedError

    def put(self, node_id, node):
        raise NotImplementedError

    def transaction(self):
        """Open and return a transaction object which can be used by with statement"""
        raise NotImplementedError

    def get_safe(self):
        # already in transaction, no need to handle lock
        raise NotImplementedError

    def get_node_safe(self, node_id):
        # already in transaction, no need to handle lock
        raise NotImplementedError

    def put_safe(self, node_id, node):
        # already in transaction, no need to handle lock
        raise NotImplementedError

    def load_config(self, provider_config):
        raise NotImplementedError

    def create_workspace(self, workspace_name):
        raise NotImplementedError

    def delete_workspace(self, workspace_name):
        raise NotImplementedError

    def get_workspace(self, workspace_name):
        raise NotImplementedError


class FileStateStore(StateStore):
    def __init__(self, provider_config, lock_path, state_path):
        super().__init__(provider_config)

        self.lock = RLock()
        self.file_lock = FileLock(lock_path)
        self.state_path = state_path
        self.cached_state = {}
        self._load_config(provider_config)

    def get(self):
        with self.lock:
            with self.file_lock:
                return copy.deepcopy(self.get_safe())

    def get_node(self, node_id):
        with self.lock:
            with self.file_lock:
                node = self.get_node_safe(node_id)
                if node is None:
                    return node
                return copy.deepcopy(node)

    def put(self, node_id, node):
        assert "tags" in node
        assert "state" in node
        with self.lock:
            with self.file_lock:
                self.put_safe(node_id, node)

    def transaction(self):
        return self.file_lock

    def get_safe(self):
        return self.cached_state["nodes"]

    def get_node_safe(self, node_id):
        nodes = self.get_safe()
        if node_id not in nodes:
            return None
        return nodes[node_id]

    def put_safe(self, node_id, node):
        nodes = self.get_safe()
        nodes[node_id] = node
        self._save_state()

    def load_config(self, provider_config):
        with self.lock:
            with self.file_lock:
                self._load_config(provider_config)

    def _save_state(self):
        state = self.cached_state
        with open(self.state_path, "w") as f:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Writing cluster state: {}".format(state))
            f.write(json.dumps(state))

    def _load_config(self, provider_config):
        list_of_node_ips = get_list_of_node_ips(provider_config)
        if os.path.exists(self.state_path):
            state = json.loads(open(self.state_path).read())
        else:
            state = {}
        logger.info(
            "Loaded cluster state: {}".format(state))

        if "nodes" not in state:
            state["nodes"] = {}
        nodes = state["nodes"]
        # Filter removed node ips.
        for node_ip in list(nodes):
            if node_ip not in list_of_node_ips:
                del nodes[node_ip]

        for node_ip in list_of_node_ips:
            if node_ip not in nodes:
                nodes[node_ip] = {
                    "tags": {},
                    "state": "terminated",
                }
        assert len(nodes) == len(list_of_node_ips)
        logger.info(
            "Initial cluster state: {}".format(state))
        self.cached_state = state
        self._save_state()

    def _get_workspaces(self):
        if "workspaces" not in self.cached_state:
            self.cached_state["workspaces"] = {}
        return self.cached_state["workspaces"]

    def create_workspace(self, workspace_name):
        with self.lock:
            with self.file_lock:
                workspaces = self._get_workspaces()
                if workspace_name in workspaces:
                    raise RuntimeError(
                        "Workspace with name {} already exists.".format(workspace_name))
                workspaces[workspace_name] = {
                    "name": workspace_name
                }

    def delete_workspace(self, workspace_name):
        with self.lock:
            with self.file_lock:
                workspaces = self._get_workspaces()
                if workspace_name not in workspaces:
                    raise RuntimeError(
                        "Workspace with name {} doesn't exist.".format(workspace_name))
                workspaces.pop(workspace_name)

    def get_workspace(self, workspace_name):
        with self.lock:
            with self.file_lock:
                workspaces = self._get_workspaces()
                if workspace_name not in workspaces:
                    return None
                return copy.deepcopy(workspaces[workspace_name])
