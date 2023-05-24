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

    def get_nodes(self):
        raise NotImplementedError

    def get_node(self, node_id):
        raise NotImplementedError

    def put_node(self, node_id, node):
        raise NotImplementedError

    def transaction(self):
        """Open and return a transaction object which can be used by with statement"""
        raise NotImplementedError

    def get_nodes_safe(self):
        # already in transaction, no need to handle lock
        raise NotImplementedError

    def get_node_safe(self, node_id):
        # already in transaction, no need to handle lock
        raise NotImplementedError

    def put_node_safe(self, node_id, node):
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


class TransactionContext(object):
    def __init__(self, lock_path):
        self.lock = RLock()
        self.file_lock = FileLock(lock_path)

    def __enter__(self):
        self.lock.acquire()
        self.file_lock.acquire()
        return self

    def __exit__(self, *args):
        self.file_lock.release()
        self.lock.release()


class FileStateStore(StateStore):
    def __init__(self, provider_config, lock_path, state_path):
        super().__init__(provider_config)

        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        os.makedirs(os.path.dirname(state_path), exist_ok=True)

        self.ctx = TransactionContext(lock_path)
        self.state_path = state_path
        self.cached_state = {}
        self._load_config(provider_config)

    def get_nodes(self):
        with self.ctx:
            return copy.deepcopy(self.get_nodes_safe())

    def get_node(self, node_id):
        with self.ctx:
            node = self.get_node_safe(node_id)
            if node is None:
                return node
            return copy.deepcopy(node)

    def put_node(self, node_id, node):
        assert "tags" in node
        assert "state" in node
        with self.ctx:
            self.put_node_safe(node_id, node)

    def transaction(self):
        return self.ctx

    def get_nodes_safe(self):
        return self.cached_state["nodes"]

    def get_node_safe(self, node_id):
        nodes = self.get_nodes_safe()
        if node_id not in nodes:
            return None
        return nodes[node_id]

    def put_node_safe(self, node_id, node):
        nodes = self.get_nodes_safe()
        nodes[node_id] = node
        self._save()

    def load_config(self, provider_config):
        with self.ctx:
            self._load_config(provider_config)

    def _load(self):
        if os.path.exists(self.state_path):
            state = json.loads(open(self.state_path).read())
        else:
            state = {}

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Loaded cluster state: {}".format(state))

        if "nodes" not in state:
            state["nodes"] = {}
        self.cached_state = state
        return state

    def _save(self):
        state = self.cached_state
        with open(self.state_path, "w") as f:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Writing cluster state: {}".format(state))
            f.write(json.dumps(state))

    def _load_config(self, provider_config):
        state = self._load()
        nodes = state["nodes"]

        list_of_node_ips = get_list_of_node_ips(provider_config)

        # Filter removed node ips.
        for node_ip in list(nodes):
            if node_ip not in list_of_node_ips:
                node = nodes[node_ip]
                # remove node only if it is terminated (not in use)
                if node == "terminated":
                    del nodes[node_ip]

        # new nodes set to terminated
        for node_ip in list_of_node_ips:
            if node_ip not in nodes:
                nodes[node_ip] = {
                    "tags": {},
                    "state": "terminated",
                }
        assert len(nodes) == len(list_of_node_ips)
        self._save()

    def _get_workspaces(self):
        if "workspaces" not in self.cached_state:
            self.cached_state["workspaces"] = {}
        return self.cached_state["workspaces"]

    def create_workspace(self, workspace_name):
        with self.ctx:
            workspaces = self._get_workspaces()
            if workspace_name in workspaces:
                raise RuntimeError(
                    "Workspace with name {} already exists.".format(workspace_name))
            workspaces[workspace_name] = {
                "name": workspace_name
            }
            self._save()

    def delete_workspace(self, workspace_name):
        with self.ctx:
            workspaces = self._get_workspaces()
            if workspace_name not in workspaces:
                raise RuntimeError(
                    "Workspace with name {} doesn't exist.".format(workspace_name))
            workspaces.pop(workspace_name)
            self._save()

    def get_workspace(self, workspace_name):
        with self.ctx:
            workspaces = self._get_workspaces()
            if workspace_name not in workspaces:
                return None
            return copy.deepcopy(workspaces[workspace_name])
