import copy
import json
import logging
import os
import socket
from threading import RLock

from filelock import FileLock

logger = logging.getLogger(__name__)


def _update_node_tags(node, tags):
    if "tags" not in node:
        node["tags"] = tags
    else:
        node["tags"].update(tags)


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


class LocalStateStore:
    def __init__(self, lock_path, state_path):
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        os.makedirs(os.path.dirname(state_path), exist_ok=True)

        self.ctx = TransactionContext(lock_path)
        self.state_path = state_path
        # this state object is just for convenience to pass among load and save
        self.state = None

    def get_nodes(self):
        with self.ctx:
            return self.get_nodes_safe()

    def get_nodes_safe(self):
        state = self._load()
        nodes = state["nodes"]
        return nodes

    def get_node(self, node_id):
        with self.ctx:
            return self.get_node_safe(node_id)

    def get_node_safe(self, node_id):
        nodes = self.get_nodes_safe()
        if node_id not in nodes:
            return None
        return nodes[node_id]

    def put_node(self, node_id, node):
        assert "tags" in node
        with self.ctx:
            self.put_node_safe(node_id, node)

    def put_node_safe(self, node_id, node):
        state = self._load()
        nodes = state["nodes"]
        nodes[node_id] = node
        self._save()

    def remove_node(self, node_id):
        with self.ctx:
            self.remove_node_safe(node_id)

    def remove_node_safe(self, node_id):
        state = self._load()
        nodes = state["nodes"]
        if node_id not in nodes:
            return
        del nodes[node_id]
        self._save()

    def cleanup(self, valid_ids):
        with self.ctx:
            self.cleanup_safe(valid_ids)

    def cleanup_safe(self, valid_ids):
        state = self._load()
        nodes = state["nodes"]
        if not valid_ids:
            state["nodes"] = {}
        else:
            for node_id in list(nodes):
                if node_id not in valid_ids:
                    del nodes[node_id]
        self._save()

    def set_node_tags(self, node_id, tags, non_exists_ok=True):
        with self.ctx:
            state = self._load()
            nodes = state["nodes"]
            if node_id not in nodes:
                if not non_exists_ok:
                    raise RuntimeError("Node with id {} doesn't exist.".format(node_id))
                nodes[node_id] = {}
            node = nodes[node_id]
            _update_node_tags(node, tags)
            self._save()

    def get_node_tags(self, node_id):
        node = self.get_node(node_id)
        if node is None or "tags" not in node:
            return {}
        return node["tags"]

    def _load(self):
        if os.path.exists(self.state_path):
            state = json.loads(open(self.state_path).read())
        else:
            state = {}

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Loaded cluster state: {}".format(state))

        if "nodes" not in state:
            state["nodes"] = {}
        self.state = state
        return state

    def _save(self):
        state = self.state
        with open(self.state_path, "w") as f:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Writing cluster state: {}".format(list(state)))
            f.write(json.dumps(state))


class LocalHostStateStore(LocalStateStore):
    def __init__(self, lock_path, state_path):
        super().__init__(lock_path, state_path)
        self._init_state()

    def _init_state(self):
        with self.ctx:
            state = self._load()
            nodes = state["nodes"]

            # Filter removed node ips.
            local_host_name = socket.gethostname()
            list_of_node_ids = [local_host_name]
            for node_id in list(nodes):
                if node_id not in list_of_node_ids:
                    del nodes[node_id]

            for node_id in list_of_node_ids:
                if node_id not in nodes:
                    nodes[node_id] = {
                        "name": node_id,
                        "ip": socket.gethostbyname(node_id),
                        "tags": {},
                        "state": "terminated",
                    }
            assert len(nodes) == len(list_of_node_ids)
            self._save()

    def _get_workspaces(self):
        state = self._load()
        if "workspaces" not in state:
            state["workspaces"] = {}
        return state["workspaces"]

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


class LocalContainerStateStore(LocalStateStore):
    def __init__(self, lock_path, state_path):
        super().__init__(lock_path, state_path)
        self._init_state()

    def _init_state(self):
        with self.ctx:
            self._load()
            self._save()
