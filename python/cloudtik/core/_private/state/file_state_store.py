import json
import logging
import os
from threading import RLock

from filelock import FileLock

logger = logging.getLogger(__name__)


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


class FileStateStore:
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
        updated = False
        if not valid_ids:
            if nodes:
                state["nodes"] = {}
                updated = True
        else:
            for node_id in list(nodes):
                if node_id not in valid_ids:
                    del nodes[node_id]
                    updated = True
        if updated:
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
            self.update_node_tags(node, tags)
            self._save()

    def get_node_tags(self, node_id):
        node = self.get_node(node_id)
        if node is None or "tags" not in node:
            return {}
        return node["tags"]

    def _load(self):
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                state = json.loads(f.read())
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

    @staticmethod
    def update_node_tags(node, tags):
        if "tags" not in node:
            node["tags"] = tags
        else:
            node["tags"].update(tags)
