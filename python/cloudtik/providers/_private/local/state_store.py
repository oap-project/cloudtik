import copy
import logging
import socket

from cloudtik.core._private.state.file_state_store import FileStateStore

logger = logging.getLogger(__name__)


class LocalHostStateStore(FileStateStore):
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
