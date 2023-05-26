import copy
import logging
import socket

from cloudtik.core._private.state.file_state_store import FileStateStore
from cloudtik.providers._private.local.config import get_all_node_ips

logger = logging.getLogger(__name__)


class LocalStateStore(FileStateStore):
    def __init__(self, lock_path, state_path, provider_config):
        super().__init__(lock_path, state_path)
        self.provider_config = provider_config
        self._init_state()

    def _init_state(self):
        with self.ctx:
            state = self._load()
            nodes = state["nodes"]

            # Filter removed node ips.
            list_of_node_ids = get_all_node_ips(self.provider_config)
            for node_id in list(nodes):
                if node_id not in list_of_node_ids:
                    node = nodes[node_id]
                    # remove node only if it is terminated (not in use)
                    if node == "terminated":
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
