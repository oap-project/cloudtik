import logging
import socket

from cloudtik.core._private.core_utils import get_ip_by_name
from cloudtik.core._private.state.file_state_store import FileStateStore
from cloudtik.providers._private.local.config import get_all_node_ids

logger = logging.getLogger(__name__)


class LocalStateStore(FileStateStore):
    def __init__(self, lock_path, state_path, provider_config, init_and_validate=True):
        super().__init__(lock_path, state_path)
        self.provider_config = provider_config
        self._init_state(init_and_validate)

    def _init_state(self, init_and_validate):
        with self.ctx:
            state = self._load()
            nodes = state["nodes"]

            # For cases that workspace to query information
            # There is no modification of nodes
            if init_and_validate:
                # Filter removed node ips.
                list_of_node_ids = get_all_node_ids(self.provider_config)
                for node_id in list(nodes):
                    if node_id not in list_of_node_ids:
                        node = nodes[node_id]
                        # remove node only if it is terminated (not in use)
                        if node["state"] == "terminated":
                            del nodes[node_id]

                for node_id in list_of_node_ids:
                    if node_id not in nodes:
                        # Don't cache here if we want to handle host IP change
                        node_ip = get_ip_by_name(node_id)
                        nodes[node_id] = {
                            "name": node_id,
                            "ip": node_ip,
                            "tags": {},
                            "state": "terminated",
                        }
            self._save()
