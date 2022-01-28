import logging


from cloudtik.core._private.state.state_table_store import StateTableStore

logger = logging.getLogger(__name__)


class StateNodeManager:
    """
    Manager class for node information
    """

    def __init__(self, state_table_store: StateTableStore):
        self._state_table_store = state_table_store

    def register_node(self, node_id, node_info):
        self._state_table_store.get_node_table().put(node_id, node_info)

    def drain_node(self, node_id):
        # TODO: update node table info to DEAD instead of delete it.
        self._state_table_store.get_node_table().delete(node_id)

    def get_node_table(self):
        self._state_table_store.get_node_table().get_all()
