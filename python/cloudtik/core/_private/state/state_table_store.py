import logging


from cloudtik.core._private.state.store_client import StoreClient

logger = logging.getLogger(__name__)


class StateTable:
    def __init__(self, store_client: StoreClient, table_name):
        self._store_client = store_client
        self._table_name = table_name

    def put(self, key, value):
        self._store_client.put(self._table_name, key, value)

    def get(self, key):
        return self._store_client.get(self._table_name, key)

    def get_all(self):
        return self._store_client.get_all(self._table_name)

    def delete(self, key):
        self._store_client.delete(self._table_name, key)


class NodeStateTable(StateTable):
    def __init__(self, store_client: StoreClient):
        super().__init__(self, "node_table", store_client)


class StateTableStore:
    """
    Class wraps the access of all the table tables from Redis sharding
    """

    def __init__(self, store_client: StoreClient):
        self._store_client = store_client
        self._node_table = NodeStateTable(store_client)

    def get_node_table(self):
        return self._node_table

