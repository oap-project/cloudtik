import logging

from cloudtik.core._private.state.redis_shards_client import RedisShardsClient
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
        super().__init__(store_client, "node_table")


class StateTableStore:
    """
    Class wraps the access of all the table tables from Redis sharding
    """

    def __init__(self, redis_shards_client: RedisShardsClient):
        self._store_client = StoreClient(redis_shards_client)
        self._node_table = NodeStateTable(self._store_client)
        self._user_state_tables = {}

    def get_node_table(self) -> NodeStateTable:
        return self._node_table

    def get_user_state_table(self, table_name) -> StateTable:
        user_state_table = self._user_state_tables.get(table_name)
        if user_state_table is not None:
            return user_state_table

        user_state_table = StateTable(self._store_client, table_name)
        self._user_state_tables[table_name] = user_state_table
        return user_state_table
