import logging
import redis

from cloudtik.core._private.state.redis_shards_client import RedisShardsClient

logger = logging.getLogger(__name__)


class RedisShardsScanner:
    def __init__(self, redis_shards_client: RedisShardsClient, table_name):
        self._redis_shards_client = redis_shards_client
        self._table_name = table_name

    def scan_keys_and_values(self):
        # TODO: Implement the scan keys and value from the shards
        pass
