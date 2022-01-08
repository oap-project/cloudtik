import logging

from cloudtik.core._private.state.redis_shards_client import \
    RedisShardsClient, generate_match_pattern, generate_redis_key
from cloudtik.core._private.state.redis_shards_scanner import RedisShardsScanner

logger = logging.getLogger(__name__)


class StoreClient:
    def __init__(self, redis_shards_client: RedisShardsClient):
        self._redis_shards_client = redis_shards_client

    def put(self, table_name, key, value):
        redis_key = generate_redis_key(table_name, key)
        redis_shard = self._redis_shards_client.get_shard(redis_key)
        redis_shard.put(redis_key, value)

    def get(self, table_name, key):
        redis_key = generate_redis_key(table_name, key)
        redis_shard = self._redis_shards_client.get_shard(redis_key)
        return redis_shard.gut(redis_key)

    def delete(self, table_name, key):
        redis_key = generate_redis_key(table_name, key)
        redis_shard = self._redis_shards_client.get_shard(redis_key)
        redis_shard.delete(redis_key)

    def get_all(self, table_name):
        match_pattern = generate_match_pattern(table_name)
        scanner = RedisShardsScanner(self._redis_shards_client, table_name)
        return scanner.scan_keys_and_values(match_pattern)



