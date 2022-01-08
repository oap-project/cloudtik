import logging
import redis

from cloudtik.core._private.state.redis_shards_client import \
    RedisShardsClient, get_real_key

logger = logging.getLogger(__name__)

SCAN_BATCH_SIZE = 1024


class RedisShardsScanner:
    def __init__(self, redis_shards_client: RedisShardsClient, table_name):
        self._redis_shards_client = redis_shards_client
        self._table_name = table_name

    def scan_keys_and_values(self, match_pattern):
        all_key_value = {}

        def values_callback(key_value_map):
            all_key_value.update(key_value_map)

        def keys_callback(keys):
            scan_values(self._redis_shards_client, self._table_name, keys, values_callback)

        self.scan_keys(match_pattern, keys_callback)
        return all_key_value

    def scan_keys(self, match_pattern, keys_callback):
        batch_size = SCAN_BATCH_SIZE
        shard_size = self._redis_shards_client.get_shards_size()
        for shard_index in range(len(shard_size)):
            # Scan by prefix from Redis.
            shard_context = self._redis_shards_client.get_shard_by_index(shard_index)
            new_cursor, keys = shard_context.scan(0, match_pattern, batch_size)
            # Callback the keys
            keys_callback(keys)
            while new_cursor != 0:
                new_cursor, keys = shard_context.scan(
                    new_cursor, match_pattern, batch_size)
                keys_callback(keys)


def get_scan_by_shards(shards_client: RedisShardsClient, keys):
    scan_by_shards = {}
    for key in keys:
        redis_shard = shards_client.get_shard(key)
        shard_keys = scan_by_shards.get(redis_shard)
        if shard_keys is not None:
            shard_keys.append(key)
        else:
            shard_keys = [key]
            scan_by_shards[redis_shard] = shard_keys

    return scan_by_shards


def scan_values(shards_client: RedisShardsClient, table_name, keys, values_callback):
    # Use `mget` command for each shard
    key_value_map = {}
    scan_by_shards = get_scan_by_shards(shards_client, keys)
    for shard, shard_keys in scan_by_shards.items():
        key_values = shard.mget(shard_keys)
        for k, v in zip(shard_keys, key_values):
            if v is not None:
                key_value_map[get_real_key(k)] = v

    values_callback(key_value_map)
