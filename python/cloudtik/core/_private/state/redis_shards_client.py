import logging
import redis

logger = logging.getLogger(__name__)


TABLE_SEPERATOR = ":"


def generate_match_pattern(table_name):
    return table_name + TABLE_SEPERATOR + "*"


def generate_redis_key(table_name, key):
    return table_name + TABLE_SEPERATOR + key


def hash_redis_key(self, redis_key):
    return hash(redis_key)


def get_real_key(key):
    # TODO: get the real key from shard key
    return key


# TODO: Implement operations on a specific redis shard
class RedisShard:
    def __init__(self):
        self._redis_client = None

    def connect(self, redis_address, redis_port, redis_password):
        self._redis_client = redis.StrictRedis(
            host=redis_address, port=redis_port, password=redis_password)

    def put(self, key, value):
        pass

    def get(self, key):
        pass

    def delete(self, key):
        pass

    def mget(self, keys):
        return self._redis_client.mget(keys)

    def scan(self, cursor, match_pattern, batch_size):
        return self._redis_client.scan(cursor, match_pattern, batch_size)


class RedisShardsClient:
    def __init__(self, redis_address, redis_port, redis_password):
        self._redis_address = redis_address
        self._redis_port = redis_port
        self._redis_password = redis_password
        self._is_connected = False
        # TODO: Add initialization here
        self._primary_shard = None
        self._redis_shards = None
        self._shards_count = 1

    def get_primary_shard(self):
        return self._primary_shard

    def get_shard(self, redis_key):
        shard_index = hash_redis_key(redis_key) % self._shards_count
        return self._redis_shards[shard_index]

    def get_shards(self):
        return self._redis_shards

    def get_shard_by_index(self, shard_index):
        return self._redis_shards[shard_index]

    def connect(self):
        # TODO: Implement connect here
        pass

    def disconnect(self):
        # TODO: Implement disconnect here
        pass
