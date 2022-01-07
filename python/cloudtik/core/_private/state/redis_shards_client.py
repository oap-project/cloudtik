import logging
import redis

logger = logging.getLogger(__name__)


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
        shard_index = self._hash_redis_key(redis_key) % self._shards_count
        return self._redis_shards[shard_index]

    def connect(self):
        # TODO: Implement connect here
        pass

    def disconnect(self):
        # TODO: Implement disconnect here
        pass

    def _hash_redis_key(self, redis_key) -> int:
        # TODO: Implement hash from the key here
        return 0