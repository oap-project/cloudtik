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


def get_real_key(redis_key, table_name):
    # get the real key from shard key
    pos = len(table_name) + len(TABLE_SEPERATOR)
    return redis_key[pos:]


# TODO: Implement operations on a specific redis shard
class RedisShard:
    def __init__(self):
        self._redis_client = None

    def connect(self, redis_address, redis_port, redis_password):
        self._redis_client = redis.StrictRedis(
            host=redis_address, port=redis_port, password=redis_password)

    def put(self, key, value):
        self._redis_client.set(key, value)

    def get(self, key):
        return self._redis_client.get(key)

    def delete(self, key):
        return self._redis_client.delete(key)

    def mget(self, keys):
        return self._redis_client.mget(keys)

    def scan(self, cursor, match_pattern, batch_size):
        return self._redis_client.scan(cursor, match_pattern, batch_size)

    def lrange(self, key, start=0, stop=-1):
        return self._redis_client.lrange(key, start, stop)


def get_redis_shards_addresses(primary_shard: RedisShard):
    redis_addresses = []
    redis_ports = []

    str_shards = primary_shard.get("NumRedisShards")
    if str_shards is None:
        return redis_addresses, redis_ports

    num_shards = int(str_shards.decode())
    if num_shards <= 0:
        return redis_addresses, redis_ports

    elements = primary_shard.lrange("RedisShards")
    if elements is None or not elements:
        return redis_addresses, redis_ports

    # Parse the redis shard address
    for element in elements:
        redis_address, redis_port = element.split(":")
        redis_addresses.append(redis_address)
        redis_ports.append(redis_port)

    return redis_addresses, redis_ports


class RedisShardsClient:
    def __init__(self, redis_address, redis_port, redis_password):
        self._redis_address = redis_address
        self._redis_port = redis_port
        self._redis_password = redis_password
        self._is_connected = False
        self._primary_shard = None
        self._redis_shards = None
        self._shards_count = 0

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
        self._primary_shard = RedisShard()
        self._primary_shard.connect(self._redis_address,
                                    self._redis_port,
                                    self._redis_password)

        # get redis shards and connect redis shards
        shard_addresses, shard_ports = get_redis_shards_addresses(
            self._primary_shard)
        if not shard_addresses:
            shard_addresses.append(self._redis_address)
            shard_ports.append(self._redis_port)

        for shard_address, shard_port in zip(shard_addresses, shard_ports):
            redis_shard = RedisShard
            redis_shard.connect(shard_address, shard_port, self._redis_password)
            self._redis_shards.append(redis_shard)

        self._shards_count = len(self._redis_shards)
        self._is_connected = True

    def disconnect(self):
        self._primary_shard = None
        if self._redis_shards is not None:
            self._redis_shards.clear()
            self._redis_shards = None
        self._shards_count = 0
        self._is_connected = False
