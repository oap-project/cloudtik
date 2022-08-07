import logging
from typing import List, Optional

from cloudtik.core._private.state.redis_shards_client import RedisShardsClient
from cloudtik.core._private.state.state_table_store import StateTableStore


logger = logging.getLogger(__name__)

# b'@:' will be the leading characters for namespace
# If the key in storage has this, it'll contain namespace
__NS_START_CHAR = b"@namespace_"


def _make_key(namespace: Optional[str], key: bytes) -> bytes:
    if namespace is None:
        if key.startswith(__NS_START_CHAR):
            raise ValueError("key is not allowed to start with"
                             f" '{__NS_START_CHAR}'")
        return key
    assert isinstance(namespace, str)
    assert isinstance(key, bytes)
    return b":".join([__NS_START_CHAR + namespace.encode(), key])


def _get_key(key: bytes) -> bytes:
    assert isinstance(key, bytes)
    if not key.startswith(__NS_START_CHAR):
        return key
    _, key = key.split(b":", 1)
    return key


class StateClient:
    """
    Client to Redis state store without sharding
    Will use the primary redis instance to manage the keys
    """

    def __init__(self,
                 redis_client,
                 nums_reconnect_retry: int = 5):
        self._redis_client = redis_client
        self._nums_reconnect_retry = nums_reconnect_retry

    def kv_get(self, key: bytes, namespace: Optional[str]) -> bytes:
        logger.debug(f"internal_kv_get {key} {namespace}")
        key = _make_key(namespace, key)
        try:
            return self._redis_client.get(key)
        except Exception:
            raise RuntimeError(f"Failed to get value for key {key}")

    def kv_put(self, key: bytes, value: bytes, overwrite: bool,
                        namespace: Optional[str]) -> int:
        logger.debug(f"internal_kv_put {key} {value} {overwrite} {namespace}")
        key = _make_key(namespace, key)
        try:
            if not overwrite:
                result = self._redis_client.set(key, value, nx=True)
                if result is None:
                    return 1
                return 0
            else:
                self._redis_client.set(key, value)
                # TODO: do we know whether the key exists and overwritten?
                return 0
        except Exception:
            raise RuntimeError(f"Failed to put value {value} to key {key}")

    def kv_del(self, key: bytes, namespace: Optional[str]) -> int:
        logger.debug(f"internal_kv_del {key} {namespace}")
        key = _make_key(namespace, key)
        try:
            return self._redis_client.delete(key)
        except Exception:
            raise RuntimeError(f"Failed to delete key {key}")

    def kv_exists(self, key: bytes, namespace: Optional[str]) -> bool:
        logger.debug(f"internal_kv_exists {key} {namespace}")
        key = _make_key(namespace, key)
        try:
            count = self._redis_client.exists(key)
            if count > 0:
                return True
            return False
        except Exception:
            raise RuntimeError(f"Failed to check existence of key {key}")

    def kv_keys(self, prefix: bytes,
                         namespace: Optional[str]) -> List[bytes]:
        logger.debug(f"internal_kv_keys {prefix} {namespace}")
        prefix = _make_key(namespace, prefix)
        try:
            return [_get_key(key) for key in self._redis_client.scan_iter(prefix + "*")]
        except Exception:
            raise RuntimeError(f"Failed to list prefix {prefix}")

    @staticmethod
    def create_from_redis(redis_cli):
        return StateClient(redis_client=redis_cli)


class ControlStateAccessor:
    """A class used to access control state from redis.
    """
    def __init__(self, redis_address, redis_port, redis_password):
        """Create an Access object."""
        # Args used for lazy init of this object.
        self.redis_address = redis_address
        self.redis_port = redis_port
        self.redis_password = redis_password
        self.redis_shard_client = None
        self.state_table_store = None
        self.connected = False

    def connect(self):
        self.redis_shard_client = RedisShardsClient(
            redis_address=self.redis_address,
            redis_port=self.redis_port,
            redis_password=self.redis_password)
        self.redis_shard_client.connect()
        self.state_table_store = StateTableStore(self.redis_shard_client)
        self.connected = True

    def disconnect(self):
        # Don't need to do anything for Redis
        # Because it uses connection pool underlayer
        self.redis_shard_client.disconnect()
        self.redis_shard_client = None
        self.state_table_store = None
        self.connected = False

    def get_node_table(self):
        assert self.connected, "Control state accessor not connected"
        return self.state_table_store.get_node_table()

    def get_user_state_table(self, table_name):
        assert self.connected, "Control state accessor not connected"
        return self.state_table_store.get_user_state_table(table_name)


class ControlState:
    """A class used to interface with the global control state.

    Attributes:
        control_state_accessor: The client used to query global state table from redis
            server.
    """

    def __init__(self):
        """Create a GlobalState object."""
        # Args used for lazy init of this object.
        self.redis_address = None
        self.redis_port = None
        self.redis_password = None
        self.control_state_accessor = None

    def _check_connected(self):
        """Ensure that the object has been initialized before it is used.

        This lazily initializes clients needed for state accessors.

        Raises:
            RuntimeError: An exception is raised if error.
        """
        if (self.redis_address is not None
                and self.control_state_accessor is None):
            self._really_init_global_state()

        # _really_init_global_state should have set self.global_state_accessor
        if self.control_state_accessor is None:
            raise RuntimeError(
                "Failed to gain access to system control state from redis.")

    def disconnect(self):
        """Disconnect control state accessor."""
        self.redis_address = None
        self.redis_port = None
        self.redis_password = None
        if self.control_state_accessor is not None:
            self.control_state_accessor.disconnect()
            self.control_state_accessor = None

    def initialize_control_state(self, redis_address, redis_port, redis_password):
        """Set args for lazily initialization of the ControlState object.

        It's possible that certain keys in gcs kv may not have been fully
        populated yet. In this case, we will retry this method until they have
        been populated or we exceed a timeout.

        Args:
            redis_address (str): The redis address
            redis_port (int): The redis server port
            redis_password (str): The redis password if needed
        """

        # Save args for lazy init of global state. This avoids opening extra
        # gcs connections from each worker until needed.
        self.redis_address = redis_address
        self.redis_port = redis_port
        self.redis_password = redis_password

    def _really_init_global_state(self):
        self.control_state_accessor = ControlStateAccessor(self.redis_address,
                                                           self.redis_port,
                                                           self.redis_password)
        self.control_state_accessor.connect()

    def get_node_table(self):
        self._check_connected()
        node_table = self.control_state_accessor.get_node_table()
        return node_table

    def get_user_state_table(self, table_name):
        self._check_connected()
        state_table = self.control_state_accessor.get_user_state_table(table_name)
        return state_table
