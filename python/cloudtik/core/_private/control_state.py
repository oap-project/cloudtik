import logging
from typing import List, Optional

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
    """Client to Redis state store with sharding"""

    def __init__(self,
                 redis_client,
                 nums_reconnect_retry: int = 5):
        self._redis_client = redis_client
        self._nums_reconnect_retry = nums_reconnect_retry

    def kv_get(self, key: bytes, namespace: Optional[str]) -> bytes:
        logger.debug(f"internal_kv_get {key} {namespace}")
        key = _make_key(namespace, key)
        # TODO (haifeng): Add implementation
        if True:
            return b""
        else:
            raise RuntimeError(f"Failed to get value for key {key} "
                               f"due to error {reply.status.message}")

    def kv_put(self, key: bytes, value: bytes, overwrite: bool,
                        namespace: Optional[str]) -> int:
        logger.debug(f"internal_kv_put {key} {value} {overwrite} {namespace}")
        key = _make_key(namespace, key)
        # TODO (haifeng): Add implementation
        if True:
            return 0
        else:
            raise RuntimeError(f"Failed to put value {value} to key {key} "
                               f"due to error {reply.status.message}")

    def kv_del(self, key: bytes, namespace: Optional[str]) -> int:
        logger.debug(f"internal_kv_del {key} {namespace}")
        key = _make_key(namespace, key)
        # TODO (haifeng): Add implementation
        if True:
            return 0
        else:
            raise RuntimeError(f"Failed to delete key {key} "
                               f"due to error {reply.status.message}")

    def kv_exists(self, key: bytes, namespace: Optional[str]) -> bool:
        logger.debug(f"internal_kv_exists {key} {namespace}")
        key = _make_key(namespace, key)
        # TODO (haifeng): Add implementation
        if True:
            return False
        else:
            raise RuntimeError(f"Failed to check existence of key {key} "
                               f"due to error {reply.status.message}")


    def kv_keys(self, prefix: bytes,
                         namespace: Optional[str]) -> List[bytes]:
        logger.debug(f"internal_kv_keys {prefix} {namespace}")
        prefix = _make_key(namespace, prefix)
        # TODO (haifeng): Add implementation
        if True:
            return [_get_key(key) for key in []]
        else:
            raise RuntimeError(f"Failed to list prefix {prefix} "
                               f"due to error {reply.status.message}")

    @staticmethod
    def create_from_redis(redis_cli):
        return StateClient(redis_client=redis_cli)


class ControlStateAccessor:
    """A class used to access control state from redis.
    """
    def __init__(self, redis_address, redis_password):
        """Create an Access object."""
        # Args used for lazy init of this object.
        self.redis_address = redis_address
        self.redis_password = redis_password
        self.connected = False

    def connect(self):
        # TODO (haifeng): connect to the redis
        self.connected = True
        pass

    def disconnect(self):
        # TODO (haifeng): disconnect from the redis
        self.connected = False
        pass

    def get_node_table(self):
        assert self.connected, "Control state accessor not connected"
        # TODO (haifeng): read the node table from the redis
        return []


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
        self.redis_password = None
        if self.control_state_accessor is not None:
            self.control_state_accessor.disconnect()
            self.control_state_accessor = None

    def initialize_control_state(self, redis_address, redis_password):
        """Set args for lazily initialization of the ControlState object.

        It's possible that certain keys in gcs kv may not have been fully
        populated yet. In this case, we will retry this method until they have
        been populated or we exceed a timeout.

        Args:
            redis_address (str): The redis address
            redis_password (str): The redis password if needed
        """

        # Save args for lazy init of global state. This avoids opening extra
        # gcs connections from each worker until needed.
        self.redis_address = redis_address
        self.redis_password = redis_password

    def _really_init_global_state(self):
        self.control_state_accessor = ControlStateAccessor(self.redis_address, self.redis_password)
        self.control_state_accessor.connect()

    def node_table(self):
        """Fetch and parse the node info table.

        Returns:
            Information about the node in the cluster.
        """
        self._check_connected()

        node_table = self.global_state_accessor.get_node_table()

        results = []
        for node_info_item in node_table:
            node_info = {}
            # TODO (haifeng): parse and populate node info from the node info string
            node_info["alive"] = node_info["Alive"]
            results.append(node_info)
        return results


class ResourceUsageBatch:
    def __init__(self):
        self.batch = []
        self.resource_demands = []


class ResourceInfoClient:
    """Client to read resource information from Redis"""

    def __init__(self,
                 state_client,
                 nums_reconnect_retry: int = 5):
        self._state_client = state_client
        self._nums_reconnect_retry = nums_reconnect_retry

    def get_cluster_resource_usage(self, timeout: int = 60):
        resources_usage_batch = ResourceUsageBatch
        # TODO (haifeng): implement the resource usage metrics of cluster
        return resources_usage_batch

    @staticmethod
    def create_from_state_client(state_cli):
        return ResourceInfoClient(state_client=state_cli)