from typing import List, Union, Optional

from cloudtik.core._private.state.control_state import StateClient

_initialized = False
global_state_client = None


def kv_reset():
    global global_state_client, _initialized
    global_state_client = None
    _initialized = False


def kv_get_state_client():
    return global_state_client


def kv_initialize(state_client: StateClient):
    """Initialize the internal KV for use in other function calls.
    """
    global global_state_client, _initialized
    assert state_client is not None
    global_state_client = state_client
    _initialized = True


def kv_initialized():
    return global_state_client is not None


def kv_get(key: Union[str, bytes],
                     *,
                     namespace: Optional[str] = None) -> bytes:
    """Fetch the value of a binary key."""

    if isinstance(key, str):
        key = key.encode()
    assert isinstance(key, bytes)
    return global_state_client.kv_get(key, namespace)


def kv_exists(key: Union[str, bytes],
                        *,
                        namespace: Optional[str] = None) -> bool:
    """Check key exists or not."""

    if isinstance(key, str):
        key = key.encode()
    assert isinstance(key, bytes)
    return global_state_client.kv_exists(key, namespace)


def kv_put(key: Union[str, bytes],
                     value: Union[str, bytes],
                     overwrite: bool = True,
                     *,
                     namespace: Optional[str] = None) -> bool:
    """Globally associates a value with a given binary key.

    This only has an effect if the key does not already have a value.

    Returns:
        already_exists (bool): whether the value already exists.
    """

    if isinstance(key, str):
        key = key.encode()
    if isinstance(value, str):
        value = value.encode()
    assert isinstance(key, bytes) and isinstance(value, bytes) and isinstance(
        overwrite, bool)
    return global_state_client.kv_put(key, value, overwrite,
                                             namespace) == 0


def kv_del(key: Union[str, bytes],
                     *,
                     namespace: Optional[str] = None):
    if isinstance(key, str):
        key = key.encode()
    assert isinstance(key, bytes)
    return global_state_client.kv_del(key, namespace)


def kv_list(prefix: Union[str, bytes],
                      *,
                      namespace: Optional[str] = None) -> List[bytes]:
    """List all keys in the internal KV store that start with the prefix.
    """
    if isinstance(prefix, str):
        prefix = prefix.encode()
    return global_state_client.kv_keys(prefix, namespace)
