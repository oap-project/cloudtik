import threading


class ConcurrentObjectCache:
    """An object cache which is thread safe.
    """
    def __init__(self):
        self._lock = threading.RLock()
        self._cache = {}

    def get(self, key, load_function, **load_args):
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            value = load_function(**load_args)
            self._cache[key] = value
            return value

    def clear(self):
        with self._lock:
            self._cache = {}
