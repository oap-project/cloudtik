import logging
import json
import os
from typing import Any, Dict

from cloudtik.core.runtime import Runtime

logger = logging.getLogger(__name__)

# For caching runtime instantiations across API calls of one python session
_runtime_instances = {}

RUNTIME_MINIMAL_EXTERNAL_CONFIG = {}


def _import_ganglia():
    from cloudtik.runtime.ganglia.runtime import GangliaRuntime
    return GangliaRuntime


def _load_ganglia_runtime_home():
    import cloudtik.runtime.ganglia as ganglia
    return os.path.dirname(ganglia.__file__)


def _import_spark():
    from cloudtik.runtime.spark.runtime import SparkRuntime
    return SparkRuntime


def _load_spark_runtime_home():
    import cloudtik.runtime.spark as spark
    return os.path.dirname(spark.__file__)


def _import_hdfs():
    from cloudtik.runtime.hdfs.runtime import HDFSRuntime
    return HDFSRuntime


def _load_hdfs_runtime_home():
    import cloudtik.runtime.hdfs as hdfs
    return os.path.dirname(hdfs.__file__)


_RUNTIMES = {
    "ganglia": _import_ganglia,
    "spark": _import_spark,
    "hdfs": _import_hdfs,
}

_RUNTIME_HOMES = {
    "ganglia": _load_ganglia_runtime_home,
    "spark": _load_spark_runtime_home,
    "hdfs": _load_hdfs_runtime_home,
}


def _get_runtime_cls(runtime_type: str):
    """Get the runtime class for a given runtime config.

    Note that this may be used by private runtimes that proxy methods to
    built-in runtimes, so we should maintain backwards compatibility.

    Returns:
        Runtime class
    """
    importer = _RUNTIMES.get(runtime_type)
    if importer is None:
        raise NotImplementedError("Unsupported runtime: {}".format(
            runtime_type))
    return importer()


def _get_runtime(runtime_type: str, runtime_config: Dict[str, Any],
                 use_cache: bool = True) -> Runtime:
    """Get the instantiated runtime for a given runtime config.

    Note that this may be used by private runtimes that proxy methods to
    built-in runtimes, so we should maintain backwards compatibility.

    Args:
        runtime_type: the runtime type from the runtime config.
        runtime_config: runtime section of the cluster config.
        use_cache: whether to use a cached definition if available. If
            False, the returned object will also not be stored in the cache.

    Returns:
        Runtime
    """
    runtime_section = runtime_config.get(runtime_type, {})
    runtime_key = (runtime_type, json.dumps(runtime_section, sort_keys=True))
    if use_cache and runtime_key in _runtime_instances:
        return _runtime_instances[runtime_key]

    runtime_cls = _get_runtime_cls(runtime_type)
    new_runtime = runtime_cls(runtime_config)

    if use_cache:
        _runtime_instances[runtime_key] = new_runtime

    return new_runtime


def _clear_runtime_cache():
    global _runtime_instances
    _runtime_instances = {}


def _get_runtime_home(runtime_type: str):
    load_config_home = _RUNTIME_HOMES.get(runtime_type)
    if load_config_home is None:
        raise NotImplementedError("Unsupported runtime: {}".format(
            runtime_type))
    path_to_home = load_config_home()
    return path_to_home

