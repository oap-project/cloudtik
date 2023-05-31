import inspect
import logging
import json
import os
from typing import Any, Dict

from cloudtik.core._private.concurrent_cache import ConcurrentObjectCache
from cloudtik.core._private.core_utils import _load_class
from cloudtik.core.runtime import Runtime

logger = logging.getLogger(__name__)

# For caching runtime instantiations across API calls of one python session
_runtime_instances = ConcurrentObjectCache()

RUNTIME_MINIMAL_EXTERNAL_CONFIG = {}

BUILT_IN_RUNTIME_GANGLIA = "ganglia"
BUILT_IN_RUNTIME_HDFS = "hdfs"
BUILT_IN_RUNTIME_METASTORE = "metastore"
BUILT_IN_RUNTIME_SPARK = "spark"
BUILT_IN_RUNTIME_PRESTO = "presto"
BUILT_IN_RUNTIME_TRINO = "trino"
BUILT_IN_RUNTIME_ZOOKEEPER = "zookeeper"
BUILT_IN_RUNTIME_KAFKA = "kafka"
BUILT_IN_RUNTIME_AI = "ai"
BUILT_IN_RUNTIME_FLINK = "flink"
BUILT_IN_RUNTIME_RAY = "ray"

DEFAULT_RUNTIMES = [BUILT_IN_RUNTIME_GANGLIA, BUILT_IN_RUNTIME_SPARK]


def _import_ganglia():
    from cloudtik.runtime.ganglia.runtime import GangliaRuntime
    return GangliaRuntime


def _import_spark():
    from cloudtik.runtime.spark.runtime import SparkRuntime
    return SparkRuntime


def _import_hdfs():
    from cloudtik.runtime.hdfs.runtime import HDFSRuntime
    return HDFSRuntime


def _import_metastore():
    from cloudtik.runtime.metastore.runtime import MetastoreRuntime
    return MetastoreRuntime


def _import_presto():
    from cloudtik.runtime.presto.runtime import PrestoRuntime
    return PrestoRuntime


def _import_trino():
    from cloudtik.runtime.trino.runtime import TrinoRuntime
    return TrinoRuntime


def _import_zookeeper():
    from cloudtik.runtime.zookeeper.runtime import ZooKeeperRuntime
    return ZooKeeperRuntime


def _import_kafka():
    from cloudtik.runtime.kafka.runtime import KafkaRuntime
    return KafkaRuntime


def _import_ai():
    from cloudtik.runtime.ai.runtime import AIRuntime
    return AIRuntime


def _import_flink():
    from cloudtik.runtime.flink.runtime import FlinkRuntime
    return FlinkRuntime


def _import_ray():
    from cloudtik.runtime.ray.runtime import RayRuntime
    return RayRuntime


_RUNTIMES = {
    "ganglia": _import_ganglia,
    "spark": _import_spark,
    "hdfs": _import_hdfs,
    "metastore": _import_metastore,
    "presto": _import_presto,
    "trino": _import_trino,
    "zookeeper": _import_zookeeper,
    "kafka": _import_kafka,
    "ai": _import_ai,
    "flink": _import_flink,
    "ray": _import_ray,
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
        # This is not a built-in runtime, it may be an external
        # try load external runtime: runtime_type is the full class name with package
        try:
            runtime_cls = _load_class(runtime_type)
            return runtime_cls
        except (ModuleNotFoundError, ImportError) as e:
            raise NotImplementedError("Unsupported runtime: {}.".format(
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
    def load_runtime(runtime_type: str, runtime_config: Dict[str, Any]):
        runtime_cls = _get_runtime_cls(runtime_type)
        return runtime_cls(runtime_config)

    if not use_cache:
        return load_runtime(runtime_type, runtime_config)

    runtime_section = runtime_config.get(runtime_type, {})
    runtime_key = (runtime_type, json.dumps(runtime_section, sort_keys=True))
    return _runtime_instances.get(
        runtime_key, load_runtime,
        runtime_type=runtime_type, runtime_config=runtime_config)


def _clear_runtime_cache():
    _runtime_instances.clear()


def _get_runtime_home(runtime_type: str):
    runtime_cls = _get_runtime_cls(runtime_type)
    runtime_module = inspect.getmodule(runtime_cls)
    return os.path.dirname(runtime_module.__file__)
