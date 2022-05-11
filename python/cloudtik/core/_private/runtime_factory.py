import logging
import json
import os
from typing import Any, Dict

from cloudtik.core._private.concurrent_cache import ConcurrentObjectCache
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
BUILT_IN_RUNTIME_ZOOKEEPER = "zookeeper"
BUILT_IN_RUNTIME_KAFKA = "kafka"


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


def _import_metastore():
    from cloudtik.runtime.metastore.runtime import MetastoreRuntime
    return MetastoreRuntime


def _load_metastore_runtime_home():
    import cloudtik.runtime.metastore as metastore
    return os.path.dirname(metastore.__file__)


def _import_presto():
    from cloudtik.runtime.presto.runtime import PrestoRuntime
    return PrestoRuntime


def _load_presto_runtime_home():
    import cloudtik.runtime.presto as presto
    return os.path.dirname(presto.__file__)


def _import_zookeeper():
    from cloudtik.runtime.zookeeper.runtime import ZooKeeperRuntime
    return ZooKeeperRuntime


def _load_zookeeper_runtime_home():
    import cloudtik.runtime.zookeeper as zookeeper
    return os.path.dirname(zookeeper.__file__)


def _import_kafka():
    from cloudtik.runtime.kafka.runtime import KafkaRuntime
    return KafkaRuntime


def _load_kafka_runtime_home():
    import cloudtik.runtime.kafka as kafka
    return os.path.dirname(kafka.__file__)


_RUNTIMES = {
    "ganglia": _import_ganglia,
    "spark": _import_spark,
    "hdfs": _import_hdfs,
    "metastore": _import_metastore,
    "presto": _import_presto,
    "zookeeper": _import_zookeeper,
    "kafka": _import_kafka,
}

_RUNTIME_HOMES = {
    "ganglia": _load_ganglia_runtime_home,
    "spark": _load_spark_runtime_home,
    "hdfs": _load_hdfs_runtime_home,
    "metastore": _load_metastore_runtime_home,
    "presto": _load_presto_runtime_home,
    "zookeeper": _load_zookeeper_runtime_home,
    "kafka": _load_kafka_runtime_home,
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
    load_config_home = _RUNTIME_HOMES.get(runtime_type)
    if load_config_home is None:
        raise NotImplementedError("Unsupported runtime: {}".format(
            runtime_type))
    path_to_home = load_config_home()
    return path_to_home

