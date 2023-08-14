import inspect
import logging
import json
import os
from typing import Any, Dict

from cloudtik.core._private.concurrent_cache import ConcurrentObjectCache
from cloudtik.core._private.core_utils import load_class
from cloudtik.core.runtime import Runtime

logger = logging.getLogger(__name__)

# For caching runtime instantiations across API calls of one python session
_runtime_instances = ConcurrentObjectCache()

RUNTIME_MINIMAL_EXTERNAL_CONFIG = {}

BUILT_IN_RUNTIME_AI = "ai"
BUILT_IN_RUNTIME_SPARK = "spark"
BUILT_IN_RUNTIME_HDFS = "hdfs"
BUILT_IN_RUNTIME_METASTORE = "metastore"
BUILT_IN_RUNTIME_PRESTO = "presto"
BUILT_IN_RUNTIME_TRINO = "trino"
BUILT_IN_RUNTIME_ZOOKEEPER = "zookeeper"
BUILT_IN_RUNTIME_KAFKA = "kafka"
BUILT_IN_RUNTIME_FLINK = "flink"
BUILT_IN_RUNTIME_RAY = "ray"
BUILT_IN_RUNTIME_SSHSERVER = "sshserver"
BUILT_IN_RUNTIME_CONSUL = "consul"
BUILT_IN_RUNTIME_NGINX = "nginx"
BUILT_IN_RUNTIME_HAPROXY = "haproxy"
BUILT_IN_RUNTIME_ETCD = "etcd"
BUILT_IN_RUNTIME_PROMETHEUS = "prometheus"
BUILT_IN_RUNTIME_NODE_EXPORTER = "node_exporter"
BUILT_IN_RUNTIME_GRAFANA = "grafana"
BUILT_IN_RUNTIME_MYSQL = "mysql"
BUILT_IN_RUNTIME_POSTGRES = "postgres"

DEFAULT_RUNTIMES = [BUILT_IN_RUNTIME_PROMETHEUS, BUILT_IN_RUNTIME_NODE_EXPORTER, BUILT_IN_RUNTIME_SPARK]


def _import_ai():
    from cloudtik.runtime.ai.runtime import AIRuntime
    return AIRuntime


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


def _import_flink():
    from cloudtik.runtime.flink.runtime import FlinkRuntime
    return FlinkRuntime


def _import_ray():
    from cloudtik.runtime.ray.runtime import RayRuntime
    return RayRuntime


def _import_ssh_server():
    from cloudtik.runtime.sshserver.runtime import SSHServerRuntime
    return SSHServerRuntime


def _import_consul():
    from cloudtik.runtime.consul.runtime import ConsulRuntime
    return ConsulRuntime


def _import_nginx():
    from cloudtik.runtime.nginx.runtime import NGINXRuntime
    return NGINXRuntime


def _import_haproxy():
    from cloudtik.runtime.haproxy.runtime import HAProxyRuntime
    return HAProxyRuntime


def _import_etcd():
    from cloudtik.runtime.etcd.runtime import EtcdRuntime
    return EtcdRuntime


def _import_prometheus():
    from cloudtik.runtime.prometheus.runtime import PrometheusRuntime
    return PrometheusRuntime


def _import_node_exporter():
    from cloudtik.runtime.node_exporter.runtime import NodeExporterRuntime
    return NodeExporterRuntime


def _import_grafana():
    from cloudtik.runtime.grafana.runtime import GrafanaRuntime
    return GrafanaRuntime


def _import_mysql():
    from cloudtik.runtime.mysql.runtime import MySQLRuntime
    return MySQLRuntime


def _import_postgres():
    from cloudtik.runtime.postgres.runtime import PostgresRuntime
    return PostgresRuntime


_RUNTIMES = {
    BUILT_IN_RUNTIME_AI: _import_ai,
    BUILT_IN_RUNTIME_SPARK: _import_spark,
    BUILT_IN_RUNTIME_HDFS: _import_hdfs,
    BUILT_IN_RUNTIME_METASTORE: _import_metastore,
    BUILT_IN_RUNTIME_PRESTO: _import_presto,
    BUILT_IN_RUNTIME_TRINO: _import_trino,
    BUILT_IN_RUNTIME_ZOOKEEPER: _import_zookeeper,
    BUILT_IN_RUNTIME_KAFKA: _import_kafka,
    BUILT_IN_RUNTIME_FLINK: _import_flink,
    BUILT_IN_RUNTIME_RAY: _import_ray,
    BUILT_IN_RUNTIME_SSHSERVER: _import_ssh_server,
    BUILT_IN_RUNTIME_CONSUL: _import_consul,
    BUILT_IN_RUNTIME_NGINX: _import_nginx,
    BUILT_IN_RUNTIME_HAPROXY: _import_haproxy,
    BUILT_IN_RUNTIME_ETCD: _import_etcd,
    BUILT_IN_RUNTIME_PROMETHEUS: _import_prometheus,
    BUILT_IN_RUNTIME_NODE_EXPORTER: _import_node_exporter,
    BUILT_IN_RUNTIME_GRAFANA: _import_grafana,
    BUILT_IN_RUNTIME_MYSQL: _import_mysql,
    BUILT_IN_RUNTIME_POSTGRES: _import_postgres,
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
            runtime_cls = load_class(runtime_type)
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
