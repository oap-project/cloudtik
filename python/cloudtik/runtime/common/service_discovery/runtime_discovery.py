from typing import Dict, Any

from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_HDFS, BUILT_IN_RUNTIME_METASTORE, \
    BUILT_IN_RUNTIME_CONSUL, BUILT_IN_RUNTIME_ZOOKEEPER
from cloudtik.core._private.service_discovery.utils import get_service_selector_for_update, select_runtime
from cloudtik.runtime.common.service_discovery.discovery import query_one_service


def discover_runtime_service(
        config: Dict[str, Any],
        service_selector_key: str,
        runtime_type: str,
        cluster_config: Dict[str, Any],
        discovery_type,):
    service_selector = get_service_selector_for_update(
        config, service_selector_key)
    service_selector = select_runtime(
        service_selector, runtime_type)
    service_addresses = query_one_service(
        cluster_config, service_selector,
        discovery_type=discovery_type)
    return service_addresses


def discover_consul(
        config: Dict[str, Any],
        service_selector_key: str,
        cluster_config: Dict[str, Any],
        discovery_type):
    return discover_runtime_service(
        config, service_selector_key,
        runtime_type=BUILT_IN_RUNTIME_CONSUL,
        cluster_config=cluster_config,
        discovery_type=discovery_type,
    )


def discover_zookeeper(
        config: Dict[str, Any],
        service_selector_key: str,
        cluster_config: Dict[str, Any],
        discovery_type):
    return discover_runtime_service(
        config, service_selector_key,
        runtime_type=BUILT_IN_RUNTIME_ZOOKEEPER,
        cluster_config=cluster_config,
        discovery_type=discovery_type,
    )


def discover_hdfs(
        config: Dict[str, Any],
        service_selector_key: str,
        cluster_config: Dict[str, Any],
        discovery_type):
    service_addresses = discover_runtime_service(
        config, service_selector_key,
        runtime_type=BUILT_IN_RUNTIME_HDFS,
        cluster_config=cluster_config,
        discovery_type=discovery_type,
    )
    if not service_addresses:
        return None
    service_address = service_addresses[0]
    hdfs_uri = "hdfs://{}:{}".format(
        service_address[0], service_address[1])
    return hdfs_uri


def discover_metastore(
        config: Dict[str, Any],
        service_selector_key: str,
        cluster_config: Dict[str, Any],
        discovery_type):
    service_addresses = discover_runtime_service(
        config, service_selector_key,
        runtime_type=BUILT_IN_RUNTIME_METASTORE,
        cluster_config=cluster_config,
        discovery_type=discovery_type,
    )
    if not service_addresses:
        return None
    # take one of them
    service_address = service_addresses[0]
    hive_metastore_uri = "thrift://{}:{}".format(
        service_address[0], service_address[1])
    return hive_metastore_uri
