from typing import Any, Dict

from cloudtik.core._private.constants import CLOUDTIK_DEFAULT_STORAGE_URI, CLOUDTIK_DEFAULT_CLOUD_STORAGE_URI
from cloudtik.core._private.providers import _get_node_provider
from cloudtik.core._private.runtime_factory import _get_runtime, BUILT_IN_RUNTIME_HDFS
from cloudtik.core._private.utils import RUNTIME_CONFIG_KEY, get_cluster_head_ip, is_runtime_enabled


def get_runtime_endpoints_of(config: Dict[str, Any], runtime_name: str):
    runtime_config = config.get(RUNTIME_CONFIG_KEY)

    # Verify runtime is in configured
    if not is_runtime_enabled(runtime_config, runtime_name):
        raise RuntimeError("Runtime {} is not enabled.".format(runtime_name))

    # Get the cluster head ip
    head_internal_ip = get_cluster_head_ip(config)

    runtime = _get_runtime(runtime_name, runtime_config)
    return runtime.get_runtime_endpoints(head_internal_ip)


def get_runtime_default_storage_of(config: Dict[str, Any], runtime_name: str):
    runtime_config = config.get(RUNTIME_CONFIG_KEY)
    config_of_runtime = runtime_config.get(runtime_name, {})

    # 1) Try to use local hdfs first;
    # 2) Try to use defined hdfs_namenode_uri;
    # 3) Try to use cloud storage;
    if is_runtime_enabled(runtime_config, BUILT_IN_RUNTIME_HDFS):
        # Use local HDFS, for this to work, cluster must be running
        head_internal_ip = get_cluster_head_ip(config)
        default_storage = {CLOUDTIK_DEFAULT_STORAGE_URI: "hdfs://{}:9000".format(head_internal_ip)}
        return default_storage
    else:
        hdfs_namenode_uri = config_of_runtime.get("hdfs_namenode_uri")
        if hdfs_namenode_uri:
            default_storage = {CLOUDTIK_DEFAULT_STORAGE_URI: hdfs_namenode_uri}
            return default_storage

        # cloud storage
        provider = _get_node_provider(config["provider"], config["cluster_name"])
        default_cloud_storage = provider.get_default_cloud_storage()
        if default_cloud_storage:
            default_storage = {}
            default_storage.update(default_cloud_storage)
            if CLOUDTIK_DEFAULT_CLOUD_STORAGE_URI in default_cloud_storage:
                default_storage[CLOUDTIK_DEFAULT_STORAGE_URI] =\
                    default_cloud_storage[CLOUDTIK_DEFAULT_CLOUD_STORAGE_URI]
            return default_storage

        return None
