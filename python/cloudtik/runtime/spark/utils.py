import os
from typing import Any, Dict
import yaml

from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_HDFS, BUILT_IN_RUNTIME_METASTORE
from cloudtik.core._private.utils import merge_rooted_config_hierarchy, \
    _get_runtime_config_object, is_runtime_enabled, round_memory_size_to_gb, load_head_cluster_config, \
    RUNTIME_CONFIG_KEY, load_properties_file, save_properties_file
from cloudtik.core._private.workspace.workspace_operator import _get_workspace_provider

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["proc_resourcemanager", False, "ResourceManager", "head"],
    ["proc_nodemanager", False, "NodeManager", "worker"],
]

RUNTIME_ROOT_PATH = os.path.abspath(os.path.dirname(__file__))

YARN_RESOURCE_MEMORY_RATIO = 0.8
SPARK_EXECUTOR_MEMORY_RATIO = 1
SPARK_DRIVER_MEMORY_RATIO = 0.1
SPARK_APP_MASTER_MEMORY_RATIO = 0.02
SPARK_DRIVER_MEMORY_MINIMUM = 1024
SPARK_DRIVER_MEMORY_MAXIMUM = 8192
SPARK_EXECUTOR_CORES_DEFAULT = 4
SPARK_ADDITIONAL_OVERHEAD = 1024
SPARK_EXECUTOR_OVERHEAD_MINIMUM = 384
SPARK_EXECUTOR_OVERHEAD_RATIO = 0.1


def get_spark_driver_memory(cluster_resource: Dict[str, Any]) -> int:
    spark_driver_memory = round_memory_size_to_gb(
        int(cluster_resource["head_memory"] * SPARK_DRIVER_MEMORY_RATIO))
    return max(min(spark_driver_memory,
               SPARK_DRIVER_MEMORY_MAXIMUM), SPARK_DRIVER_MEMORY_MINIMUM)


def get_spark_app_master_memory(worker_memory_for_spark: int) -> int:
    spark_app_master_memory = round_memory_size_to_gb(
        int(worker_memory_for_spark * SPARK_APP_MASTER_MEMORY_RATIO))
    return max(min(spark_app_master_memory,
               SPARK_DRIVER_MEMORY_MAXIMUM), SPARK_DRIVER_MEMORY_MINIMUM)


def get_spark_overhead(worker_memory_for_spark: int) -> int:
    # Calculate the spark overhead including one app master based on worker_memory_for_spark
    spark_app_master = get_spark_app_master_memory(worker_memory_for_spark)
    return spark_app_master + SPARK_ADDITIONAL_OVERHEAD


def get_spark_executor_overhead(spark_executor_memory_all: int) -> int:
    return max(int(spark_executor_memory_all * SPARK_EXECUTOR_OVERHEAD_RATIO),
               SPARK_EXECUTOR_OVERHEAD_MINIMUM)


def _get_cluster_resources(
        cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    """Fills out spark executor resource for available_node_types."""
    cluster_resource = {}
    if "available_node_types" not in cluster_config:
        return cluster_resource

    # Since we have filled the resources for node types
    # We simply don't retrieve it from cloud provider again
    available_node_types = cluster_config["available_node_types"]
    head_node_type = cluster_config["head_node_type"]
    for node_type in available_node_types:
        resources = available_node_types[node_type].get("resources", {})
        memory_total_in_mb = int(resources.get("memory", 0) / (1024 * 1024))
        cpu_total = resources.get("CPU", 0)
        if node_type != head_node_type:
            if memory_total_in_mb > 0:
                cluster_resource["worker_memory"] = memory_total_in_mb
            if cpu_total > 0:
                cluster_resource["worker_cpu"] = cpu_total
        else:
            if memory_total_in_mb > 0:
                cluster_resource["head_memory"] = memory_total_in_mb
            if cpu_total > 0:
                cluster_resource["head_cpu"] = cpu_total

    # If there is only one node type, worker type uses the head type
    if ("worker_memory" not in cluster_resource) and ("head_memory" in cluster_resource):
        cluster_resource["worker_memory"] = cluster_resource["head_memory"]
    if ("worker_cpu" not in cluster_resource) and ("head_cpu" in cluster_resource):
        cluster_resource["worker_cpu"] = cluster_resource["head_cpu"]

    return cluster_resource


def _config_depended_services(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    runtime_config = cluster_config.get(RUNTIME_CONFIG_KEY)
    if "spark" not in runtime_config:
        runtime_config["spark"] = {}
    spark_config = runtime_config["spark"]

    workspace_name = cluster_config.get("workspace_name", "")
    workspace_provider = _get_workspace_provider(cluster_config["provider"], workspace_name)
    global_variables = workspace_provider.subscribe_global_variables(cluster_config)

    # 1) Try to use local hdfs first;
    # 2) Try to use defined hdfs_namenode_uri;
    # 3) If subscribed_hdfs_namenode_uri=true,try to subscribe global variables to find remote hdfs_namenode_uri

    if not is_runtime_enabled(runtime_config, "hdfs"):
        if spark_config.get("hdfs_namenode_uri") is None:
            if spark_config.get("auto_detect_hdfs", False):
                hdfs_namenode_uri = global_variables.get("hdfs-namenode-uri")
                if hdfs_namenode_uri is not None:
                    spark_config["hdfs_namenode_uri"] = hdfs_namenode_uri

    # Check metastore
    if not is_runtime_enabled(runtime_config, "metastore"):
        if spark_config.get("hive_metastore_uri") is None:
            if spark_config.get("auto_detect_metastore", True):
                hive_metastore_uri = global_variables.get("hive-metastore-uri")
                if hive_metastore_uri is not None:
                    spark_config["hive_metastore_uri"] = hive_metastore_uri

    return cluster_config


def _config_runtime_resources(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    cluster_resource = _get_cluster_resources(cluster_config)
    container_resource = {"yarn_container_maximum_vcores": cluster_resource["worker_cpu"]}

    worker_memory_for_yarn = round_memory_size_to_gb(
        int(cluster_resource["worker_memory"] * YARN_RESOURCE_MEMORY_RATIO))
    container_resource["yarn_container_maximum_memory"] = worker_memory_for_yarn

    executor_resource = {"spark_driver_memory": get_spark_driver_memory(cluster_resource)}

    # Calculate Spark executor cores
    spark_executor_cores = SPARK_EXECUTOR_CORES_DEFAULT
    if spark_executor_cores > cluster_resource["worker_cpu"]:
        spark_executor_cores = cluster_resource["worker_cpu"]

    executor_resource["spark_executor_cores"] = spark_executor_cores

    # For Spark executor memory, we use the following formula:
    # x = worker_memory_for_yarn
    # n = number_of_executors
    # m = spark_executor_memory
    # a = spark_overhead (app_master_memory + others)
    # x = n * m + a

    number_of_executors = int(cluster_resource["worker_cpu"] / spark_executor_cores)
    worker_memory_for_spark = round_memory_size_to_gb(
        int(worker_memory_for_yarn * SPARK_EXECUTOR_MEMORY_RATIO))
    spark_overhead = round_memory_size_to_gb(
        get_spark_overhead(worker_memory_for_spark))
    worker_memory_for_executors = worker_memory_for_spark - spark_overhead
    spark_executor_memory_all = round_memory_size_to_gb(
        int(worker_memory_for_executors / number_of_executors))
    executor_resource["spark_executor_memory"] =\
        spark_executor_memory_all - get_spark_executor_overhead(spark_executor_memory_all)

    if RUNTIME_CONFIG_KEY not in cluster_config:
        cluster_config[RUNTIME_CONFIG_KEY] = {}
    runtime_config = cluster_config[RUNTIME_CONFIG_KEY]

    if "spark" not in runtime_config:
        runtime_config["spark"] = {}
    spark_config = runtime_config["spark"]

    spark_config["yarn_container_resource"] = container_resource
    spark_config["spark_executor_resource"] = executor_resource
    return cluster_config


def get_runtime_processes():
    return RUNTIME_PROCESSES


def _is_runtime_scripts(script_file):
    if script_file.endswith(".scala"):
        return True

    return False


def _get_runnable_command(target):
    command_parts = ["spark-shell", "-i", target]
    return command_parts


def _get_spark_config(config: Dict[str, Any]):
    runtime = config.get(RUNTIME_CONFIG_KEY)
    if not runtime:
        return None

    spark = runtime.get("spark")
    if not spark:
        return None

    return spark.get("config")


def update_spark_configurations():
    # Merge user specified configuration and default configuration
    config = load_head_cluster_config()
    spark_config = _get_spark_config(config)
    if not spark_config:
        return

    spark_conf_file = os.path.join(os.getenv("SPARK_HOME"), "conf/spark-defaults.conf")

    # Read in the existing configurations
    spark_conf, comments = load_properties_file(spark_conf_file, ' ')

    # Merge with the user configurations
    spark_conf.update(spark_config)

    # Write back the configuration file
    save_properties_file(spark_conf_file, spark_conf, separator=' ', comments=comments)


def _with_runtime_environment_variables(runtime_config, config, provider, node_id: str):
    runtime_envs = {}
    spark_config = runtime_config.get("spark", {})
    cluster_runtime_config = config.get(RUNTIME_CONFIG_KEY)

    # 1) Try to use local hdfs first;
    # 2) Try to use defined hdfs_namenode_uri;
    # 3) Try to use provider storage;
    if is_runtime_enabled(cluster_runtime_config, BUILT_IN_RUNTIME_HDFS):
        runtime_envs["HDFS_ENABLED"] = True
    elif spark_config.get("hdfs_namenode_uri") is not None:
        runtime_envs["HDFS_NAMENODE_URI"] = spark_config.get("hdfs_namenode_uri")
    else:
        # Whether we need to expert the cloud storage for HDFS case
        provider_envs = provider.with_environment_variables()
        runtime_envs.update(provider_envs)

    # 1) Try to use local metastore if there is one started;
    # 2) Try to use defined metastore_uri;
    if is_runtime_enabled(cluster_runtime_config, BUILT_IN_RUNTIME_METASTORE):
        runtime_envs["METASTORE_ENABLED"] = True
    elif spark_config.get("hive_metastore_uri") is not None:
        runtime_envs["HIVE_METASTORE_URI"] = spark_config.get("hive_metastore_uri")
    return runtime_envs


def get_runtime_logs():
    hadoop_logs_dir = os.path.join(os.getenv("HADOOP_HOME"), "logs")
    spark_logs_dir = os.path.join(os.getenv("SPARK_HOME"), "logs")
    all_logs = {"hadoop": hadoop_logs_dir,
                "spark": spark_logs_dir,
                "other": "/tmp/logs"
                }
    return all_logs


def _validate_config(config: Dict[str, Any], provider):
    # if HDFS enabled, we ignore the cloud storage configurations
    if not is_runtime_enabled(config.get(RUNTIME_CONFIG_KEY), "hdfs"):
        # Check any cloud storage is configured
        provider_config = config["provider"]
        if ("azure_cloud_storage" not in provider_config) and (
                "aws_s3_storage" not in provider_config) and (
                "gcp_cloud_storage" not in provider_config) and \
                not provider_config.get("use_managed_cloud_storage", False):
            raise ValueError("No storage configuration found for Spark.")


def _verify_config(config: Dict[str, Any], provider):
    pass


def _get_config_object(cluster_config: Dict[str, Any], object_name: str) -> Dict[str, Any]:
    config_root = os.path.join(RUNTIME_ROOT_PATH, "config")
    runtime_commands = _get_runtime_config_object(config_root, cluster_config["provider"], object_name)
    return merge_rooted_config_hierarchy(config_root, runtime_commands, object_name)


def _get_runtime_commands(runtime_config: Dict[str, Any],
                          cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    return _get_config_object(cluster_config, "commands")


def _get_defaults_config(runtime_config: Dict[str, Any],
                         cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    return _get_config_object(cluster_config, "defaults")


def _get_useful_urls(cluster_head_ip):
    urls = [
        {"name": "Yarn Web UI", "url": "http://{}:8088".format(cluster_head_ip)},
        {"name": "Jupyter Web UI", "url": "http://{}:8888, default password is \'cloudtik\'".format(cluster_head_ip)},
        {"name": "Spark History Server Web UI", "url": "http://{}:18080".format(cluster_head_ip)},
    ]
    return urls
