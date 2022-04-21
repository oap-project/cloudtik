import os
import copy
from typing import Any, Dict
import yaml

SPARK_RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["proc_resourcemanager", False, "ResourceManager", "head"],
    ["proc_nodemanager", False, "NodeManager", "worker"],
]

CLOUDTIK_RUNTIME_SPARK_PATH = os.path.abspath(os.path.dirname(__file__))

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


def round_memory_size_to_gb(memory_size: int) -> int:
    gb = int(memory_size / 1024)
    if gb < 1:
        gb = 1
    return gb * 1024


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
    return cluster_resource


def config_spark_runtime_resources(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
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

    if "runtime" not in cluster_config:
        cluster_config["runtime"] = {}
    runtime_config = cluster_config["runtime"]

    if "spark" not in runtime_config:
        runtime_config["spark"] = {}
    spark_config = runtime_config["spark"]

    spark_config["yarn_container_resource"] = container_resource
    spark_config["spark_executor_resource"] = executor_resource
    return cluster_config


def get_runtime_processes():
    return SPARK_RUNTIME_PROCESSES


def is_spark_runtime_scripts(script_file):
    if script_file.endswith(".scala"):
        return True

    return False


def get_spark_runtime_command(target):
    command_parts = ["spark-shell", "-i", target]
    return command_parts


def _get_spark_config(config: Dict[str, Any]):
    runtime = config.get("runtime")
    if not runtime:
        return None

    spark = runtime.get("spark")
    if not spark:
        return None

    return spark.get("config")


def update_spark_configurations():
    # Merge user specified configuration and default configuration
    bootstrap_config = os.path.expanduser("~/cloudtik_bootstrap_config.yaml")
    if not os.path.exists(bootstrap_config):
        return

    config = yaml.safe_load(open(bootstrap_config).read())
    spark_config = _get_spark_config(config)
    if not spark_config:
        return

    spark_conf_file = os.path.join(os.getenv("SPARK_HOME"), "conf/spark-defaults.conf")

    # Read in the existing configurations
    spark_conf = {}
    with open(spark_conf_file, "r") as f:
        for line in f.readlines():
            # Strip all the spaces and tabs
            line = line.strip()
            if line != "" and not line.startswith("#"):
                # Filtering out the empty and comment lines
                # Use split() instead of split(" ") to split value with multiple spaces
                key, value = line.split()
                spark_conf[key] = value

    # Merge with the user configurations
    spark_conf.update(spark_config)

    # Write back the configuration file
    with open(spark_conf_file, "w+") as f:
        for key, value in spark_conf.items():
            f.write("{}    {}\n".format(key, value))


def with_spark_runtime_environment_variables(runtime_config, provider):
    runtime_envs = {}
    if runtime_config and runtime_config.get("spark", {}).get("enable_hdfs", False):
        runtime_envs["ENABLE_HDFS"] = True
    else:
        # Whether we need to expert the cloud storage for HDFS case
        provider_envs = provider.with_provider_environment_variables()
        runtime_envs.update(provider_envs)

    return runtime_envs


def get_spark_runtime_logs():
    hadoop_logs_dir = os.path.join(os.getenv("HADOOP_HOME"), "logs")
    spark_logs_dir = os.path.join(os.getenv("SPARK_HOME"), "logs")
    all_logs = [("hadoop", hadoop_logs_dir),
                ("spark", spark_logs_dir),
                ("other", "/tmp/logs")]
    return all_logs


def spark_runtime_validate_config(config: Dict[str, Any], provider):
    # if HDFS enabled, we ignore the cloud storage configurations
    if not config.get("runtime", {}).get("spark", {}).get("enable_hdfs", False):
        provider.validate_storage_config(config["provider"])
