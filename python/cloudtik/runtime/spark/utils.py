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

CLOUDTIK_SPARK_RUNTIME_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
SPARK_OUT_CONF = os.path.join(CLOUDTIK_SPARK_RUNTIME_PATH, "spark/conf/outconf/spark/spark-defaults.conf")
CLOUDTIK_BOOTSTRAP_CONFIG_PATH = "~/cloudtik_bootstrap_config.yaml"

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


def config_spark_runtime_resources(
        cluster_config: Dict[str, Any], cluster_resource: Dict[str, Any]) -> Dict[str, Any]:
    cluster_config = copy.deepcopy(cluster_config)

    container_resource = {}
    container_resource["yarn_container_maximum_vcores"] = cluster_resource["worker_cpu"]

    worker_memory_for_yarn = round_memory_size_to_gb(
        int(cluster_resource["worker_memory"] * YARN_RESOURCE_MEMORY_RATIO))
    container_resource["yarn_container_maximum_memory"] = worker_memory_for_yarn

    executor_resource = {}
    executor_resource["spark_driver_memory"] = get_spark_driver_memory(cluster_resource)

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
    bootstrap_config = os.path.expanduser(CLOUDTIK_BOOTSTRAP_CONFIG_PATH)
    if not os.path.exists(bootstrap_config):
        return

    config = yaml.safe_load(open(bootstrap_config).read())
    spark_config = _get_spark_config(config)
    if not spark_config:
        return

    spark_conf = {}
    with open(SPARK_OUT_CONF, "r") as f:
        for line in f.readlines():
            if not line.startswith("#"):
                key, value = line.split(" ")
                spark_conf[key] = value
    spark_conf.update(spark_config)
    with open(os.path.join(os.getenv("SPARK_HOME"), "conf/spark-defaults.conf"), "w+") as f:
        for key, value in spark_conf.items():
            f.write("{}    {}\n".format(key, value))


def is_cloud_storage_mount_enabled() -> bool:
    bootstrap_config = os.path.expanduser(CLOUDTIK_BOOTSTRAP_CONFIG_PATH)

    if not os.path.exists(bootstrap_config):
        return False

    config = yaml.safe_load(open(bootstrap_config).read())
    return config.get("cloud_storage_mount_enabled", False)
