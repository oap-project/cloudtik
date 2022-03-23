import copy
from typing import Any, Dict


SPARK_RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["proc_resourcemanager", False, "ResourceManager", "head"],
    ["proc_nodemanager", False, "NodeManager", "worker"],
]

YARN_RESOURCE_MEMORY_RATIO = 0.8
SPARK_EXECUTOR_MEMORY_RATIO = 0.8
SPARK_DRIVER_MEMORY_RATIO = 0.25


def config_spark_runtime_resources(
        cluster_config: Dict[str, Any], cluster_resource: Dict[str, Any]) -> Dict[str, Any]:
    cluster_config = copy.deepcopy(cluster_config)

    container_resource = {}
    container_resource["yarn_container_maximum_vcores"] = cluster_resource["worker_cpu"]
    container_resource["yarn_container_maximum_memory"] = \
        int(cluster_resource["worker_memory"] * YARN_RESOURCE_MEMORY_RATIO)

    executor_resource = {}
    if int(cluster_resource["worker_cpu"]) < 4:
        executor_resource["spark_executor_cores"] = cluster_resource["worker_cpu"]
        executor_resource["spark_executor_memory"] = int(cluster_resource["worker_memory"] *
                                                         YARN_RESOURCE_MEMORY_RATIO * SPARK_EXECUTOR_MEMORY_RATIO)
    else:
        worker_per_core_memory_MB = cluster_resource["worker_memory"] /  cluster_resource["worker_cpu"]
        executor_resource["spark_executor_cores"] = 4
        executor_resource["spark_executor_memory"] = int(worker_per_core_memory_MB * 4 *
                                                         YARN_RESOURCE_MEMORY_RATIO * SPARK_EXECUTOR_MEMORY_RATIO)
    executor_resource["spark_driver_memory"] = min(int(cluster_resource["head_memory"] * SPARK_DRIVER_MEMORY_RATIO), 8192)

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
