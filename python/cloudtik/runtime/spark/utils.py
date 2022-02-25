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


def config_spark_runtime_resources(
        cluster_config: Dict[str, Any], cluster_resource: Dict[str, Any]) -> Dict[str, Any]:
    cluster_config = copy.deepcopy(cluster_config)
    spark_executor_resource = {}
    if int(cluster_resource["worker_cpu"]) < 4:
        spark_executor_resource["spark_executor_cores"] = cluster_resource["worker_cpu"]
        spark_executor_resource["spark_executor_memory"] = int(cluster_resource["worker_memory"] * 0.8)
    else:
        spark_executor_resource["spark_executor_cores"] = 4
        spark_executor_resource["spark_executor_memory"] = 8096
    spark_executor_resource["spark_driver_memory"] = int(cluster_resource["head_memory"] * 0.8)
    cluster_config["spark_executor_resource"] = spark_executor_resource
    return cluster_config


def get_runtime_processes():
    return SPARK_RUNTIME_PROCESSES
