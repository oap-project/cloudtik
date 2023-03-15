import os
import time
from typing import Any, Dict, Optional

from cloudtik.core._private.cli_logger import cli_logger
from cloudtik.core._private.cluster.cluster_config import _load_cluster_config
from cloudtik.core._private.cluster.cluster_rest_request import _request_rest_to_head
from cloudtik.core._private.core_utils import double_quote
from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_HDFS, BUILT_IN_RUNTIME_METASTORE, \
    BUILT_IN_RUNTIME_SPARK
from cloudtik.core._private.utils import merge_rooted_config_hierarchy, \
    _get_runtime_config_object, is_runtime_enabled, round_memory_size_to_gb, load_head_cluster_config, \
    RUNTIME_CONFIG_KEY, load_properties_file, save_properties_file, is_use_managed_cloud_storage, get_node_type_config, \
    print_json_formatted
from cloudtik.core._private.workspace.workspace_operator import _get_workspace_provider
from cloudtik.core.scaling_policy import ScalingPolicy
from cloudtik.runtime.common.utils import get_runtime_services_of, get_runtime_default_storage_of
from cloudtik.runtime.spark.scaling_policy import SparkScalingPolicy

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["proc_resourcemanager", False, "ResourceManager", "head"],
    ["org.apache.spark.deploy.history.HistoryServer", False, "SparkHistoryServer", "head"],
    ["proc_nodemanager", False, "NodeManager", "worker"],
]

RUNTIME_ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
SPARK_RUNTIME_CONFIG_KEY = "spark"

YARN_RESOURCE_MEMORY_RATIO = 0.8
SPARK_EXECUTOR_MEMORY_RATIO = 1
SPARK_DRIVER_MEMORY_RATIO = 0.1
SPARK_APP_MASTER_MEMORY_RATIO = 0.02
SPARK_DRIVER_MEMORY_MINIMUM = 1024
SPARK_DRIVER_MEMORY_MAXIMUM = 8192
SPARK_EXECUTOR_CORES_DEFAULT = 4
SPARK_EXECUTOR_CORES_SINGLE_BOUND = 8
SPARK_ADDITIONAL_OVERHEAD = 1024
SPARK_EXECUTOR_OVERHEAD_MINIMUM = 384
SPARK_EXECUTOR_OVERHEAD_RATIO = 0.1

SPARK_YARN_WEB_API_PORT = 8088
SPARK_HISTORY_SERVER_API_PORT = 18080

YARN_REQUEST_REST_RETRY_DELAY_S = 5
YARN_REQUEST_REST_RETRY_COUNT = 36


def get_yarn_resource_memory_ratio(cluster_config: Dict[str, Any]):
    yarn_resource_memory_ratio = YARN_RESOURCE_MEMORY_RATIO
    spark_config = cluster_config.get(RUNTIME_CONFIG_KEY, {}).get(SPARK_RUNTIME_CONFIG_KEY, {})
    memory_ratio = spark_config.get("yarn_resource_memory_ratio")
    if memory_ratio:
        yarn_resource_memory_ratio = memory_ratio
    return yarn_resource_memory_ratio


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
    workspace_name = cluster_config.get("workspace_name")
    if workspace_name is None:
        return cluster_config

    runtime_config = cluster_config.get(RUNTIME_CONFIG_KEY)
    if SPARK_RUNTIME_CONFIG_KEY not in runtime_config:
        runtime_config[SPARK_RUNTIME_CONFIG_KEY] = {}
    spark_config = runtime_config[SPARK_RUNTIME_CONFIG_KEY]

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
    worker_cpu = cluster_resource["worker_cpu"]

    container_resource = {"yarn_container_maximum_vcores": worker_cpu}

    yarn_resource_memory_ratio = get_yarn_resource_memory_ratio(cluster_config)
    worker_memory_for_yarn = round_memory_size_to_gb(
        int(cluster_resource["worker_memory"] * yarn_resource_memory_ratio))
    container_resource["yarn_container_maximum_memory"] = worker_memory_for_yarn

    executor_resource = {"spark_driver_memory": get_spark_driver_memory(cluster_resource)}

    # Calculate Spark executor cores
    spark_executor_cores = SPARK_EXECUTOR_CORES_DEFAULT
    if spark_executor_cores > worker_cpu:
        spark_executor_cores = worker_cpu
    elif (worker_cpu % SPARK_EXECUTOR_CORES_DEFAULT) != 0:
        if worker_cpu <= SPARK_EXECUTOR_CORES_SINGLE_BOUND:
            spark_executor_cores = worker_cpu
        elif worker_cpu <= SPARK_EXECUTOR_CORES_SINGLE_BOUND * 2:
            if (worker_cpu % 2) != 0:
                # Overload 1 core
                worker_cpu += 1
            spark_executor_cores = int(worker_cpu / 2)
        else:
            # Overload max number of SPARK_EXECUTOR_CORES_DEFAULT - 1 cores
            overload_cores = SPARK_EXECUTOR_CORES_DEFAULT - (worker_cpu % SPARK_EXECUTOR_CORES_DEFAULT)
            worker_cpu += overload_cores

    executor_resource["spark_executor_cores"] = spark_executor_cores

    # For Spark executor memory, we use the following formula:
    # x = worker_memory_for_yarn
    # n = number_of_executors
    # m = spark_executor_memory
    # a = spark_overhead (app_master_memory + others)
    # x = n * m + a

    number_of_executors = int(worker_cpu / spark_executor_cores)
    worker_memory_for_spark = round_memory_size_to_gb(
        int(worker_memory_for_yarn * SPARK_EXECUTOR_MEMORY_RATIO))
    spark_overhead = round_memory_size_to_gb(
        get_spark_overhead(worker_memory_for_spark))
    worker_memory_for_executors = worker_memory_for_spark - spark_overhead
    spark_executor_memory_all = round_memory_size_to_gb(
        int(worker_memory_for_executors / number_of_executors))
    executor_resource["spark_executor_memory"] = \
        spark_executor_memory_all - get_spark_executor_overhead(spark_executor_memory_all)

    if RUNTIME_CONFIG_KEY not in cluster_config:
        cluster_config[RUNTIME_CONFIG_KEY] = {}
    runtime_config = cluster_config[RUNTIME_CONFIG_KEY]

    if SPARK_RUNTIME_CONFIG_KEY not in runtime_config:
        runtime_config[SPARK_RUNTIME_CONFIG_KEY] = {}
    spark_config = runtime_config[SPARK_RUNTIME_CONFIG_KEY]

    spark_config["yarn_container_resource"] = container_resource
    spark_config["spark_executor_resource"] = executor_resource
    return cluster_config


def get_runtime_processes():
    return RUNTIME_PROCESSES


def _is_runtime_scripts(script_file):
    if script_file.endswith(".scala") or script_file.endswith(".jar") or script_file.endswith(".py"):
        return True
    return False


def _get_runnable_command(target, runtime_options):
    command_parts = []
    if target.endswith(".scala"):
        command_parts = ["spark-shell", "-i", double_quote(target)]
    elif target.endswith(".jar") or target.endswith(".py"):
        command_parts = ["spark-submit"]
        if runtime_options is not None:
            command_parts += runtime_options
        command_parts += [double_quote(target)]
    return command_parts


def _get_spark_config(config: Dict[str, Any]):
    runtime = config.get(RUNTIME_CONFIG_KEY)
    if not runtime:
        return None

    spark = runtime.get(SPARK_RUNTIME_CONFIG_KEY)
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
    spark_config = runtime_config.get(SPARK_RUNTIME_CONFIG_KEY, {})
    cluster_runtime_config = config.get(RUNTIME_CONFIG_KEY)

    # export yarn memory ratio to use if configured by user
    yarn_resource_memory_ratio = spark_config.get("yarn_resource_memory_ratio")
    if yarn_resource_memory_ratio:
        runtime_envs["YARN_RESOURCE_MEMORY_RATIO"] = yarn_resource_memory_ratio

    # export yarn scheduler
    yarn_scheduler = spark_config.get("yarn_scheduler")
    if yarn_scheduler:
        runtime_envs["YARN_SCHEDULER"] = yarn_scheduler

    # 1) Try to use local hdfs first;
    # 2) Try to use defined hdfs_namenode_uri;
    # 3) Try to use provider storage;
    if is_runtime_enabled(cluster_runtime_config, BUILT_IN_RUNTIME_HDFS):
        runtime_envs["HDFS_ENABLED"] = True
    else:
        if spark_config.get("hdfs_namenode_uri") is not None:
            runtime_envs["HDFS_NAMENODE_URI"] = spark_config.get("hdfs_namenode_uri")

        # We always export the cloud storage even for remote HDFS case
        node_type_config = get_node_type_config(config, provider, node_id)
        provider_envs = provider.with_environment_variables(node_type_config, node_id)
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
    jupyter_logs_dir = os.path.join(os.getenv("HOME"), "runtime", "jupyter", "logs")
    all_logs = {"hadoop": hadoop_logs_dir,
                "spark": spark_logs_dir,
                "jupyter": jupyter_logs_dir
                }
    return all_logs


def _validate_config(config: Dict[str, Any], provider):
    # if HDFS enabled, we ignore the cloud storage configurations
    if not is_runtime_enabled(config.get(RUNTIME_CONFIG_KEY), "hdfs"):
        # Check any cloud storage is configured
        provider_config = config["provider"]
        if ("storage" not in provider_config) and \
                not is_use_managed_cloud_storage(config):
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


def _get_runtime_services(cluster_head_ip):
    services = {
        "yarn-web": {
            "name": "Yarn Web UI",
            "url": "http://{}:{}".format(cluster_head_ip, SPARK_YARN_WEB_API_PORT)
        },
        "jupyter-web": {
            "name": "Jupyter Web UI",
            "url": "http://{}:8888".format(cluster_head_ip),
            "info": "default password is \'cloudtik\'"
        },
        "history-server": {
            "name": "Spark History Server Web UI",
            "url": "http://{}:{}".format(cluster_head_ip, SPARK_HISTORY_SERVER_API_PORT)
        },
    }
    return services


def _get_runtime_service_ports(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    service_ports = {
        "yarn-web": {
            "protocol": "TCP",
            "port": SPARK_YARN_WEB_API_PORT,
        },
        "jupyter-web": {
            "protocol": "TCP",
            "port": 8888,
        },
        "history-server": {
            "protocol": "TCP",
            "port": SPARK_HISTORY_SERVER_API_PORT,
        },
    }
    return service_ports


def _get_scaling_policy(
        runtime_config: Dict[str, Any],
        cluster_config: Dict[str, Any],
        head_ip: str) -> Optional[ScalingPolicy]:
    spark_config = runtime_config.get(SPARK_RUNTIME_CONFIG_KEY, {})
    if "scaling" not in spark_config:
        return None

    return SparkScalingPolicy(
        cluster_config, head_ip,
        rest_port=SPARK_YARN_WEB_API_PORT)


def print_request_rest_applications(
        cluster_config_file: str, cluster_name: str, endpoint: str,
        on_head: bool = False):
    config = _load_cluster_config(cluster_config_file, cluster_name)
    response = request_rest_applications(config, endpoint,
                                         on_head=on_head)
    print_json_formatted(response)


def request_rest_applications(
        config: Dict[str, Any], endpoint: str,
        on_head: bool = False):
    if endpoint is None:
        endpoint = "/applications"
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    endpoint = "api/v1" + endpoint
    return _request_rest_to_head(
        config, endpoint, SPARK_HISTORY_SERVER_API_PORT,
        on_head=on_head)


def print_request_rest_yarn(
        cluster_config_file: str, cluster_name: str, endpoint: str,
        on_head: bool = False):
    config = _load_cluster_config(cluster_config_file, cluster_name)
    response = request_rest_yarn(config, endpoint,
                                 on_head=on_head)
    print_json_formatted(response)


def request_rest_yarn(
        config: Dict[str, Any], endpoint: str,
        on_head: bool = False):
    if endpoint is None:
        endpoint = "/cluster/metrics"
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    endpoint = "ws/v1" + endpoint
    return _request_rest_to_head(
        config, endpoint, SPARK_YARN_WEB_API_PORT,
        on_head=on_head)


def request_rest_yarn_with_retry(
        config: Dict[str, Any], endpoint: str, retry=YARN_REQUEST_REST_RETRY_COUNT):
    while retry > 0:
        try:
            response = request_rest_yarn(config, endpoint)
            return response
        except Exception as e:
            retry = retry - 1
            if retry > 0:
                cli_logger.warning(f"Error when requesting yarn api. Retrying in {YARN_REQUEST_REST_RETRY_DELAY_S} seconds.")
                time.sleep(YARN_REQUEST_REST_RETRY_DELAY_S)
            else:
                cli_logger.error("Failed to request yarn api: {}", str(e))
                raise e


def get_runtime_default_storage(config: Dict[str, Any]):
    return get_runtime_default_storage_of(config, BUILT_IN_RUNTIME_SPARK)


def get_runtime_services(config: Dict[str, Any]):
    return get_runtime_services_of(config, BUILT_IN_RUNTIME_SPARK)
