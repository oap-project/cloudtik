import os
import time
from typing import Any, Dict, Optional

from cloudtik.core._private.cli_logger import cli_logger
from cloudtik.core._private.cluster.cluster_config import _load_cluster_config
from cloudtik.core._private.cluster.cluster_tunnel_request import _request_rest_to_head
from cloudtik.core._private.core_utils import double_quote, get_env_string_value
from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_HDFS, BUILT_IN_RUNTIME_METASTORE, \
    BUILT_IN_RUNTIME_SPARK
from cloudtik.core._private.service_discovery.runtime_services import get_service_discovery_runtime
from cloudtik.core._private.service_discovery.utils import get_canonical_service_name, define_runtime_service_on_head, \
    get_service_discovery_config, SERVICE_DISCOVERY_FEATURE_SCHEDULER
from cloudtik.core._private.utils import \
    round_memory_size_to_gb, load_head_cluster_config, \
    RUNTIME_CONFIG_KEY, load_properties_file, save_properties_file, is_use_managed_cloud_storage, \
    print_json_formatted, get_config_for_update, get_runtime_config, PROVIDER_STORAGE_CONFIG_KEY
from cloudtik.core.scaling_policy import ScalingPolicy
from cloudtik.runtime.common.service_discovery.cluster import has_runtime_in_cluster
from cloudtik.runtime.common.service_discovery.runtime_discovery import \
    discover_metastore_on_head, discover_hdfs_on_head, discover_hdfs_from_workspace, \
    discover_metastore_from_workspace, is_hdfs_service_discovery, HDFS_URI_KEY, METASTORE_URI_KEY
from cloudtik.runtime.common.utils import get_runtime_endpoints_of, get_runtime_default_storage_of
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

SPARK_YARN_RESOURCE_MANAGER_PORT = 8032
SPARK_YARN_WEB_API_PORT = 8088
SPARK_HISTORY_SERVER_API_PORT = 18080
SPARK_JUPYTER_WEB_PORT = 8888

YARN_REQUEST_REST_RETRY_DELAY_S = 5
YARN_REQUEST_REST_RETRY_COUNT = 36

SPARK_YARN_SERVICE_NAME = "yarn"


def _get_config(runtime_config: Dict[str, Any]):
    return runtime_config.get(BUILT_IN_RUNTIME_SPARK, {})


def get_yarn_resource_memory_ratio(cluster_config: Dict[str, Any]):
    yarn_resource_memory_ratio = YARN_RESOURCE_MEMORY_RATIO
    spark_config = cluster_config.get(RUNTIME_CONFIG_KEY, {}).get(BUILT_IN_RUNTIME_SPARK, {})
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
    cluster_config = discover_hdfs_from_workspace(
        cluster_config, BUILT_IN_RUNTIME_SPARK)
    cluster_config = discover_metastore_from_workspace(
        cluster_config, BUILT_IN_RUNTIME_SPARK)

    return cluster_config


def _prepare_config_on_head(cluster_config: Dict[str, Any]):
    cluster_config = discover_hdfs_on_head(
        cluster_config, BUILT_IN_RUNTIME_SPARK)
    cluster_config = discover_metastore_on_head(
        cluster_config, BUILT_IN_RUNTIME_SPARK)

    # call validate config to fail earlier
    _validate_config(cluster_config, final=True)
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

    runtime_config = get_config_for_update(cluster_config, RUNTIME_CONFIG_KEY)
    spark_config = get_config_for_update(runtime_config, BUILT_IN_RUNTIME_SPARK)

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
    runtime_config = get_runtime_config(config)
    if not runtime_config:
        return None
    spark_config = _get_config(runtime_config)
    return spark_config.get("config")


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


def _with_hdfs_mount_method(spark_config, runtime_envs):
    mount_method = spark_config.get("hdfs_mount_method")
    if mount_method:
        runtime_envs["HDFS_MOUNT_METHOD"] = mount_method


def _with_runtime_environment_variables(
        runtime_config, config, provider, node_id: str):
    runtime_envs = {}
    spark_config = _get_config(runtime_config)
    cluster_runtime_config = config.get(RUNTIME_CONFIG_KEY)

    # export yarn memory ratio to use if configured by user
    yarn_resource_memory_ratio = spark_config.get("yarn_resource_memory_ratio")
    if yarn_resource_memory_ratio:
        runtime_envs["YARN_RESOURCE_MEMORY_RATIO"] = yarn_resource_memory_ratio

    # export yarn scheduler
    yarn_scheduler = spark_config.get("yarn_scheduler")
    if yarn_scheduler:
        runtime_envs["YARN_SCHEDULER"] = yarn_scheduler

    # We now support co-existence of local HDFS and remote HDFS, and cloud storage
    if has_runtime_in_cluster(
            cluster_runtime_config, BUILT_IN_RUNTIME_HDFS):
        runtime_envs["HDFS_ENABLED"] = True

    _with_hdfs_mount_method(spark_config, runtime_envs)

    if has_runtime_in_cluster(
            cluster_runtime_config, BUILT_IN_RUNTIME_METASTORE):
        runtime_envs["METASTORE_ENABLED"] = True
    return runtime_envs


def _configure(runtime_config, head: bool):
    spark_config = _get_config(runtime_config)

    hadoop_default_cluster = spark_config.get(
        "hadoop_default_cluster", False)
    if hadoop_default_cluster:
        os.environ["HADOOP_DEFAULT_CLUSTER"] = get_env_string_value(
            hadoop_default_cluster)

    hdfs_uri = spark_config.get(HDFS_URI_KEY)
    if hdfs_uri:
        os.environ["HDFS_NAMENODE_URI"] = hdfs_uri

    metastore_uri = spark_config.get(METASTORE_URI_KEY)
    if metastore_uri:
        os.environ["HIVE_METASTORE_URI"] = metastore_uri


def get_runtime_logs():
    hadoop_logs_dir = os.path.join(os.getenv("HADOOP_HOME"), "logs")
    spark_logs_dir = os.path.join(os.getenv("SPARK_HOME"), "logs")
    jupyter_logs_dir = os.path.join(os.getenv("HOME"), "runtime", "jupyter", "logs")
    all_logs = {"hadoop": hadoop_logs_dir,
                "spark": spark_logs_dir,
                "jupyter": jupyter_logs_dir
                }
    return all_logs


def _is_valid_storage_config(config: Dict[str, Any], final=False):
    runtime_config = get_runtime_config(config)
    # if HDFS enabled, we ignore the cloud storage configurations
    if has_runtime_in_cluster(runtime_config, BUILT_IN_RUNTIME_HDFS):
        return True
    # check if there is remote HDFS configured
    spark_config = _get_config(runtime_config)
    if spark_config.get(HDFS_URI_KEY) is not None:
        return True

    # Check any cloud storage is configured
    provider_config = config["provider"]
    if (PROVIDER_STORAGE_CONFIG_KEY in provider_config or
            (not final and is_use_managed_cloud_storage(config))):
        return True

    # if there is service discovery mechanism, assume we can get from service discovery
    if (not final and is_hdfs_service_discovery(spark_config) and
            get_service_discovery_runtime(runtime_config)):
        return True

    return False


def _validate_config(config: Dict[str, Any], final=False):
    if not _is_valid_storage_config(config, final=final):
        raise ValueError("No storage configuration found for Spark.")


def _get_runtime_endpoints(cluster_head_ip):
    endpoints = {
        "yarn": {
            "name": "Yarn",
            "url": "{}:{}".format(cluster_head_ip, SPARK_YARN_RESOURCE_MANAGER_PORT)
        },
        "yarn-web": {
            "name": "Yarn Web UI",
            "url": "http://{}:{}".format(cluster_head_ip, SPARK_YARN_WEB_API_PORT)
        },
        "jupyter-web": {
            "name": "Jupyter Web UI",
            "url": "http://{}:{}".format(cluster_head_ip, SPARK_JUPYTER_WEB_PORT),
            "info": "default password is \'cloudtik\'"
        },
        "history-server": {
            "name": "Spark History Server Web UI",
            "url": "http://{}:{}".format(cluster_head_ip, SPARK_HISTORY_SERVER_API_PORT)
        },
    }
    return endpoints


def _get_head_service_ports(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    service_ports = {
        "yarn": {
            "protocol": "TCP",
            "port": SPARK_YARN_RESOURCE_MANAGER_PORT,
        },
        "yarn-web": {
            "protocol": "TCP",
            "port": SPARK_YARN_WEB_API_PORT,
        },
        "jupyter-web": {
            "protocol": "TCP",
            "port": SPARK_JUPYTER_WEB_PORT,
        },
        "history-server": {
            "protocol": "TCP",
            "port": SPARK_HISTORY_SERVER_API_PORT,
        },
    }
    return service_ports


def _get_runtime_services(
        runtime_config: Dict[str, Any], cluster_name: str) -> Dict[str, Any]:
    spark_config = _get_config(runtime_config)
    service_discovery_config = get_service_discovery_config(spark_config)
    yarn_service_name = get_canonical_service_name(
        service_discovery_config, cluster_name, SPARK_YARN_SERVICE_NAME)
    services = {
        yarn_service_name: define_runtime_service_on_head(
            service_discovery_config, SPARK_YARN_RESOURCE_MANAGER_PORT,
            features=[SERVICE_DISCOVERY_FEATURE_SCHEDULER]),
    }
    return services


def _get_scaling_policy(
        runtime_config: Dict[str, Any],
        cluster_config: Dict[str, Any],
        head_ip: str) -> Optional[ScalingPolicy]:
    spark_config = _get_config(runtime_config)
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


def get_runtime_endpoints(config: Dict[str, Any]):
    return get_runtime_endpoints_of(config, BUILT_IN_RUNTIME_SPARK)

