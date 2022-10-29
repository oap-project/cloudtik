import os
from typing import Any, Dict, Optional

from cloudtik.core._private.cluster.cluster_config import _load_cluster_config
from cloudtik.core._private.cluster.cluster_rest_request import _request_rest_to_head
from cloudtik.core._private.core_utils import double_quote
from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_HDFS, BUILT_IN_RUNTIME_METASTORE, \
    BUILT_IN_RUNTIME_FLINK
from cloudtik.core._private.utils import merge_rooted_config_hierarchy, \
    _get_runtime_config_object, is_runtime_enabled, round_memory_size_to_gb, load_head_cluster_config, \
    RUNTIME_CONFIG_KEY, load_properties_file, save_properties_file, is_use_managed_cloud_storage, get_node_type_config, \
    print_json_formatted
from cloudtik.core._private.workspace.workspace_operator import _get_workspace_provider
from cloudtik.core.scaling_policy import ScalingPolicy
from cloudtik.runtime.common.utils import get_runtime_services_of, get_runtime_default_storage_of
from cloudtik.runtime.flink.scaling_policy import FlinkScalingPolicy

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["proc_resourcemanager", False, "ResourceManager", "head"],
    ["org.apache.flink.runtime.webmonitor.history.HistoryServer", False, "FlinkHistoryServer", "head"],
    ["proc_nodemanager", False, "NodeManager", "worker"],
]

RUNTIME_ROOT_PATH = os.path.abspath(os.path.dirname(__file__))

YARN_RESOURCE_MEMORY_RATIO = 0.8

FLINK_TASKMANAGER_MEMORY_RATIO = 1
FLINK_JOBMANAGER_MEMORY_RATIO = 0.02
FLINK_JOBMANAGER_MEMORY_MINIMUM = 1024
FLINK_JOBMANAGER_MEMORY_MAXIMUM = 8192
FLINK_TASKMANAGER_CORES_DEFAULT = 4
FLINK_ADDITIONAL_OVERHEAD = 1024
FLINK_TASKMANAGER_OVERHEAD_MINIMUM = 384
FLINK_TASKMANAGER_OVERHEAD_RATIO = 0.1

FLINK_YARN_WEB_API_PORT = 8088
FLINK_HISTORY_SERVER_API_PORT = 8082


def get_yarn_resource_memory_ratio(cluster_config: Dict[str, Any]):
    yarn_resource_memory_ratio = YARN_RESOURCE_MEMORY_RATIO
    flink_config = cluster_config.get(RUNTIME_CONFIG_KEY, {}).get("flink", {})
    memory_ratio = flink_config.get("yarn_resource_memory_ratio")
    if memory_ratio:
        yarn_resource_memory_ratio = memory_ratio
    return yarn_resource_memory_ratio


def get_flink_jobmanager_memory(worker_memory_for_flink: int) -> int:
    flink_jobmanager_memory = round_memory_size_to_gb(
        int(worker_memory_for_flink * FLINK_JOBMANAGER_MEMORY_RATIO))
    return max(min(flink_jobmanager_memory,
               FLINK_JOBMANAGER_MEMORY_MAXIMUM), FLINK_JOBMANAGER_MEMORY_MINIMUM)


def get_flink_overhead(worker_memory_for_flink: int) -> int:
    # Calculate the flink overhead including one jobmanager based on worker_memory_for_flink
    flink_jobmanager = get_flink_jobmanager_memory(worker_memory_for_flink)
    return flink_jobmanager + FLINK_ADDITIONAL_OVERHEAD


def get_flink_taskmanager_overhead(flink_taskmanager_memory_all: int) -> int:
    return max(int(flink_taskmanager_memory_all * FLINK_TASKMANAGER_OVERHEAD_RATIO),
               FLINK_TASKMANAGER_OVERHEAD_MINIMUM)


def _get_cluster_resources(
        cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    """Fills out flink runtime resource for available_node_types."""
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
    if "flink" not in runtime_config:
        runtime_config["flink"] = {}
    flink_config = runtime_config["flink"]

    workspace_provider = _get_workspace_provider(cluster_config["provider"], workspace_name)
    global_variables = workspace_provider.subscribe_global_variables(cluster_config)

    # 1) Try to use local hdfs first;
    # 2) Try to use defined hdfs_namenode_uri;
    # 3) If subscribed_hdfs_namenode_uri=true,try to subscribe global variables to find remote hdfs_namenode_uri

    if not is_runtime_enabled(runtime_config, "hdfs"):
        if flink_config.get("hdfs_namenode_uri") is None:
            if flink_config.get("auto_detect_hdfs", False):
                hdfs_namenode_uri = global_variables.get("hdfs-namenode-uri")
                if hdfs_namenode_uri is not None:
                    flink_config["hdfs_namenode_uri"] = hdfs_namenode_uri

    # Check metastore
    if not is_runtime_enabled(runtime_config, "metastore"):
        if flink_config.get("hive_metastore_uri") is None:
            if flink_config.get("auto_detect_metastore", True):
                hive_metastore_uri = global_variables.get("hive-metastore-uri")
                if hive_metastore_uri is not None:
                    flink_config["hive_metastore_uri"] = hive_metastore_uri

    return cluster_config


def _config_runtime_resources(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    cluster_resource = _get_cluster_resources(cluster_config)
    container_resource = {"yarn_container_maximum_vcores": cluster_resource["worker_cpu"]}

    yarn_resource_memory_ratio = get_yarn_resource_memory_ratio(cluster_config)
    worker_memory_for_yarn = round_memory_size_to_gb(
        int(cluster_resource["worker_memory"] * yarn_resource_memory_ratio))
    container_resource["yarn_container_maximum_memory"] = worker_memory_for_yarn

    # Calculate Flink taskmanager cores
    flink_taskmanager_cores = FLINK_TASKMANAGER_CORES_DEFAULT
    if flink_taskmanager_cores > cluster_resource["worker_cpu"]:
        flink_taskmanager_cores = cluster_resource["worker_cpu"]

    runtime_resource = {"flink_taskmanager_cores": flink_taskmanager_cores}

    # For Flink taskmanager memory, we use the following formula:
    # x = worker_memory_for_yarn
    # n = number_of_taskmanagers
    # m = flink_taskmanager_memory
    # a = flink_overhead (jobmanager_memory + others)
    # x = n * m + a

    number_of_taskmanagers = int(cluster_resource["worker_cpu"] / flink_taskmanager_cores)
    worker_memory_for_flink = round_memory_size_to_gb(
        int(worker_memory_for_yarn * FLINK_TASKMANAGER_MEMORY_RATIO))
    flink_overhead = round_memory_size_to_gb(
        get_flink_overhead(worker_memory_for_flink))
    worker_memory_for_taskmanagers = worker_memory_for_flink - flink_overhead
    flink_taskmanager_memory_all = round_memory_size_to_gb(
        int(worker_memory_for_taskmanagers / number_of_taskmanagers))
    runtime_resource["flink_taskmanager_memory"] =\
        flink_taskmanager_memory_all - get_flink_taskmanager_overhead(flink_taskmanager_memory_all)
    runtime_resource["flink_jobmanager_memory"] = get_flink_jobmanager_memory(worker_memory_for_flink)

    if RUNTIME_CONFIG_KEY not in cluster_config:
        cluster_config[RUNTIME_CONFIG_KEY] = {}
    runtime_config = cluster_config[RUNTIME_CONFIG_KEY]

    if "flink" not in runtime_config:
        runtime_config["flink"] = {}
    flink_config = runtime_config["flink"]

    flink_config["yarn_container_resource"] = container_resource
    flink_config["flink_resource"] = runtime_resource
    return cluster_config


def get_runtime_processes():
    return RUNTIME_PROCESSES


def _is_runtime_scripts(script_file):
    if script_file.endswith(".scala"):
        return True

    return False


def _get_runnable_command(target):
    command_parts = ["flink", "-i", double_quote(target)]
    return command_parts


def _get_flink_config(config: Dict[str, Any]):
    runtime = config.get(RUNTIME_CONFIG_KEY)
    if not runtime:
        return None

    flink = runtime.get("flink")
    if not flink:
        return None

    return flink.get("config")


def update_flink_configurations():
    # Merge user specified configuration and default configuration
    config = load_head_cluster_config()
    flink_config = _get_flink_config(config)
    if not flink_config:
        return

    flink_conf_file = os.path.join(os.getenv("FLINK_HOME"), "conf/flink-conf.yaml")

    # Read in the existing configurations
    flink_conf, comments = load_properties_file(flink_conf_file, ':')

    # Merge with the user configurations
    flink_conf.update(flink_config)

    # Write back the configuration file
    save_properties_file(flink_conf_file, flink_conf, separator=': ', comments=comments)


def _with_runtime_environment_variables(runtime_config, config, provider, node_id: str):
    runtime_envs = {}
    flink_config = runtime_config.get("flink", {})
    cluster_runtime_config = config.get(RUNTIME_CONFIG_KEY)

    # export yarn memory ratio to use if configured by user
    yarn_resource_memory_ratio = flink_config.get("yarn_resource_memory_ratio")
    if yarn_resource_memory_ratio:
        runtime_envs["YARN_RESOURCE_MEMORY_RATIO"] = yarn_resource_memory_ratio

    # export yarn scheduler
    yarn_scheduler = flink_config.get("yarn_scheduler")
    if yarn_scheduler:
        runtime_envs["YARN_SCHEDULER"] = yarn_scheduler

    # 1) Try to use local hdfs first;
    # 2) Try to use defined hdfs_namenode_uri;
    # 3) Try to use provider storage;
    if is_runtime_enabled(cluster_runtime_config, BUILT_IN_RUNTIME_HDFS):
        runtime_envs["HDFS_ENABLED"] = True
    else:
        if flink_config.get("hdfs_namenode_uri") is not None:
            runtime_envs["HDFS_NAMENODE_URI"] = flink_config.get("hdfs_namenode_uri")

        # We always export the cloud storage even for remote HDFS case
        node_type_config = get_node_type_config(config, provider, node_id)
        provider_envs = provider.with_environment_variables(node_type_config, node_id)
        runtime_envs.update(provider_envs)

    # 1) Try to use local metastore if there is one started;
    # 2) Try to use defined metastore_uri;
    if is_runtime_enabled(cluster_runtime_config, BUILT_IN_RUNTIME_METASTORE):
        runtime_envs["METASTORE_ENABLED"] = True
    elif flink_config.get("hive_metastore_uri") is not None:
        runtime_envs["HIVE_METASTORE_URI"] = flink_config.get("hive_metastore_uri")
    return runtime_envs


def get_runtime_logs():
    hadoop_logs_dir = os.path.join(os.getenv("HADOOP_HOME"), "logs")
    flink_logs_dir = os.path.join(os.getenv("FLINK_HOME"), "logs")
    all_logs = {"hadoop": hadoop_logs_dir,
                "flink": flink_logs_dir,
                "other": "/tmp/logs"
                }
    return all_logs


def _validate_config(config: Dict[str, Any], provider):
    # if HDFS enabled, we ignore the cloud storage configurations
    if not is_runtime_enabled(config.get(RUNTIME_CONFIG_KEY), "hdfs"):
        # Check any cloud storage is configured
        provider_config = config["provider"]
        if ("storage" not in provider_config) and \
                not is_use_managed_cloud_storage(config):
            raise ValueError("No storage configuration found for Flink.")


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
    urls = [
        {
            "id": "yarn-web",
            "name": "Yarn Web UI",
            "url": "http://{}:{}".format(cluster_head_ip, FLINK_YARN_WEB_API_PORT)
         },
        {
            "id": "jupyter-web",
            "name": "Jupyter Web UI",
            "url": "http://{}:8888".format(cluster_head_ip),
            "info": "default password is \'cloudtik\'"
         },
        {
            "id": "flink-history",
            "name": "Flink History Server Web UI",
            "url": "http://{}:{}".format(cluster_head_ip, FLINK_HISTORY_SERVER_API_PORT)
         },
    ]
    return urls


def _get_runtime_service_ports(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    service_ports = {
        "yarn-web": {
            "protocol": "TCP",
            "port": FLINK_YARN_WEB_API_PORT,
        },
        "jupyter-web": {
            "protocol": "TCP",
            "port": 8888,
        },
        "flink-history": {
            "protocol": "TCP",
            "port": FLINK_HISTORY_SERVER_API_PORT,
        },
    }
    return service_ports


def _get_scaling_policy(
        runtime_config: Dict[str, Any],
        cluster_config: Dict[str, Any],
        head_ip: str) -> Optional[ScalingPolicy]:
    flink_config = runtime_config.get("flink", {})
    scaling_config = flink_config.get("scaling", {})

    node_resource_states = scaling_config.get("node_resource_states", True)
    auto_scaling = scaling_config.get("auto_scaling", False)
    if not node_resource_states and not auto_scaling:
        return None

    return FlinkScalingPolicy(
        cluster_config, head_ip,
        rest_port=FLINK_YARN_WEB_API_PORT)


def print_request_rest_jobs(
        cluster_config_file: str, cluster_name: str, endpoint: str,
        on_head: bool = False):
    config = _load_cluster_config(cluster_config_file, cluster_name)
    response = request_rest_jobs(config, endpoint,
                                 on_head=on_head)
    print_json_formatted(response)


def request_rest_jobs(
        config: Dict[str, Any], endpoint: str,
        on_head: bool = False):
    if endpoint is None:
        endpoint = "/jobs/overview"
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    return _request_rest_to_head(
        config, endpoint, FLINK_HISTORY_SERVER_API_PORT,
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
        config, endpoint, FLINK_YARN_WEB_API_PORT,
        on_head=on_head)


def get_runtime_default_storage(config: Dict[str, Any]):
    return get_runtime_default_storage_of(config, BUILT_IN_RUNTIME_FLINK)


def get_runtime_services(config: Dict[str, Any]):
    return get_runtime_services_of(config, BUILT_IN_RUNTIME_FLINK)
