import os
from typing import Any, Dict

from cloudtik.core._private.core_utils import get_config_for_update, get_string_value_for_env
from cloudtik.core._private.providers import _get_node_provider
from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_AI
from cloudtik.core._private.service_discovery.utils import get_canonical_service_name, define_runtime_service_on_head, \
    get_service_discovery_config, SERVICE_DISCOVERY_PROTOCOL_HTTP
from cloudtik.core._private.utils import export_runtime_flags, RUNTIME_CONFIG_KEY, get_runtime_config
from cloudtik.runtime.common.service_discovery.discovery import DiscoveryType
from cloudtik.runtime.common.service_discovery.runtime_discovery import discover_hdfs
from cloudtik.runtime.common.service_discovery.workspace import register_service_to_workspace
from cloudtik.runtime.common.utils import get_runtime_endpoints_of

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["mlflow.server:app", False, "MLflow", "head"],
]

AI_HDFS_NAMENODE_URI_KEY = "hdfs_namenode_uri"
AI_HDFS_SERVICE_SELECTOR_KEY = "hdfs_service_selector"

MLFLOW_SERVICE_NAME = "mlflow"
MLFLOW_SERVICE_PORT = 5001


def _get_config(runtime_config: Dict[str, Any]):
    return runtime_config.get(BUILT_IN_RUNTIME_AI, {})


def _is_hdfs_service_discovery(ai_config):
    return ai_config.get("hdfs_service_discovery", True)


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _config_depended_services(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    runtime_config = get_config_for_update(cluster_config, RUNTIME_CONFIG_KEY)
    ai_config = get_config_for_update(runtime_config, BUILT_IN_RUNTIME_AI)

    if ai_config.get(AI_HDFS_NAMENODE_URI_KEY) is None:
        if _is_hdfs_service_discovery(ai_config):
            hdfs_namenode_uri = discover_hdfs(
                ai_config, AI_HDFS_SERVICE_SELECTOR_KEY,
                cluster_config=cluster_config,
                discovery_type=DiscoveryType.WORKSPACE)
            if hdfs_namenode_uri is not None:
                ai_config[AI_HDFS_NAMENODE_URI_KEY] = hdfs_namenode_uri

    return cluster_config


def _prepare_config_on_head(cluster_config: Dict[str, Any]):
    cluster_config = _discover_hdfs_on_head(cluster_config)
    return cluster_config


def _discover_hdfs_on_head(cluster_config: Dict[str, Any]):
    runtime_config = get_runtime_config(cluster_config)
    ai_config = _get_config(runtime_config)
    if not _is_hdfs_service_discovery(ai_config):
        return cluster_config

    hdfs_namenode_uri = ai_config.get(AI_HDFS_NAMENODE_URI_KEY)
    if hdfs_namenode_uri:
        # HDFS already configured
        return cluster_config

    # There is service discovery to come here
    hdfs_namenode_uri = discover_hdfs(
        ai_config, AI_HDFS_SERVICE_SELECTOR_KEY,
        cluster_config=cluster_config,
        discovery_type=DiscoveryType.CLUSTER)
    if hdfs_namenode_uri:
        ai_config = get_config_for_update(
            runtime_config, BUILT_IN_RUNTIME_AI)
        ai_config[AI_HDFS_NAMENODE_URI_KEY] = hdfs_namenode_uri
    return cluster_config


def _with_runtime_environment_variables(
        runtime_config, config, provider, node_id: str):
    runtime_envs = {"AI_ENABLED": True}

    ai_config = _get_config(runtime_config)
    export_runtime_flags(
        ai_config, BUILT_IN_RUNTIME_AI, runtime_envs)

    return runtime_envs


def _configure(runtime_config, head: bool):
    ai_config = _get_config(runtime_config)

    hadoop_default_cluster = ai_config.get(
        "hadoop_default_cluster", False)
    if hadoop_default_cluster:
        os.environ["HADOOP_DEFAULT_CLUSTER"] = get_string_value_for_env(
            hadoop_default_cluster)

    hdfs_namenode_uri = ai_config.get(AI_HDFS_NAMENODE_URI_KEY)
    if hdfs_namenode_uri:
        os.environ["HDFS_NAMENODE_URI"] = hdfs_namenode_uri


def register_service(cluster_config: Dict[str, Any], head_node_id: str) -> None:
    provider = _get_node_provider(
        cluster_config["provider"], cluster_config["cluster_name"])
    head_ip = provider.internal_ip(head_node_id)
    register_service_to_workspace(
        cluster_config, BUILT_IN_RUNTIME_AI,
        service_addresses=[(head_ip, MLFLOW_SERVICE_PORT)],
        service_name=MLFLOW_SERVICE_NAME)


def _get_runtime_logs():
    mlflow_logs_dir = os.path.join(
        os.getenv("HOME"), "runtime", "mlflow", "logs")
    all_logs = {"mlflow": mlflow_logs_dir}
    return all_logs


def _get_runtime_endpoints(cluster_head_ip):
    endpoints = {
        "mlflow": {
            "name": "MLflow",
            "url": "http://{}:{}".format(cluster_head_ip, MLFLOW_SERVICE_PORT)
        },
    }
    return endpoints


def _get_head_service_ports(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    service_ports = {
        "mlflow": {
            "protocol": "TCP",
            "port": MLFLOW_SERVICE_PORT,
        },
    }
    return service_ports


def get_runtime_endpoints(config: Dict[str, Any]):
    return get_runtime_endpoints_of(config, BUILT_IN_RUNTIME_AI)


def _get_runtime_services(
        runtime_config: Dict[str, Any], cluster_name: str) -> Dict[str, Any]:
    ai_config = _get_config(runtime_config)
    service_discovery_config = get_service_discovery_config(ai_config)
    service_name = get_canonical_service_name(
        service_discovery_config, cluster_name, MLFLOW_SERVICE_NAME)
    services = {
        service_name: define_runtime_service_on_head(
            service_discovery_config, MLFLOW_SERVICE_PORT,
            protocol=SERVICE_DISCOVERY_PROTOCOL_HTTP),
    }
    return services
