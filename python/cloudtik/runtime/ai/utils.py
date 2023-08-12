import os
from typing import Any, Dict

from cloudtik.core._private.providers import _get_node_provider, _get_workspace_provider
from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_AI
from cloudtik.core._private.service_discovery.utils import get_canonical_service_name, define_runtime_service_on_head, \
    get_service_discovery_config, SERVICE_DISCOVERY_PROTOCOL_HTTP
from cloudtik.core._private.utils import export_runtime_flags
from cloudtik.runtime.common.utils import get_runtime_endpoints_of

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["mlflow.server:app", False, "MLflow", "head"],
]

AI_RUNTIME_CONFIG_KEY = "ai"

MLFLOW_SERVICE_NAME = "mlflow"
MLFLOW_SERVICE_PORT = 5001


def _get_config(runtime_config: Dict[str, Any]):
    return runtime_config.get(AI_RUNTIME_CONFIG_KEY, {})


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _with_runtime_environment_variables(runtime_config, config, provider, node_id: str):
    runtime_envs = {"AI_ENABLED": True}

    ai_config = _get_config(runtime_config)
    export_runtime_flags(
        ai_config, AI_RUNTIME_CONFIG_KEY, runtime_envs)

    return runtime_envs


def publish_service_endpoint(cluster_config: Dict[str, Any], head_node_id: str) -> None:
    workspace_name = cluster_config.get("workspace_name")
    if workspace_name is None:
        return

    provider = _get_node_provider(cluster_config["provider"], cluster_config["cluster_name"])
    head_internal_ip = provider.internal_ip(head_node_id)
    service_endpoints = {"mlflow-uri": "http://{}:{}".format(
        head_internal_ip, MLFLOW_SERVICE_PORT)}

    workspace_provider = _get_workspace_provider(cluster_config["provider"], workspace_name)
    workspace_provider.publish_global_variables(cluster_config, service_endpoints)


def _get_runtime_logs():
    mlflow_logs_dir = os.path.join(os.getenv("HOME"), "runtime", "mlflow", "logs")
    all_logs = {"mlflow": mlflow_logs_dir
                }
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
