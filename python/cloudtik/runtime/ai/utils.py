import os
from typing import Any, Dict

from cloudtik.core._private.providers import _get_node_provider, _get_workspace_provider
from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_AI
from cloudtik.core._private.utils import export_runtime_flags
from cloudtik.runtime.common.utils import get_runtime_services_of

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["mlflow.server:app", False, "MLflow", "head"],
]

RUNTIME_CONFIG_KEY = "ai"


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _with_runtime_environment_variables(runtime_config, config, provider, node_id: str):
    runtime_envs = {"AI_ENABLED": True}

    ai_config = runtime_config.get(RUNTIME_CONFIG_KEY, {})
    export_runtime_flags(
        ai_config, RUNTIME_CONFIG_KEY, runtime_envs)

    return runtime_envs


def publish_service_uri(cluster_config: Dict[str, Any], head_node_id: str) -> None:
    workspace_name = cluster_config.get("workspace_name")
    if workspace_name is None:
        return

    provider = _get_node_provider(cluster_config["provider"], cluster_config["cluster_name"])
    head_internal_ip = provider.internal_ip(head_node_id)
    service_uris = {"mlflow-service-uri": "http://{}:5001".format(head_internal_ip)}

    workspace_provider = _get_workspace_provider(cluster_config["provider"], workspace_name)
    workspace_provider.publish_global_variables(cluster_config, service_uris)


def _get_runtime_logs():
    mlflow_logs_dir = os.path.join(os.getenv("HOME"), "runtime", "mlflow", "logs")
    all_logs = {"mlflow": mlflow_logs_dir
                }
    return all_logs


def _get_runtime_services(cluster_head_ip):
    services = {
        "mlflow": {
            "name": "MLflow",
            "url": "http://{}:5001".format(cluster_head_ip)
        },
    }
    return services


def _get_runtime_service_ports(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    service_ports = {
        "mlflow": {
            "protocol": "TCP",
            "port": 5001,
        },
    }
    return service_ports


def get_runtime_services(config: Dict[str, Any]):
    return get_runtime_services_of(config, BUILT_IN_RUNTIME_AI)
