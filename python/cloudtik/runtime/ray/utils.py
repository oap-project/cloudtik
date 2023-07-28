import os
from typing import Any, Dict, Optional

from cloudtik.core._private.service_discovery.utils import SERVICE_DISCOVERY_PROTOCOL, SERVICE_DISCOVERY_PORT, \
    SERVICE_DISCOVERY_NODE_KIND, SERVICE_DISCOVERY_NODE_KIND_HEAD, SERVICE_DISCOVERY_PROTOCOL_TCP, \
    get_canonical_service_name
from cloudtik.core._private.utils import get_node_type_config
from cloudtik.core.scaling_policy import ScalingPolicy
from cloudtik.runtime.ray.scaling_policy import RayScalingPolicy

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["gcs_server", True, "GCSServer", "head"],
    ["raylet", True, "Raylet", "node"],
    ["plasma_store", True, "PlasmaStore", "node"],
]

RAY_RUNTIME_CONFIG_KEY = "ray"

# The default proportion of available memory allocated to system and runtime overhead
RAY_DEFAULT_SHARED_MEMORY_PROPORTION = 0.3

RAY_SERVICE_NAME = "ray"
RAY_SERVICE_PORT = 6379
RAY_DASHBOARD_PORT = 8265


def _get_config(runtime_config: Dict[str, Any]):
    return runtime_config.get(RAY_RUNTIME_CONFIG_KEY, {})


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _with_runtime_environment_variables(runtime_config, config, provider, node_id: str):
    runtime_envs = {"RAY_ENABLED": True}
    ray_config = _get_config(runtime_config)

    _with_memory_configurations(
        runtime_envs, ray_config=ray_config,
        config=config, provider=provider, node_id=node_id)

    # We need export the cloud storage
    node_type_config = get_node_type_config(config, provider, node_id)
    provider_envs = provider.with_environment_variables(node_type_config, node_id)
    runtime_envs.update(provider_envs)

    return runtime_envs


def _get_runtime_shared_memory_ratio(runtime_config, config):
    return RAY_DEFAULT_SHARED_MEMORY_PROPORTION


def _get_runtime_logs():
    logs_dir = os.path.join("/tmp", "ray")
    all_logs = {"ray": logs_dir}
    return all_logs


def _get_runtime_endpoints(cluster_head_ip):
    endpoints = {
        "ray": {
            "name": "Ray",
            "url": "{}:{}".format(
                cluster_head_ip, RAY_SERVICE_PORT)
        },
        "dashboard": {
            "name": "Ray Dashboard",
            "url": "http://{}:{}".format(
                cluster_head_ip, RAY_DASHBOARD_PORT)
        }
    }
    return endpoints


def _with_memory_configurations(
        runtime_envs: Dict[str, Any], ray_config: Dict[str, Any],
        config: Dict[str, Any], provider, node_id: str):
    pass


def _get_head_service_ports(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    service_ports = {
        "ray": {
            "protocol": "TCP",
            "port": RAY_SERVICE_PORT,
        },
        "dashboard": {
            "protocol": "TCP",
            "port": RAY_DASHBOARD_PORT,
        },
    }
    return service_ports


def _get_scaling_policy(
        runtime_config: Dict[str, Any],
        cluster_config: Dict[str, Any],
        head_ip: str) -> Optional[ScalingPolicy]:
    ray_config = _get_config(runtime_config)
    if "scaling" not in ray_config:
        return None

    return RayScalingPolicy(
        cluster_config, head_ip,
        ray_port=RAY_SERVICE_PORT)


def _get_runtime_services(
        runtime_config: Dict[str, Any], cluster_name: str) -> Dict[str, Any]:
    ray_config = _get_config(runtime_config)
    service_name = get_canonical_service_name(
        ray_config, cluster_name, RAY_SERVICE_NAME)
    services = {
        service_name: {
            SERVICE_DISCOVERY_PROTOCOL: SERVICE_DISCOVERY_PROTOCOL_TCP,
            SERVICE_DISCOVERY_PORT: RAY_SERVICE_PORT,
            SERVICE_DISCOVERY_NODE_KIND: SERVICE_DISCOVERY_NODE_KIND_HEAD
        },
    }
    return services
