from typing import Dict, Any

from cloudtik.core._private.constants import CLOUDTIK_DEFAULT_PORT, CLOUDTIK_METRIC_PORT, CLOUDTIK_RUNTIME_NAME
from cloudtik.core._private.runtime_factory import _get_runtime, BUILT_IN_RUNTIME_CONSUL
from cloudtik.core._private.service_discovery.utils import match_service_node, get_canonical_service_name, \
    define_runtime_service_on_head, get_service_discovery_config
from cloudtik.core._private.utils import _get_node_type_specific_runtime_config, RUNTIME_TYPES_CONFIG_KEY, \
    RUNTIME_CONFIG_KEY


CLOUDTIK_REDIS_SERVICE_NAME = "cloudtik-redis"
CLOUDTIK_CLUSTER_CONTROLLER_METRICS_SERVICE_NAME = "cluster-controller-metrics"

CLOUDTIK_CLUSTER_CONTROLLER_METRICS_PORT = CLOUDTIK_METRIC_PORT
CLOUDTIK_REDIS_SERVICE_PORT = CLOUDTIK_DEFAULT_PORT


def get_runtime_services(config: Dict[str, Any]):
    # for all the runtimes, query its services per node type
    services_map = {}
    cluster_name = config["cluster_name"]
    available_node_types = config["available_node_types"]
    head_node_type = config["head_node_type"]
    built_in_services = _get_built_in_services(config, cluster_name)

    for node_type in available_node_types:
        head = True if node_type == head_node_type else False
        services_for_node_type = {}

        for service_name, runtime_service in built_in_services.items():
            if match_service_node(runtime_service, head):
                services_for_node_type[service_name] = (CLOUDTIK_RUNTIME_NAME, runtime_service)

        runtime_config = _get_node_type_specific_runtime_config(config, node_type)
        if runtime_config:
            # services runtimes
            runtime_types = runtime_config.get(RUNTIME_TYPES_CONFIG_KEY, [])
            for runtime_type in runtime_types:
                if runtime_type == BUILT_IN_RUNTIME_CONSUL:
                    continue

                runtime = _get_runtime(runtime_type, runtime_config)
                services = runtime.get_runtime_services(cluster_name)
                if not services:
                    continue

                for service_name, runtime_service in services.items():
                    if match_service_node(runtime_service, head):
                        services_for_node_type[service_name] = (runtime_type, runtime_service)
        if services_for_node_type:
            services_map[node_type] = services_for_node_type
    return services_map


def _get_built_in_services(config: Dict[str, Any], cluster_name):
    runtime_config = config.get(RUNTIME_CONFIG_KEY, {})
    service_discovery_config = get_service_discovery_config(runtime_config)
    service_name = get_canonical_service_name(
        service_discovery_config, cluster_name,
        CLOUDTIK_CLUSTER_CONTROLLER_METRICS_SERVICE_NAME)
    redis_service_name = get_canonical_service_name(
        service_discovery_config, cluster_name,
        CLOUDTIK_REDIS_SERVICE_NAME)
    services = {
        service_name: define_runtime_service_on_head(
            service_discovery_config, CLOUDTIK_CLUSTER_CONTROLLER_METRICS_PORT,
            metrics=True),
        redis_service_name: define_runtime_service_on_head(
            service_discovery_config, CLOUDTIK_REDIS_SERVICE_PORT),
    }
    return services
