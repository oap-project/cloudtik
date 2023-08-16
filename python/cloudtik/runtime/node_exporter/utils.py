import os
from typing import Any, Dict

from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_NODE_EXPORTER
from cloudtik.core._private.service_discovery.utils import \
    get_canonical_service_name, define_runtime_service, \
    get_service_discovery_config, SERVICE_DISCOVERY_PROTOCOL_HTTP

RUNTIME_PROCESSES = [
        # The first element is the substring to filter.
        # The second element, if True, is to filter ps results by command name.
        # The third element is the process name.
        # The forth element, if node, the process should on all nodes,if head, the process should on head node.
        ["node_exporter", True, "Node Exporter", "node"],
    ]

NODE_EXPORTER_SERVICE_PORT_CONFIG_KEY = "port"

NODE_EXPORTER_SERVICE_NAME = "node-exporter"
NODE_EXPORTER_SERVICE_PORT_DEFAULT = 9100


def _get_config(runtime_config: Dict[str, Any]):
    return runtime_config.get(BUILT_IN_RUNTIME_NODE_EXPORTER, {})


def _get_service_port(node_exporter_config: Dict[str, Any]):
    return node_exporter_config.get(
        NODE_EXPORTER_SERVICE_PORT_CONFIG_KEY, NODE_EXPORTER_SERVICE_PORT_DEFAULT)


def _get_home_dir():
    return os.path.join(
        os.getenv("HOME"), "runtime", BUILT_IN_RUNTIME_NODE_EXPORTER)


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _get_runtime_logs():
    home_dir = _get_home_dir()
    logs_dir = os.path.join(home_dir, "logs")
    return {"node_exporter": logs_dir}


def _with_runtime_environment_variables(
        runtime_config, config):
    runtime_envs = {}

    node_exporter_config = _get_config(runtime_config)

    service_port = _get_service_port(node_exporter_config)
    runtime_envs["NODE_EXPORTER_SERVICE_PORT"] = service_port

    return runtime_envs


def _get_runtime_services(
        runtime_config: Dict[str, Any], cluster_name: str) -> Dict[str, Any]:
    node_exporter_config = _get_config(runtime_config)
    service_discovery_config = get_service_discovery_config(node_exporter_config)
    service_name = get_canonical_service_name(
        service_discovery_config, cluster_name, NODE_EXPORTER_SERVICE_NAME)
    service_port = _get_service_port(node_exporter_config)
    services = {
        service_name: define_runtime_service(
            service_discovery_config, service_port,
            protocol=SERVICE_DISCOVERY_PROTOCOL_HTTP,
            metrics=True),
    }
    return services
