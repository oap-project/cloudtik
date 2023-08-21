import os
from typing import Any, Dict

from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_ZOOKEEPER, BUILT_IN_RUNTIME_KAFKA
from cloudtik.core._private.runtime_utils import subscribe_runtime_config
from cloudtik.core._private.service_discovery.runtime_services import get_service_discovery_runtime
from cloudtik.core._private.service_discovery.utils import get_canonical_service_name, define_runtime_service_on_worker, \
    get_service_discovery_config, SERVICE_DISCOVERY_FEATURE_MESSAGING
from cloudtik.core._private.utils import \
    RUNTIME_CONFIG_KEY, load_properties_file, \
    save_properties_file, get_config_for_update, get_runtime_config
from cloudtik.runtime.common.service_discovery.cluster import query_service_from_cluster, get_service_addresses_string, \
    has_runtime_in_cluster
from cloudtik.runtime.common.service_discovery.discovery import DiscoveryType
from cloudtik.runtime.common.service_discovery.runtime_discovery import discover_zookeeper

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["kafka.Kafka", False, "KafkaBroker", "node"],
]

KAFKA_ZOOKEEPER_CONNECT_KEY = "zookeeper_connect"
KAFKA_ZOOKEEPER_SERVICE_SELECTOR_KEY = "zookeeper_service_selector"

KAFKA_SERVICE_NAME = BUILT_IN_RUNTIME_KAFKA
KAFKA_SERVICE_PORT = 9092


def _get_config(runtime_config: Dict[str, Any]):
    return runtime_config.get(BUILT_IN_RUNTIME_KAFKA, {})


def _config_depended_services(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    runtime_config = get_config_for_update(cluster_config, RUNTIME_CONFIG_KEY)
    kafka_config = get_config_for_update(runtime_config, BUILT_IN_RUNTIME_KAFKA)

    # Check zookeeper
    if not has_runtime_in_cluster(runtime_config, BUILT_IN_RUNTIME_ZOOKEEPER):
        if kafka_config.get(KAFKA_ZOOKEEPER_CONNECT_KEY) is None:
            if kafka_config.get("zookeeper_service_discovery", True):
                zookeeper_uri = discover_zookeeper(
                    kafka_config, KAFKA_ZOOKEEPER_SERVICE_SELECTOR_KEY,
                    cluster_config=cluster_config,
                    discovery_type=DiscoveryType.WORKSPACE)
                if zookeeper_uri is not None:
                    kafka_config[KAFKA_ZOOKEEPER_CONNECT_KEY] = zookeeper_uri

    return cluster_config


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _with_runtime_environment_variables(runtime_config, config, provider, node_id: str):
    runtime_envs = {"KAFKA_ENABLED": True}
    return runtime_envs


def _get_runtime_logs():
    kafka_logs_dir = os.path.join(os.getenv("KAFKA_HOME"), "logs")
    all_logs = {"kafka": kafka_logs_dir}
    return all_logs


def _validate_config(config: Dict[str, Any]):
    # Check zookeeper connect configured
    runtime_config = get_runtime_config(config)
    kafka_config = _get_config(runtime_config)
    zookeeper_uri = kafka_config.get(KAFKA_ZOOKEEPER_CONNECT_KEY)
    if not zookeeper_uri and not has_runtime_in_cluster(
            runtime_config, BUILT_IN_RUNTIME_ZOOKEEPER):

        # if there is service discovery mechanism, assume we can get from service discovery
        if not get_service_discovery_runtime(runtime_config):
            raise ValueError("Zookeeper must be configured for Kafka.")


def _configure_on_head(config: Dict[str, Any]):
    runtime_config = get_runtime_config(config)
    kafka_config = _get_config(runtime_config)
    zookeeper_uri = kafka_config.get(KAFKA_ZOOKEEPER_CONNECT_KEY)
    if zookeeper_uri:
        # Zookeeper already configured
        return config

    if has_runtime_in_cluster(
            runtime_config, BUILT_IN_RUNTIME_ZOOKEEPER):
        # There is a local Zookeeper
        return config

    # There is service discovery to come here
    zookeeper_uri = discover_zookeeper(
                    kafka_config, KAFKA_ZOOKEEPER_SERVICE_SELECTOR_KEY,
                    cluster_config=config,
                    discovery_type=DiscoveryType.CLUSTER)
    if not zookeeper_uri:
        raise RuntimeError(
            "No Zookeeper service is discovered. "
            "You can either configure manually or start a discoverable Zookeeper service.")

    kafka_config = get_config_for_update(
        runtime_config, BUILT_IN_RUNTIME_KAFKA)
    kafka_config[KAFKA_ZOOKEEPER_CONNECT_KEY] = zookeeper_uri
    return config


def _get_runtime_endpoints(cluster_head_ip):
    # TODO: how to get the Kafka service address which established after head node
    return None


def _get_zookeeper_connect(runtime_config):
    if runtime_config is None:
        return None

    kafka_config = _get_config(runtime_config)
    # check config
    zookeeper_connect = kafka_config.get(KAFKA_ZOOKEEPER_CONNECT_KEY)
    if zookeeper_connect is not None:
        return zookeeper_connect

    # check redis endpoint publish
    service_addresses = query_service_from_cluster(
        runtime_type=BUILT_IN_RUNTIME_ZOOKEEPER)
    if not service_addresses:
        return None

    zookeeper_connect = get_service_addresses_string(
        service_addresses)
    return zookeeper_connect


def _get_server_config(runtime_config: Dict[str, Any]):
    kafka_config = _get_config(runtime_config)
    return kafka_config.get("config")


def update_configurations():
    # Merge user specified configuration and default configuration
    runtime_config = subscribe_runtime_config()
    server_config = _get_server_config(runtime_config)
    if not server_config:
        return

    server_properties_file = os.path.join(os.getenv("KAFKA_HOME"), "config/server.properties")

    # Read in the existing configurations
    server_properties, comments = load_properties_file(server_properties_file)

    # Merge with the user configurations
    server_properties.update(server_config)

    # Write back the configuration file
    save_properties_file(server_properties_file, server_properties, comments=comments)


def _get_runtime_services(
        runtime_config: Dict[str, Any], cluster_name: str) -> Dict[str, Any]:
    kafka_config = _get_config(runtime_config)
    service_discovery_config = get_service_discovery_config(kafka_config)
    service_name = get_canonical_service_name(
        service_discovery_config, cluster_name, KAFKA_SERVICE_NAME)
    services = {
        service_name: define_runtime_service_on_worker(
            service_discovery_config, KAFKA_SERVICE_PORT,
            features=[SERVICE_DISCOVERY_FEATURE_MESSAGING]),
    }
    return services
