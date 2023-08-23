import os
from typing import Any, Dict

from cloudtik.core._private.providers import _get_node_provider
from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_METASTORE
from cloudtik.core._private.service_discovery.utils import get_canonical_service_name, define_runtime_service_on_head, \
    get_service_discovery_config
from cloudtik.core._private.util.database_utils import is_database_configured, \
    export_database_environment_variables
from cloudtik.core._private.utils import export_runtime_flags
from cloudtik.runtime.common.service_discovery.runtime_discovery import \
    discover_database_from_workspace, discover_database_on_head, DATABASE_CONNECT_KEY, get_database_runtime_in_cluster, \
    export_database_runtime_environment_variables
from cloudtik.runtime.common.service_discovery.workspace import register_service_to_workspace

RUNTIME_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name.
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["proc_metastore", False, "Metastore", "head"],
    ["mysql", False, "MySQL", "head"],
]

METASTORE_SERVICE_NAME = BUILT_IN_RUNTIME_METASTORE
METASTORE_SERVICE_PORT = 9083


def _get_config(runtime_config: Dict[str, Any]):
    return runtime_config.get(BUILT_IN_RUNTIME_METASTORE, {})


def _get_database_config(metastore_config):
    return metastore_config.get(DATABASE_CONNECT_KEY, {})


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _config_depended_services(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    cluster_config = discover_database_from_workspace(
        cluster_config, BUILT_IN_RUNTIME_METASTORE)
    return cluster_config


def _prepare_config_on_head(cluster_config: Dict[str, Any]):
    cluster_config = discover_database_on_head(
        cluster_config, BUILT_IN_RUNTIME_METASTORE)
    return cluster_config


def _with_runtime_environment_variables(
        runtime_config, config, provider, node_id: str):
    runtime_envs = {"METASTORE_ENABLED": True}

    metastore_config = _get_config(runtime_config)
    export_runtime_flags(
        metastore_config, BUILT_IN_RUNTIME_METASTORE, runtime_envs)
    return runtime_envs


def _export_database_configurations(runtime_config):
    metastore_config = _get_config(runtime_config)
    database_config = _get_database_config(metastore_config)
    if is_database_configured(database_config):
        # set the database environments from database config
        # This may override the environments from provider
        export_database_environment_variables(database_config)
    else:
        database_runtime = get_database_runtime_in_cluster(
            runtime_config)
        if database_runtime:
            export_database_runtime_environment_variables(
                runtime_config, database_runtime)


def _configure(runtime_config, head: bool):
    _export_database_configurations(runtime_config)


def _services(runtime_config, command: str, head: bool):
    if command == "start":
        # We put the database schema init right before the start of metastore service
        _export_database_configurations(runtime_config)


def register_service(cluster_config: Dict[str, Any], head_node_id: str) -> None:
    provider = _get_node_provider(
        cluster_config["provider"], cluster_config["cluster_name"])
    head_ip = provider.internal_ip(head_node_id)
    register_service_to_workspace(
        cluster_config, BUILT_IN_RUNTIME_METASTORE,
        service_addresses=[(head_ip, METASTORE_SERVICE_PORT)])


def _get_runtime_logs():
    hive_logs_dir = os.path.join(os.getenv("METASTORE_HOME"), "logs")
    all_logs = {"metastore": hive_logs_dir}
    return all_logs


def _get_runtime_endpoints(cluster_head_ip):
    endpoints = {
        "metastore": {
            "name": "Metastore Uri",
            "url": "thrift://{}:{}".format(cluster_head_ip, METASTORE_SERVICE_PORT)
        },
    }
    return endpoints


def _get_head_service_ports(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    service_ports = {
        "metastore": {
            "protocol": "TCP",
            "port": METASTORE_SERVICE_PORT,
        },
    }
    return service_ports


def _get_runtime_services(
        runtime_config: Dict[str, Any], cluster_name: str) -> Dict[str, Any]:
    metastore_config = _get_config(runtime_config)
    service_discovery_config = get_service_discovery_config(metastore_config)
    service_name = get_canonical_service_name(
        service_discovery_config, cluster_name, METASTORE_SERVICE_NAME)
    services = {
        service_name: define_runtime_service_on_head(
            service_discovery_config, METASTORE_SERVICE_PORT),
    }
    return services
