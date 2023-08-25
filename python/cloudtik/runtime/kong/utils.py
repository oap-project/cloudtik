import os
from typing import Any, Dict

from cloudtik.core._private.core_utils import get_address_string
from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_KONG, BUILT_IN_RUNTIME_POSTGRES
from cloudtik.core._private.runtime_utils import get_runtime_bool, \
    get_runtime_value
from cloudtik.core._private.service_discovery.runtime_services import get_service_discovery_runtime
from cloudtik.core._private.service_discovery.utils import \
    get_canonical_service_name, define_runtime_service, \
    get_service_discovery_config, SERVICE_DISCOVERY_FEATURE_API_GATEWAY, SERVICE_DISCOVERY_PROTOCOL_HTTP
from cloudtik.core._private.util.database_utils import is_database_configured, export_database_environment_variables, \
    DATABASE_ENGINE_POSTGRES, get_database_engine, DATABASE_ENV_ENABLED, DATABASE_ENV_ENGINE
from cloudtik.core._private.utils import get_runtime_config, is_use_managed_cloud_database, PROVIDER_DATABASE_CONFIG_KEY
from cloudtik.runtime.common.service_discovery.runtime_discovery import \
    DATABASE_CONNECT_KEY, is_database_service_discovery, discover_database_on_head, \
    discover_database_from_workspace

RUNTIME_PROCESSES = [
        # The first element is the substring to filter.
        # The second element, if True, is to filter ps results by command name.
        # The third element is the process name.
        # The forth element, if node, the process should on all nodes,if head, the process should on head node.
        ["/usr/local/kong", False, "KONG", "node"],
    ]

KONG_SERVICE_PORT_CONFIG_KEY = "port"
KONG_SERVICE_SSL_PORT_CONFIG_KEY = "ssl_port"

KONG_SERVICE_NAME = BUILT_IN_RUNTIME_KONG

KONG_SERVICE_PORT_DEFAULT = 8000
KONG_SERVICE_SSL_PORT_DEFAULT = 8443

KONG_ADMIN_PORT_DEFAULT = 8001
KONG_ADMIN_SSL_PORT_DEFAULT = 8444
KONG_ADMIN_UI_PORT_DEFAULT = 8002
KONG_ADMIN_UI_SSL_PORT_DEFAULT = 8445


def _get_config(runtime_config: Dict[str, Any]):
    return runtime_config.get(BUILT_IN_RUNTIME_KONG, {})


def _get_database_config(metastore_config):
    return metastore_config.get(DATABASE_CONNECT_KEY, {})


def _get_service_port(kong_config: Dict[str, Any]):
    return kong_config.get(
        KONG_SERVICE_PORT_CONFIG_KEY, KONG_SERVICE_PORT_DEFAULT)


def _get_service_ssl_port(kong_config: Dict[str, Any]):
    return kong_config.get(
        KONG_SERVICE_SSL_PORT_CONFIG_KEY, KONG_SERVICE_SSL_PORT_DEFAULT)


def _get_home_dir():
    return os.path.join(
        os.getenv("HOME"), "runtime", BUILT_IN_RUNTIME_KONG)


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _get_runtime_logs():
    home_dir = _get_home_dir()
    logs_dir = os.path.join(home_dir, "logs")
    return {BUILT_IN_RUNTIME_KONG: logs_dir}


def _config_depended_services(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    cluster_config = discover_database_from_workspace(
        cluster_config, BUILT_IN_RUNTIME_KONG,
        database_runtime_type=BUILT_IN_RUNTIME_POSTGRES,
        allow_local=False
    )
    return cluster_config


def _prepare_config_on_head(cluster_config: Dict[str, Any]):
    cluster_config = discover_database_on_head(
        cluster_config, BUILT_IN_RUNTIME_KONG,
        database_runtime_type=BUILT_IN_RUNTIME_POSTGRES,
        allow_local=False)

    _validate_config(cluster_config, final=True)
    return cluster_config


def _is_valid_database_config(config: Dict[str, Any], final=False):
    # Check database configuration
    runtime_config = get_runtime_config(config)
    kong_config = _get_config(runtime_config)
    database_config = _get_database_config(kong_config)
    if is_database_configured(database_config):
        if get_database_engine(database_config) != DATABASE_ENGINE_POSTGRES:
            return False
        return True

    # check whether cloud database is available (must be postgres)
    provider_config = config["provider"]
    if (PROVIDER_DATABASE_CONFIG_KEY in provider_config or
            (not final and is_use_managed_cloud_database(config))):
        return True

    # if there is service discovery mechanism, assume we can get from service discovery
    if (not final and is_database_service_discovery(kong_config)
            and get_service_discovery_runtime(runtime_config)):
        return True

    return False


def _validate_config(config: Dict[str, Any], final=False):
    if not _is_valid_database_config(config, final):
        raise ValueError("Postgres must be configured for Kong.")


def _with_runtime_environment_variables(
        runtime_config, config):
    runtime_envs = {}
    kong_config = _get_config(runtime_config)

    service_port = _get_service_port(kong_config)
    runtime_envs["KONG_SERVICE_PORT"] = service_port
    service_ssl_port = _get_service_ssl_port(kong_config)
    runtime_envs["KONG_SERVICE_SSL_PORT"] = service_ssl_port
    runtime_envs["KONG_ADMIN_PORT"] = KONG_ADMIN_PORT_DEFAULT
    runtime_envs["KONG_ADMIN_SSL_PORT"] = KONG_ADMIN_SSL_PORT_DEFAULT
    runtime_envs["KONG_ADMIN_UI_PORT"] = KONG_ADMIN_UI_PORT_DEFAULT
    runtime_envs["KONG_ADMIN_UI_SSL_PORT"] = KONG_ADMIN_UI_SSL_PORT_DEFAULT

    return runtime_envs


def _export_database_configurations(runtime_config):
    kong_config = _get_config(runtime_config)
    database_config = _get_database_config(kong_config)
    if is_database_configured(database_config):
        # set the database environments from database config
        # This may override the environments from provider
        export_database_environment_variables(database_config)
    else:
        # check cloud database is configured
        database_enabled = get_runtime_bool(DATABASE_ENV_ENABLED)
        if not database_enabled:
            raise RuntimeError("No Postgres is configured for Kong.")
        database_engine = get_runtime_value(DATABASE_ENV_ENGINE)
        if database_engine != DATABASE_ENGINE_POSTGRES:
            raise RuntimeError("Postgres must be configured for Kong.")


def _configure(runtime_config, head: bool):
    _export_database_configurations(runtime_config)


def _services(runtime_config, head: bool):
    # We put the database schema init right before the start of metastore service
    _export_database_configurations(runtime_config)


def _get_runtime_endpoints(
        runtime_config: Dict[str, Any], cluster_head_ip):
    service_port = _get_service_port(runtime_config)
    endpoints = {
        "kong": {
            "name": "KONG",
            "url": "http://{}".format(
                get_address_string(cluster_head_ip, service_port))
        },
    }
    return endpoints


def _get_head_service_ports(
        runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    service_port = _get_service_port(runtime_config)
    service_ports = {
        "kong": {
            "protocol": "TCP",
            "port": service_port,
        },
    }
    return service_ports


def _get_runtime_services(
        runtime_config: Dict[str, Any], cluster_name: str) -> Dict[str, Any]:
    kong_config = _get_config(runtime_config)
    service_discovery_config = get_service_discovery_config(kong_config)
    service_name = get_canonical_service_name(
        service_discovery_config, cluster_name, KONG_SERVICE_NAME)
    service_port = _get_service_port(kong_config)
    services = {
        service_name: define_runtime_service(
            service_discovery_config, service_port,
            protocol=SERVICE_DISCOVERY_PROTOCOL_HTTP,
            features=[SERVICE_DISCOVERY_FEATURE_API_GATEWAY]),
    }
    return services
