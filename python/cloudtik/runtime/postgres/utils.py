import os
from typing import Any, Dict

from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_POSTGRES
from cloudtik.core._private.service_discovery.utils import \
    get_canonical_service_name, define_runtime_service, \
    get_service_discovery_config, SERVICE_DISCOVERY_FEATURE_DATABASE
from cloudtik.core._private.utils import RUNTIME_CONFIG_KEY

RUNTIME_PROCESSES = [
        # The first element is the substring to filter.
        # The second element, if True, is to filter ps results by command name.
        # The third element is the process name.
        # The forth element, if node, the process should on all nodes,if head, the process should on head node.
        ["postgres", True, "Postgres", "node"],
    ]

POSTGRES_SERVICE_PORT_CONFIG_KEY = "port"

POSTGRES_HIGH_AVAILABILITY_CONFIG_KEY = "high_availability"
POSTGRES_ADMIN_USER_CONFIG_KEY = "admin_user"
POSTGRES_ADMIN_PASSWORD_CONFIG_KEY = "admin_password"

POSTGRES_DATABASE_CONFIG_KEY = "database"
POSTGRES_DATABASE_NAME_CONFIG_KEY = "name"
POSTGRES_DATABASE_USER_CONFIG_KEY = "user"
POSTGRES_DATABASE_PASSWORD_CONFIG_KEY = "password"

POSTGRES_SERVICE_NAME = BUILT_IN_RUNTIME_POSTGRES
POSTGRES_SERVICE_PORT_DEFAULT = 5432

POSTGRES_ADMIN_USER_DEFAULT = "cloudtik"
POSTGRES_ADMIN_PASSWORD_DEFAULT = "cloudtik"


def _get_config(runtime_config: Dict[str, Any]):
    return runtime_config.get(BUILT_IN_RUNTIME_POSTGRES, {})


def _get_service_port(postgres_config: Dict[str, Any]):
    return postgres_config.get(
        POSTGRES_SERVICE_PORT_CONFIG_KEY, POSTGRES_SERVICE_PORT_DEFAULT)


def _get_home_dir():
    return os.path.join(
        os.getenv("HOME"), "runtime", BUILT_IN_RUNTIME_POSTGRES)


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _get_runtime_logs():
    home_dir = _get_home_dir()
    logs_dir = os.path.join(home_dir, "logs")
    return {"postgres": logs_dir}


def _validate_config(config: Dict[str, Any]):
    runtime_config = config.get(RUNTIME_CONFIG_KEY)
    mysql_config = _get_config(runtime_config)

    database = mysql_config.get(POSTGRES_DATABASE_CONFIG_KEY, {})
    user = database.get(POSTGRES_DATABASE_USER_CONFIG_KEY)
    password = database.get(POSTGRES_DATABASE_PASSWORD_CONFIG_KEY)
    if (user and not password) or (not user and password):
        raise ValueError("Database user and password must be both specified or not specified.")


def _with_runtime_environment_variables(
        runtime_config, config):
    runtime_envs = {}

    postgres_config = _get_config(runtime_config)

    service_port = _get_service_port(postgres_config)
    runtime_envs["POSTGRES_SERVICE_PORT"] = service_port

    admin_user = postgres_config.get(
        POSTGRES_ADMIN_USER_CONFIG_KEY, POSTGRES_ADMIN_USER_DEFAULT)
    runtime_envs["POSTGRES_USER"] = admin_user

    admin_password = postgres_config.get(
        POSTGRES_ADMIN_PASSWORD_CONFIG_KEY, POSTGRES_ADMIN_PASSWORD_DEFAULT)
    runtime_envs["POSTGRES_PASSWORD"] = admin_password

    database = postgres_config.get(POSTGRES_DATABASE_CONFIG_KEY, {})
    database_name = database.get(POSTGRES_DATABASE_NAME_CONFIG_KEY)
    if database_name:
        runtime_envs["POSTGRES_DATABASE_NAME"] = database_name
    user = database.get(POSTGRES_DATABASE_USER_CONFIG_KEY)
    if user:
        runtime_envs["POSTGRES_DATABASE_USER"] = user
    password = database.get(POSTGRES_DATABASE_PASSWORD_CONFIG_KEY)
    if password:
        runtime_envs["POSTGRES_DATABASE_PASSWORD"] = password

    return runtime_envs


def _get_runtime_endpoints(runtime_config: Dict[str, Any], cluster_head_ip):
    service_port = _get_service_port(runtime_config)
    endpoints = {
        "postgres": {
            "name": "Postgres",
            "url": "{}:{}".format(cluster_head_ip, service_port)
        },
    }
    return endpoints


def _get_head_service_ports(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    service_port = _get_service_port(runtime_config)
    service_ports = {
        "postgres": {
            "protocol": "TCP",
            "port": service_port,
        },
    }
    return service_ports


def _get_runtime_services(
        runtime_config: Dict[str, Any], cluster_name: str) -> Dict[str, Any]:
    postgres_config = _get_config(runtime_config)
    service_discovery_config = get_service_discovery_config(postgres_config)
    service_name = get_canonical_service_name(
        service_discovery_config, cluster_name, POSTGRES_SERVICE_NAME)
    service_port = _get_service_port(postgres_config)
    services = {
        service_name: define_runtime_service(
            service_discovery_config, service_port,
            features=[SERVICE_DISCOVERY_FEATURE_DATABASE]),
    }
    return services
