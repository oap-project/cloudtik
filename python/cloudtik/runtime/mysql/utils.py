import os
from typing import Any, Dict

from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_MYSQL
from cloudtik.core._private.service_discovery.utils import \
    get_canonical_service_name, define_runtime_service, \
    get_service_discovery_config
from cloudtik.core._private.utils import RUNTIME_CONFIG_KEY

RUNTIME_PROCESSES = [
        # The first element is the substring to filter.
        # The second element, if True, is to filter ps results by command name.
        # The third element is the process name.
        # The forth element, if node, the process should on all nodes,if head, the process should on head node.
        ["mysqld", True, "MySQL", "node"],
    ]

MYSQL_SERVICE_PORT_CONFIG_KEY = "port"

MYSQL_HIGH_AVAILABILITY_CONFIG_KEY = "high_availability"
MYSQL_ROOT_PASSWORD_CONFIG_KEY = "root_password"

MYSQL_DATABASE_CONFIG_KEY = "database"
MYSQL_DATABASE_NAME_CONFIG_KEY = "name"
MYSQL_DATABASE_USER_CONFIG_KEY = "user"
MYSQL_DATABASE_PASSWORD_CONFIG_KEY = "password"

MYSQL_SERVICE_NAME = BUILT_IN_RUNTIME_MYSQL
MYSQL_SERVICE_PORT_DEFAULT = 3306

MYSQL_ROOT_PASSWORD_DEFAULT = "cloudtik"


def _get_config(runtime_config: Dict[str, Any]):
    return runtime_config.get(BUILT_IN_RUNTIME_MYSQL, {})


def _get_service_port(mysql_config: Dict[str, Any]):
    return mysql_config.get(
        MYSQL_SERVICE_PORT_CONFIG_KEY, MYSQL_SERVICE_PORT_DEFAULT)


def _get_home_dir():
    return os.path.join(
        os.getenv("HOME"), "runtime", BUILT_IN_RUNTIME_MYSQL)


def _get_runtime_processes():
    return RUNTIME_PROCESSES


def _get_runtime_logs():
    home_dir = _get_home_dir()
    logs_dir = os.path.join(home_dir, "logs")
    return {"mysql": logs_dir}


def _validate_config(config: Dict[str, Any]):
    runtime_config = config.get(RUNTIME_CONFIG_KEY)
    mysql_config = _get_config(runtime_config)

    database = mysql_config.get(MYSQL_DATABASE_CONFIG_KEY, {})
    user = database.get(MYSQL_DATABASE_USER_CONFIG_KEY)
    password = database.get(MYSQL_DATABASE_PASSWORD_CONFIG_KEY)
    if (user and not password) or (not user and password):
        raise ValueError("User and password must be both specified or not specified.")


def _with_runtime_environment_variables(
        runtime_config, config):
    runtime_envs = {}

    mysql_config = _get_config(runtime_config)

    service_port = _get_service_port(mysql_config)
    runtime_envs["MYSQL_SERVICE_PORT"] = service_port

    root_password = mysql_config.get(
        MYSQL_ROOT_PASSWORD_CONFIG_KEY, MYSQL_ROOT_PASSWORD_DEFAULT)
    runtime_envs["MYSQL_ROOT_PASSWORD"] = root_password

    database = mysql_config.get(MYSQL_DATABASE_CONFIG_KEY, {})
    database_name = database.get(MYSQL_DATABASE_NAME_CONFIG_KEY)
    if database_name:
        runtime_envs["MYSQL_DATABASE"] = database_name
    user = database.get(MYSQL_DATABASE_USER_CONFIG_KEY)
    if user:
        runtime_envs["MYSQL_USER"] = user
    password = database.get(MYSQL_DATABASE_PASSWORD_CONFIG_KEY)
    if password:
        runtime_envs["MYSQL_PASSWORD"] = password

    return runtime_envs


def _get_runtime_endpoints(runtime_config: Dict[str, Any], cluster_head_ip):
    service_port = _get_service_port(runtime_config)
    endpoints = {
        "mysql": {
            "name": "MySQL",
            "url": "{}:{}".format(cluster_head_ip, service_port)
        },
    }
    return endpoints


def _get_head_service_ports(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    service_port = _get_service_port(runtime_config)
    service_ports = {
        "mysql": {
            "protocol": "TCP",
            "port": service_port,
        },
    }
    return service_ports


def _get_runtime_services(
        runtime_config: Dict[str, Any], cluster_name: str) -> Dict[str, Any]:
    mysql_config = _get_config(runtime_config)
    service_discovery_config = get_service_discovery_config(mysql_config)
    service_name = get_canonical_service_name(
        service_discovery_config, cluster_name, MYSQL_SERVICE_NAME)
    service_port = _get_service_port(mysql_config)
    services = {
        service_name: define_runtime_service(
            service_discovery_config, service_port),
    }
    return services
