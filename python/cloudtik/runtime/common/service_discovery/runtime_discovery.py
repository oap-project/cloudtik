import os
from typing import Dict, Any, Union, List

from cloudtik.core._private.constants import CLOUDTIK_RUNTIME_ENV_HEAD_IP
from cloudtik.core._private.core_utils import get_config_for_update, get_env_string_value
from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_HDFS, BUILT_IN_RUNTIME_METASTORE, \
    BUILT_IN_RUNTIME_CONSUL, BUILT_IN_RUNTIME_ZOOKEEPER, BUILT_IN_RUNTIME_MYSQL, BUILT_IN_RUNTIME_POSTGRES, \
    BUILT_IN_RUNTIME_ETCD
from cloudtik.core._private.runtime_utils import get_runtime_value
from cloudtik.core._private.service_discovery.utils import get_service_selector_for_update, \
    include_runtime_for_selector, include_feature_for_selector
from cloudtik.core._private.util.database_utils import is_database_configured, set_database_config, \
    DATABASE_ENV_ENABLED, DATABASE_ENV_ENGINE, DATABASE_ENV_HOST, DATABASE_ENV_PORT, DATABASE_ENV_USERNAME, \
    DATABASE_ENV_PASSWORD, get_database_default_port, get_database_default_username, get_database_default_password, \
    DATABASE_ENGINE_MYSQL
from cloudtik.core._private.utils import get_runtime_config
from cloudtik.runtime.common.service_discovery.cluster import has_runtime_in_cluster
from cloudtik.runtime.common.service_discovery.discovery import query_one_service, DiscoveryType
from cloudtik.runtime.common.service_discovery.utils import get_service_addresses_string

BUILT_IN_DATABASE_RUNTIMES = [BUILT_IN_RUNTIME_MYSQL, BUILT_IN_RUNTIME_POSTGRES]
DATABASE_SERVICE_PORT_CONFIG_KEY = "port"
MYSQL_ROOT_PASSWORD_CONFIG_KEY = "root_password"
POSTGRES_ADMIN_USER_CONFIG_KEY = "admin_user"
POSTGRES_ADMIN_PASSWORD_CONFIG_KEY = "admin_password"

HDFS_URI_KEY = "hdfs_namenode_uri"
HDFS_SERVICE_DISCOVERY_KEY = "hdfs_service_discovery"
HDFS_SERVICE_SELECTOR_KEY = "hdfs_service_selector"

METASTORE_URI_KEY = "hive_metastore_uri"
METASTORE_SERVICE_DISCOVERY_KEY = "metastore_service_discovery"
METASTORE_SERVICE_SELECTOR_KEY = "metastore_service_selector"

ZOOKEEPER_CONNECT_KEY = "zookeeper_connect"
ZOOKEEPER_SERVICE_DISCOVERY_KEY = "zookeeper_service_discovery"
ZOOKEEPER_SERVICE_SELECTOR_KEY = "zookeeper_service_selector"

DATABASE_CONNECT_KEY = "database"
DATABASE_SERVICE_DISCOVERY_KEY = "database_service_discovery"
DATABASE_SERVICE_SELECTOR_KEY = "database_service_selector"

ETCD_URI_KEY = "etcd_uri"
ETCD_SERVICE_DISCOVERY_KEY = "etcd_service_discovery"
ETCD_SERVICE_SELECTOR_KEY = "etcd_service_selector"


def get_database_runtime_in_cluster(runtime_config):
    for runtime_type in BUILT_IN_DATABASE_RUNTIMES:
        if has_runtime_in_cluster(runtime_config, runtime_type):
            return runtime_type
    return None


def get_database_engine_for_runtime(runtime_type):
    # The engine and runtime type is the same
    return runtime_type


def discover_runtime_service(
        config: Dict[str, Any],
        service_selector_key: str,
        runtime_type: Union[str, List[str]],
        cluster_config: Dict[str, Any],
        discovery_type: DiscoveryType,):
    service_selector = get_service_selector_for_update(
        config, service_selector_key)
    # if user provide runtimes in the selector, we don't override it
    # because any runtimes in the list will be selected
    service_selector = include_runtime_for_selector(
        service_selector, runtime_type)
    service_instance = query_one_service(
        cluster_config, service_selector,
        discovery_type=discovery_type)
    return service_instance


def discover_runtime_service_addresses(
        config: Dict[str, Any],
        service_selector_key: str,
        runtime_type: Union[str, List[str]],
        cluster_config: Dict[str, Any],
        discovery_type: DiscoveryType,):
    service_instance = discover_runtime_service(
        config, service_selector_key,
        runtime_type, cluster_config, discovery_type)
    if not service_instance:
        return None
    return service_instance.service_addresses


def discover_runtime_service_addresses_by_feature(
        config: Dict[str, Any],
        service_selector_key: str,
        feature: str,
        cluster_config: Dict[str, Any],
        discovery_type: DiscoveryType,):
    service_selector = get_service_selector_for_update(
        config, service_selector_key)
    # WARNING: feature selecting doesn't work for workspace service registry
    service_selector = include_feature_for_selector(
        service_selector, feature)
    service_instance = query_one_service(
        cluster_config, service_selector,
        discovery_type=discovery_type)
    if not service_instance:
        return None
    return service_instance.service_addresses


def discover_consul(
        config: Dict[str, Any],
        service_selector_key: str,
        cluster_config: Dict[str, Any],
        discovery_type: DiscoveryType,):
    return discover_runtime_service_addresses(
        config, service_selector_key,
        runtime_type=BUILT_IN_RUNTIME_CONSUL,
        cluster_config=cluster_config,
        discovery_type=discovery_type,
    )


def discover_zookeeper(
        config: Dict[str, Any],
        service_selector_key: str,
        cluster_config: Dict[str, Any],
        discovery_type: DiscoveryType,):
    service_addresses = discover_runtime_service_addresses(
        config, service_selector_key,
        runtime_type=BUILT_IN_RUNTIME_ZOOKEEPER,
        cluster_config=cluster_config,
        discovery_type=discovery_type,
    )
    if not service_addresses:
        return None
    return get_service_addresses_string(service_addresses)


def discover_hdfs(
        config: Dict[str, Any],
        service_selector_key: str,
        cluster_config: Dict[str, Any],
        discovery_type: DiscoveryType,):
    service_addresses = discover_runtime_service_addresses(
        config, service_selector_key,
        runtime_type=BUILT_IN_RUNTIME_HDFS,
        cluster_config=cluster_config,
        discovery_type=discovery_type,
    )
    if not service_addresses:
        return None
    service_address = service_addresses[0]
    hdfs_uri = "hdfs://{}:{}".format(
        service_address[0], service_address[1])
    return hdfs_uri


def discover_metastore(
        config: Dict[str, Any],
        service_selector_key: str,
        cluster_config: Dict[str, Any],
        discovery_type: DiscoveryType,):
    service_addresses = discover_runtime_service_addresses(
        config, service_selector_key,
        runtime_type=BUILT_IN_RUNTIME_METASTORE,
        cluster_config=cluster_config,
        discovery_type=discovery_type,
    )
    if not service_addresses:
        return None
    # take one of them
    service_address = service_addresses[0]
    metastore_uri = "thrift://{}:{}".format(
        service_address[0], service_address[1])
    return metastore_uri


def discover_database(
        config: Dict[str, Any],
        service_selector_key: str,
        cluster_config: Dict[str, Any],
        discovery_type: DiscoveryType,
        database_runtime_type=None):
    # TODO: because feature tag is not supported for workspace based discovery
    #   Use a list of database runtimes here.
    if not database_runtime_type:
        # if no specific database type, default to all known types
        database_runtime_type = BUILT_IN_DATABASE_RUNTIMES
    service_instance = discover_runtime_service(
        config, service_selector_key,
        runtime_type=database_runtime_type,
        cluster_config=cluster_config,
        discovery_type=discovery_type,
    )
    if not service_instance:
        return None
    engine = get_database_engine_for_runtime(
        service_instance.runtime_type)
    return engine, service_instance.service_addresses


def discover_etcd(
        config: Dict[str, Any],
        service_selector_key: str,
        cluster_config: Dict[str, Any],
        discovery_type: DiscoveryType,):
    service_addresses = discover_runtime_service_addresses(
        config, service_selector_key,
        runtime_type=BUILT_IN_RUNTIME_ETCD,
        cluster_config=cluster_config,
        discovery_type=discovery_type,
    )
    if not service_addresses:
        return None
    return get_service_addresses_string(service_addresses)


"""
Common help functions for runtime service discovery used by runtimes.
To use these common functions, some conventions are used.
The conventions to follow for each function are explained per function.
"""


"""
HDFS service discovery conventions:
    1. The hdfs service discovery flag is stored at METASTORE_SERVICE_DISCOVERY_KEY defined above
    2. The hdfs service selector is stored at HDFS_SERVICE_SELECTOR_KEY defined above
    3. The hdfs uri is stored at HDFS_URI_KEY defined above
"""


def is_hdfs_service_discovery(runtime_type_config):
    return runtime_type_config.get(HDFS_SERVICE_DISCOVERY_KEY, True)


def discover_hdfs_from_workspace(
        cluster_config: Dict[str, Any], runtime_type):
    runtime_config = get_runtime_config(cluster_config)
    runtime_type_config = runtime_config.get(runtime_type, {})
    if (runtime_type_config.get(HDFS_URI_KEY) or
            not is_hdfs_service_discovery(runtime_type_config)):
        return cluster_config

    hdfs_uri = discover_hdfs(
        runtime_type_config, HDFS_SERVICE_SELECTOR_KEY,
        cluster_config=cluster_config,
        discovery_type=DiscoveryType.WORKSPACE)
    if hdfs_uri:
        runtime_type_config = get_config_for_update(
            runtime_config, runtime_type)
        runtime_type_config[HDFS_URI_KEY] = hdfs_uri
    return cluster_config


def discover_hdfs_on_head(
        cluster_config: Dict[str, Any], runtime_type):
    runtime_config = get_runtime_config(cluster_config)
    runtime_type_config = runtime_config.get(runtime_type, {})
    if not is_hdfs_service_discovery(runtime_type_config):
        return cluster_config

    hdfs_uri = runtime_type_config.get(HDFS_URI_KEY)
    if hdfs_uri:
        # HDFS already configured
        return cluster_config

    # There is service discovery to come here
    hdfs_uri = discover_hdfs(
        runtime_type_config, HDFS_SERVICE_SELECTOR_KEY,
        cluster_config=cluster_config,
        discovery_type=DiscoveryType.CLUSTER)
    if hdfs_uri:
        runtime_type_config = get_config_for_update(
            runtime_config, runtime_type)
        runtime_type_config[HDFS_URI_KEY] = hdfs_uri
    return cluster_config


"""
Metastore service discovery conventions:
    1. The metastore service discovery flag is stored at METASTORE_SERVICE_DISCOVERY_KEY defined above
    2. The metastore service selector is stored at METASTORE_SERVICE_SELECTOR_KEY defined above
    3. The metastore uri is stored at METASTORE_URI_KEY defined above
"""


def is_metastore_service_discovery(runtime_type_config):
    return runtime_type_config.get(METASTORE_SERVICE_DISCOVERY_KEY, True)


def discover_metastore_from_workspace(
        cluster_config: Dict[str, Any], runtime_type):
    runtime_config = get_runtime_config(cluster_config)
    runtime_type_config = runtime_config.get(runtime_type, {})
    if (runtime_type_config.get(METASTORE_URI_KEY) or
            has_runtime_in_cluster(
                runtime_config, BUILT_IN_RUNTIME_METASTORE) or
            not is_metastore_service_discovery(runtime_type_config)):
        return cluster_config

    metastore_uri = discover_metastore(
        runtime_type_config, METASTORE_SERVICE_SELECTOR_KEY,
        cluster_config=cluster_config,
        discovery_type=DiscoveryType.WORKSPACE)
    if metastore_uri:
        runtime_type_config = get_config_for_update(
            runtime_config, runtime_type)
        runtime_type_config[METASTORE_URI_KEY] = metastore_uri

    return cluster_config


def discover_metastore_on_head(
        cluster_config: Dict[str, Any], runtime_type):
    runtime_config = get_runtime_config(cluster_config)
    runtime_type_config = runtime_config.get(runtime_type, {})
    if not is_metastore_service_discovery(runtime_type_config):
        return cluster_config

    metastore_uri = runtime_type_config.get(METASTORE_URI_KEY)
    if metastore_uri:
        # Metastore already configured
        return cluster_config

    if has_runtime_in_cluster(
            runtime_config, BUILT_IN_RUNTIME_METASTORE):
        # There is a metastore
        return cluster_config

    # There is service discovery to come here
    metastore_uri = discover_metastore(
        runtime_type_config, METASTORE_SERVICE_SELECTOR_KEY,
        cluster_config=cluster_config,
        discovery_type=DiscoveryType.CLUSTER)
    if metastore_uri:
        runtime_type_config = get_config_for_update(
            runtime_config, runtime_type)
        runtime_type_config[METASTORE_URI_KEY] = metastore_uri
    return cluster_config


"""
Zookeeper service discovery conventions:
    1. The Zookeeper service discovery flag is stored at ZOOKEEPER_SERVICE_DISCOVERY_KEY defined above
    2. The Zookeeper service selector is stored at ZOOKEEPER_SERVICE_SELECTOR_KEY defined above
    3. The Zookeeper connect is stored at ZOOKEEPER_CONNECT_KEY defined above
"""


def is_zookeeper_service_discovery(runtime_type_config):
    return runtime_type_config.get(ZOOKEEPER_SERVICE_DISCOVERY_KEY, True)


def discover_zookeeper_from_workspace(
        cluster_config: Dict[str, Any], runtime_type):
    runtime_config = get_runtime_config(cluster_config)
    runtime_type_config = runtime_config.get(runtime_type, {})

    # Discover zookeeper through workspace
    if (runtime_type_config.get(ZOOKEEPER_CONNECT_KEY) or
            has_runtime_in_cluster(runtime_config, BUILT_IN_RUNTIME_ZOOKEEPER) or
            not is_zookeeper_service_discovery(runtime_type_config)):
        return cluster_config

    zookeeper_uri = discover_zookeeper(
        runtime_type_config, ZOOKEEPER_SERVICE_SELECTOR_KEY,
        cluster_config=cluster_config,
        discovery_type=DiscoveryType.WORKSPACE)
    if zookeeper_uri is not None:
        runtime_type_config = get_config_for_update(
            runtime_config, runtime_type)
        runtime_type_config[ZOOKEEPER_CONNECT_KEY] = zookeeper_uri

    return cluster_config


def discover_zookeeper_on_head(
        cluster_config: Dict[str, Any], runtime_type):
    runtime_config = get_runtime_config(cluster_config)
    runtime_type_config = runtime_config.get(runtime_type, {})
    if not is_zookeeper_service_discovery(runtime_type_config):
        return cluster_config

    zookeeper_uri = runtime_type_config.get(ZOOKEEPER_CONNECT_KEY)
    if zookeeper_uri:
        # Zookeeper already configured
        return cluster_config

    if has_runtime_in_cluster(
            runtime_config, BUILT_IN_RUNTIME_ZOOKEEPER):
        # There is a local Zookeeper
        return cluster_config

    # There is service discovery to come here
    zookeeper_uri = discover_zookeeper(
                    runtime_type_config, ZOOKEEPER_SERVICE_SELECTOR_KEY,
                    cluster_config=cluster_config,
                    discovery_type=DiscoveryType.CLUSTER)
    if zookeeper_uri:
        runtime_type_config = get_config_for_update(
            runtime_config, runtime_type)
        runtime_type_config[ZOOKEEPER_CONNECT_KEY] = zookeeper_uri
    return cluster_config


"""
Database service discovery conventions:
    1. The Database service discovery flag is stored at DATABASE_SERVICE_DISCOVERY_KEY defined above
    2. The Database service selector is stored at DATABASE_SERVICE_SELECTOR_KEY defined above
    3. The Database connect is stored at DATABASE_CONNECT_KEY defined above
"""


def is_database_service_discovery(runtime_type_config):
    return runtime_type_config.get(
        DATABASE_SERVICE_DISCOVERY_KEY, True)


def discover_database_from_workspace(
        cluster_config: Dict[str, Any], runtime_type,
        database_runtime_type=None,
        allow_local=True):
    runtime_config = get_runtime_config(cluster_config)
    runtime_type_config = runtime_config.get(runtime_type, {})
    database_config = runtime_type_config.get(DATABASE_CONNECT_KEY, {})

    # Database check order:
    # 1. if there is a configured database
    # 2. if there is database runtime in the same cluster
    # 3. if there is a cloud database configured (unless disabled by use_managed_database)
    # 4. if there is database can be discovered

    if (is_database_configured(database_config) or
            (allow_local and get_database_runtime_in_cluster(runtime_config)) or
            not is_database_service_discovery(runtime_type_config)):
        return cluster_config

    database_service = discover_database(
        runtime_type_config, DATABASE_SERVICE_SELECTOR_KEY,
        cluster_config=cluster_config,
        discovery_type=DiscoveryType.WORKSPACE,
        database_runtime_type=database_runtime_type)
    if database_service:
        runtime_type_config = get_config_for_update(
            runtime_config, runtime_type)
        database_config = get_config_for_update(
            runtime_type_config, DATABASE_CONNECT_KEY)
        set_database_config(database_config, database_service)

    return cluster_config


def discover_database_on_head(
        cluster_config: Dict[str, Any], runtime_type,
        database_runtime_type=None,
        allow_local=True):
    runtime_config = get_runtime_config(cluster_config)
    runtime_type_config = runtime_config.get(runtime_type, {})
    if not is_database_service_discovery(runtime_type_config):
        return cluster_config

    database_config = runtime_type_config.get(DATABASE_CONNECT_KEY, {})
    if (is_database_configured(database_config) or
            (allow_local and get_database_runtime_in_cluster(runtime_config))):
        # Database already configured
        return cluster_config

    # There is service discovery to come here
    database_service = discover_database(
        runtime_type_config, DATABASE_SERVICE_SELECTOR_KEY,
        cluster_config=cluster_config,
        discovery_type=DiscoveryType.CLUSTER,
        database_runtime_type=database_runtime_type)
    if database_service:
        runtime_type_config = get_config_for_update(
            runtime_config, runtime_type)
        database_config = get_config_for_update(
            runtime_type_config, DATABASE_CONNECT_KEY)
        set_database_config(database_config, database_service)
    return cluster_config


"""
ETCD service discovery conventions:
    1. The ETCD service discovery flag is stored at ETCD_SERVICE_DISCOVERY_KEY defined above
    2. The ETCD service selector is stored at ETCD_SERVICE_SELECTOR_KEY defined above
    3. The ETCD uri is stored at ETCD_URI_KEY defined above
"""


def is_etcd_service_discovery(runtime_type_config):
    return runtime_type_config.get(ETCD_SERVICE_DISCOVERY_KEY, True)


def discover_etcd_from_workspace(
        cluster_config: Dict[str, Any], runtime_type):
    runtime_config = get_runtime_config(cluster_config)
    runtime_type_config = runtime_config.get(runtime_type, {})

    # Discover etcd through workspace
    if (runtime_type_config.get(ETCD_URI_KEY) or
            has_runtime_in_cluster(runtime_config, BUILT_IN_RUNTIME_ETCD) or
            not is_etcd_service_discovery(runtime_type_config)):
        return cluster_config

    etcd_uri = discover_etcd(
        runtime_type_config, ETCD_SERVICE_SELECTOR_KEY,
        cluster_config=cluster_config,
        discovery_type=DiscoveryType.WORKSPACE)
    if etcd_uri is not None:
        runtime_type_config = get_config_for_update(
            runtime_config, runtime_type)
        runtime_type_config[ETCD_URI_KEY] = etcd_uri

    return cluster_config


def discover_etcd_on_head(
        cluster_config: Dict[str, Any], runtime_type):
    runtime_config = get_runtime_config(cluster_config)
    runtime_type_config = runtime_config.get(runtime_type, {})
    if not is_etcd_service_discovery(runtime_type_config):
        return cluster_config

    etcd_uri = runtime_type_config.get(ETCD_URI_KEY)
    if etcd_uri:
        # Zookeeper already configured
        return cluster_config

    if has_runtime_in_cluster(
            runtime_config, BUILT_IN_RUNTIME_ETCD):
        # There is a local Zookeeper
        return cluster_config

    # There is service discovery to come here
    etcd_uri = discover_etcd(
                    runtime_type_config, ETCD_SERVICE_SELECTOR_KEY,
                    cluster_config=cluster_config,
                    discovery_type=DiscoveryType.CLUSTER)
    if etcd_uri:
        runtime_type_config = get_config_for_update(
            runtime_config, runtime_type)
        runtime_type_config[ETCD_URI_KEY] = etcd_uri
    return cluster_config


def export_database_runtime_environment_variables(
        runtime_config, runtime_type):
    # WARNING: This is helper function to get database parameters from a runtime configuration
    # if there are changes in the runtime configuration side, this needs to be changed too.
    # if it is called at the node configure context, the head ip will be set as host

    runtime_type_config = runtime_config.get(runtime_type, {})
    engine = get_database_engine_for_runtime(
        runtime_type)
    head_ip = get_runtime_value(CLOUDTIK_RUNTIME_ENV_HEAD_IP)
    port = runtime_type_config.get(DATABASE_SERVICE_PORT_CONFIG_KEY)
    if not port:
        port = get_database_default_port(engine)

    if engine == DATABASE_ENGINE_MYSQL:
        username = None
        password = runtime_type_config.get(
            MYSQL_ROOT_PASSWORD_CONFIG_KEY)
    else:
        username = runtime_type_config.get(
            POSTGRES_ADMIN_USER_CONFIG_KEY)
        password = runtime_type_config.get(
            POSTGRES_ADMIN_PASSWORD_CONFIG_KEY)
    if not username:
        username = get_database_default_username(engine)
    if not password:
        password = get_database_default_password(engine)

    os.environ[DATABASE_ENV_ENABLED] = get_env_string_value(True)
    os.environ[DATABASE_ENV_ENGINE] = engine
    os.environ[DATABASE_ENV_HOST] = head_ip
    os.environ[DATABASE_ENV_PORT] = str(port)

    # The defaults apply to built-in Database runtime.
    os.environ[DATABASE_ENV_USERNAME] = username
    os.environ[DATABASE_ENV_PASSWORD] = password
