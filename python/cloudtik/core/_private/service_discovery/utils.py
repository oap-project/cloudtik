import copy
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Union

from cloudtik.core._private.core_utils import deserialize_config, serialize_config, \
    get_list_for_update

# The standard keys and values used for service discovery

SERVICE_DISCOVERY_PROTOCOL = "protocol"
SERVICE_DISCOVERY_PROTOCOL_TCP = "tcp"
SERVICE_DISCOVERY_PROTOCOL_HTTP = "http"

SERVICE_DISCOVERY_PORT = "port"

SERVICE_DISCOVERY_NODE_KIND = "node_kind"
SERVICE_DISCOVERY_NODE_KIND_HEAD = "head"
SERVICE_DISCOVERY_NODE_KIND_WORKER = "worker"
SERVICE_DISCOVERY_NODE_KIND_NODE = "node"

SERVICE_DISCOVERY_TAGS = "tags"
SERVICE_DISCOVERY_LABELS = "labels"

SERVICE_DISCOVERY_TAG_CLUSTER_PREFIX = "cloudtik-c-"
SERVICE_DISCOVERY_TAG_FEATURE_PREFIX = "cloudtik-f-"
SERVICE_DISCOVERY_TAG_SYSTEM_PREFIX = "cloudtik-"

SERVICE_DISCOVERY_LABEL_CLUSTER = "cloudtik-cluster"
SERVICE_DISCOVERY_LABEL_RUNTIME = "cloudtik-runtime"

SERVICE_DISCOVERY_CHECK_INTERVAL = "check_interval"
SERVICE_DISCOVERY_CHECK_TIMEOUT = "check_timeout"

# A set of predefined features that a service belongs to
# A single service can provide more than one features
SERVICE_DISCOVERY_FEATURES = "features"

SERVICE_DISCOVERY_FEATURE_DATABASE = "database"
SERVICE_DISCOVERY_FEATURE_STORAGE = "storage"
SERVICE_DISCOVERY_FEATURE_ANALYTICS = "analytics"
SERVICE_DISCOVERY_FEATURE_SCHEDULER = "scheduler"
SERVICE_DISCOVERY_FEATURE_DNS = "dns"
SERVICE_DISCOVERY_FEATURE_LOAD_BALANCER = "load-balancer"
SERVICE_DISCOVERY_FEATURE_KEY_VALUE = "key-value"
SERVICE_DISCOVERY_FEATURE_METRICS = "metrics"
SERVICE_DISCOVERY_FEATURE_MESSAGING = "messaging"
SERVICE_DISCOVERY_FEATURE_API_GATEWAY = "api-gateway"

# Standard runtime configurations for service discovery
SERVICE_DISCOVERY_CONFIG_SERVICE_DISCOVERY = "service"
SERVICE_DISCOVERY_CONFIG_MEMBER_OF = "member_of"
SERVICE_DISCOVERY_CONFIG_TAGS = "tags"
SERVICE_DISCOVERY_CONFIG_LABELS = "labels"
SERVICE_DISCOVERY_CONFIG_PREFER_WORKSPACE = "prefer_workspace"

# The config keys for a standard service selector
SERVICE_SELECTOR_SERVICES = "services"
SERVICE_SELECTOR_TAGS = "tags"
SERVICE_SELECTOR_LABELS = "labels"
SERVICE_SELECTOR_EXCLUDE_LABELS = "exclude_labels"
SERVICE_SELECTOR_EXCLUDE_JOINED_LABELS = "exclude_joined_labels"
SERVICE_SELECTOR_RUNTIMES = "runtimes"
SERVICE_SELECTOR_CLUSTERS = "clusters"


class ServiceAddressType(Enum):
    NODE_IP = auto()
    NODE_FQDN = auto()
    SERVICE_FQDN = auto()


class ServiceScope(Enum):
    """The service scope decide how the canonical service name is formed.
    For workspace scoped service, the runtime service name is used directly
    as the service name and the cluster name as a tag.
    For cluster scoped service, the cluster name will be prefixed with the
    runtime service name to form a unique canonical service name.

    """
    WORKSPACE = 1
    CLUSTER = 2


class ServiceRegisterException(RuntimeError):
    pass


def get_service_discovery_config(config):
    return config.get(SERVICE_DISCOVERY_CONFIG_SERVICE_DISCOVERY, {})


def is_prefer_workspace_discovery(
        service_discovery_config: Optional[Dict[str, Any]]):
    return service_discovery_config.get(
        SERVICE_DISCOVERY_CONFIG_PREFER_WORKSPACE, False)


def get_canonical_service_name(
        service_discovery_config: Optional[Dict[str, Any]],
        cluster_name,
        runtime_service_name,
        service_scope: ServiceScope = ServiceScope.WORKSPACE):
    member_of = service_discovery_config.get(
        SERVICE_DISCOVERY_CONFIG_MEMBER_OF)
    if member_of:
        # override the service name
        return member_of
    else:
        if service_scope == ServiceScope.WORKSPACE:
            return runtime_service_name
        else:
            # cluster name as prefix of service name
            return "{}-{}".format(cluster_name, runtime_service_name)


def define_runtime_service(
        service_discovery_config: Optional[Dict[str, Any]],
        service_port,
        node_kind=SERVICE_DISCOVERY_NODE_KIND_NODE,
        protocol: str = None,
        features: List[str] = None):
    if not protocol:
        protocol = SERVICE_DISCOVERY_PROTOCOL_TCP
    service_def = {
        SERVICE_DISCOVERY_PROTOCOL: protocol,
        SERVICE_DISCOVERY_PORT: service_port,
    }

    if node_kind and node_kind != SERVICE_DISCOVERY_NODE_KIND_NODE:
        service_def[SERVICE_DISCOVERY_NODE_KIND] = node_kind

    tags = service_discovery_config.get(SERVICE_DISCOVERY_CONFIG_TAGS)
    if tags:
        service_def[SERVICE_DISCOVERY_TAGS] = tags
    labels = service_discovery_config.get(SERVICE_DISCOVERY_CONFIG_LABELS)
    if labels:
        service_def[SERVICE_DISCOVERY_LABELS] = labels
    if features:
        service_def[SERVICE_DISCOVERY_FEATURES] = features

    return service_def


def define_runtime_service_on_worker(
        service_discovery_config: Optional[Dict[str, Any]],
        service_port,
        protocol: str = None,
        features: List[str] = None):
    return define_runtime_service(
        service_discovery_config,
        service_port,
        node_kind=SERVICE_DISCOVERY_NODE_KIND_WORKER,
        protocol=protocol,
        features=features)


def define_runtime_service_on_head(
        service_discovery_config,
        service_port,
        protocol: str = None,
        features: List[str] = None):
    return define_runtime_service(
        service_discovery_config,
        service_port,
        node_kind=SERVICE_DISCOVERY_NODE_KIND_HEAD,
        protocol=protocol,
        features=features)


def define_runtime_service_on_head_or_all(
        service_discovery_config,
        service_port, head_or_all,
        protocol: str = None,
        features: List[str] = None):
    node_kind = SERVICE_DISCOVERY_NODE_KIND_NODE \
        if head_or_all else SERVICE_DISCOVERY_NODE_KIND_HEAD
    return define_runtime_service(
        service_discovery_config,
        service_port,
        node_kind=node_kind,
        protocol=protocol,
        features=features)


def match_service_node(runtime_service, head):
    node_kind = runtime_service.get(SERVICE_DISCOVERY_NODE_KIND)
    if not node_kind or node_kind == SERVICE_DISCOVERY_NODE_KIND_NODE:
        return True
    if head:
        if node_kind == SERVICE_DISCOVERY_NODE_KIND_HEAD:
            return True
    else:
        if node_kind == SERVICE_DISCOVERY_NODE_KIND_WORKER:
            return True

    return False


def get_runtime_service_features(runtime_service):
    return runtime_service.get(SERVICE_DISCOVERY_FEATURES)


def has_runtime_service_feature(runtime_service, feature):
    features = get_runtime_service_features(runtime_service)
    return False if not features or feature not in features else True


def serialize_service_selector(service_selector):
    if not service_selector:
        return None
    return serialize_config(service_selector)


def deserialize_service_selector(service_selector_str):
    if not service_selector_str:
        return None
    return deserialize_config(service_selector_str)


def exclude_runtime_of_cluster(
        service_selector, runtime, cluster_name):
    if not (runtime or cluster_name):
        return
    exclude_joined_labels = get_list_for_update(
        service_selector, SERVICE_SELECTOR_EXCLUDE_JOINED_LABELS)

    joined_labels = {}
    if runtime:
        joined_labels[SERVICE_DISCOVERY_LABEL_RUNTIME] = runtime
    if cluster_name:
        joined_labels[SERVICE_DISCOVERY_LABEL_CLUSTER] = cluster_name

    exclude_joined_labels.append(joined_labels)
    return service_selector


def get_service_selector_for_update(config, config_key):
    service_selector = config.get(
        config_key)
    if service_selector is None:
        service_selector = {}
    else:
        service_selector = copy.deepcopy(service_selector)
    return service_selector


def include_runtime_for_selector(
        service_selector, runtime: Union[str, List[str]], override=False):
    runtimes = get_list_for_update(
        service_selector, SERVICE_SELECTOR_RUNTIMES)
    if runtimes:
        if not override:
            return service_selector
        runtimes.clear()

    if isinstance(runtime, str):
        runtimes.append(runtime)
    else:
        # list of runtime types
        for runtime_type in runtime:
            runtimes.append(runtime_type)
    return service_selector


def include_feature_for_selector(service_selector, feature):
    tags = get_list_for_update(
        service_selector, SERVICE_SELECTOR_TAGS)
    feature_tag = SERVICE_DISCOVERY_TAG_FEATURE_PREFIX + feature
    if feature_tag not in tags:
        tags.append(feature_tag)
    return service_selector
