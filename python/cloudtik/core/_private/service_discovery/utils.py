from enum import Enum
from typing import Optional, Dict, Any

from cloudtik.core._private.core_utils import deserialize_config, serialize_config

# The standard keys and values used for service discovery

SERVICE_DISCOVERY_PROTOCOL = "protocol"
SERVICE_DISCOVERY_PROTOCOL_TCP = "TCP"

SERVICE_DISCOVERY_PORT = "port"

SERVICE_DISCOVERY_NODE_KIND = "node_kind"
SERVICE_DISCOVERY_NODE_KIND_HEAD = "head"
SERVICE_DISCOVERY_NODE_KIND_WORKER = "worker"
SERVICE_DISCOVERY_NODE_KIND_NODE = "node"

SERVICE_DISCOVERY_TAGS = "tags"
SERVICE_DISCOVERY_LABELS = "labels"

SERVICE_DISCOVERY_LABEL_CLUSTER = "cloudtik-cluster"
SERVICE_DISCOVERY_LABEL_RUNTIME = "cloudtik-runtime"

SERVICE_DISCOVERY_CHECK_INTERVAL = "check_interval"
SERVICE_DISCOVERY_CHECK_TIMEOUT = "check_timeout"

# A boolean value indicate whether this is a service for exporting metrics
# for auto discovering the metrics services from collector server
SERVICE_DISCOVERY_METRICS = "metrics"

# Standard runtime configurations for service discovery
SERVICE_DISCOVERY_CONFIG_SERVICE_DISCOVERY = "service"
SERVICE_DISCOVERY_CONFIG_MEMBER_OF = "member_of"
SERVICE_DISCOVERY_CONFIG_TAGS = "tags"
SERVICE_DISCOVERY_CONFIG_LABELS = "labels"

# The config keys for a standard service selector
SERVICE_SELECTOR_SERVICES = "services"
SERVICE_SELECTOR_TAGS = "tags"
SERVICE_SELECTOR_LABELS = "labels"
SERVICE_SELECTOR_EXCLUDE_LABELS = "exclude_labels"
SERVICE_SELECTOR_RUNTIMES = "runtimes"
SERVICE_SELECTOR_CLUSTERS = "clusters"


class ServiceScope(Enum):
    """The service scope decide how the canonical service name is formed.
    For workspace scoped service, the runtime service name is used directly
    as the service name and the cluster name as a tag.
    For cluster scoped service, the cluster name will be prefixed with the
    runtime service name to form a unique canonical service name.

    """
    WORKSPACE = 1
    CLUSTER = 2


def get_service_discovery_config(config):
    return config.get(SERVICE_DISCOVERY_CONFIG_SERVICE_DISCOVERY, {})


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
        metrics: bool = False):
    service_def = {
        SERVICE_DISCOVERY_PROTOCOL: SERVICE_DISCOVERY_PROTOCOL_TCP,
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
    if metrics:
        service_def[SERVICE_DISCOVERY_METRICS] = metrics

    return service_def


def define_runtime_service_on_worker(
        service_discovery_config: Optional[Dict[str, Any]],
        service_port,
        metrics: bool = False):
    return define_runtime_service(
        service_discovery_config,
        service_port,
        node_kind=SERVICE_DISCOVERY_NODE_KIND_WORKER,
        metrics=metrics)


def define_runtime_service_on_head(
        service_discovery_config,
        service_port,
        metrics: bool = False):
    return define_runtime_service(
        service_discovery_config,
        service_port,
        node_kind=SERVICE_DISCOVERY_NODE_KIND_HEAD,
        metrics=metrics)


def define_runtime_service_on_head_or_all(
        service_discovery_config,
        service_port, head_or_all,
        metrics: bool = False):
    node_kind = SERVICE_DISCOVERY_NODE_KIND_NODE \
        if head_or_all else SERVICE_DISCOVERY_NODE_KIND_HEAD
    return define_runtime_service(
        service_discovery_config,
        service_port,
        node_kind=node_kind,
        metrics=metrics)


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


def is_service_for_metrics(runtime_service):
    return runtime_service.get(SERVICE_DISCOVERY_METRICS, False)


def serialize_service_selector(service_selector):
    if not service_selector:
        return None
    return serialize_config(service_selector)


def deserialize_service_selector(service_selector_str):
    if not service_selector_str:
        return None
    return deserialize_config(service_selector_str)
