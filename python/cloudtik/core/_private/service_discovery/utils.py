from enum import Enum

# The standard keys and values used for service discovery

SERVICE_DISCOVERY_PROTOCOL = "protocol"
SERVICE_DISCOVERY_PROTOCOL_TCP = "TCP"

SERVICE_DISCOVERY_PORT = "port"

SERVICE_DISCOVERY_NODE_KIND = "node_kind"
SERVICE_DISCOVERY_NODE_KIND_HEAD = "head"
SERVICE_DISCOVERY_NODE_KIND_WORKER = "worker"
SERVICE_DISCOVERY_NODE_KIND_NODE = "node"

SERVICE_DISCOVERY_TAGS = "tags"
SERVICE_DISCOVERY_META = "meta"

SERVICE_DISCOVERY_META_CLUSTER = "cloudtik-cluster"
SERVICE_DISCOVERY_META_RUNTIME = "cloudtik-runtime"

SERVICE_DISCOVERY_CHECK_INTERVAL = "check_interval"
SERVICE_DISCOVERY_CHECK_TIMEOUT = "check_timeout"


# Standard runtime configurations for service discovery
SERVICE_DISCOVERY_CONFIG_MEMBER_OF = "member_of"


class ServiceScope(Enum):
    """The service scope decide how the canonical service name is formed.
    For workspace scoped service, the runtime service name is used directly
    as the service name and the cluster name as a tag.
    For cluster scoped service, the cluster name will be prefixed with the
    runtime service name to form a unique canonical service name.

    """
    WORKSPACE = 1
    CLUSTER = 2


def get_canonical_service_name(
        config, cluster_name, runtime_service_name,
        service_scope: ServiceScope = ServiceScope.WORKSPACE):
    member_of = config.get(SERVICE_DISCOVERY_CONFIG_MEMBER_OF)
    if member_of:
        # override the service name
        return member_of
    else:
        if service_scope == ServiceScope.WORKSPACE:
            return runtime_service_name
        else:
            # cluster name as prefix of service name
            return "{}-{}".format(cluster_name, runtime_service_name)


def define_runtime_service(service_port, node_kind=SERVICE_DISCOVERY_NODE_KIND_NODE):
    return {
        SERVICE_DISCOVERY_PROTOCOL: SERVICE_DISCOVERY_PROTOCOL_TCP,
        SERVICE_DISCOVERY_PORT: service_port,
        SERVICE_DISCOVERY_NODE_KIND: node_kind
    }


def define_runtime_service_on_worker(service_port):
    return define_runtime_service(
        service_port, SERVICE_DISCOVERY_NODE_KIND_WORKER)


def define_runtime_service_on_head(service_port):
    return define_runtime_service(
        service_port, SERVICE_DISCOVERY_NODE_KIND_HEAD)


def define_runtime_service_on_head_or_all(service_port, head_or_all):
    node_kind = SERVICE_DISCOVERY_NODE_KIND_NODE \
        if head_or_all else SERVICE_DISCOVERY_NODE_KIND_HEAD
    return define_runtime_service(
        service_port, node_kind)


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
