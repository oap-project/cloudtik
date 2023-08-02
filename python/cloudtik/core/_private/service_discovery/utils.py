
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

SERVICE_DISCOVERY_META_CLUSTER = "cluster"
SERVICE_DISCOVERY_META_RUNTIME = "runtime"

SERVICE_DISCOVERY_CHECK_INTERVAL = "check_interval"
SERVICE_DISCOVERY_CHECK_TIMEOUT = "check_timeout"


# Standard runtime configurations for service discovery
SERVICE_DISCOVERY_CONFIG_MEMBER_OF = "member_of"


def get_canonical_service_name(
        config, cluster_name, runtime_service_name):
    member_of = config.get(SERVICE_DISCOVERY_CONFIG_MEMBER_OF)
    if member_of:
        # This service is a member of a service group
        return member_of
    else:
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
