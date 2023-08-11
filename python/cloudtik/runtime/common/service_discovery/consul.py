from urllib.parse import quote

from cloudtik.core._private.service_discovery.utils import SERVICE_SELECTOR_SERVICES, SERVICE_SELECTOR_TAGS, \
    SERVICE_SELECTOR_LABELS, SERVICE_SELECTOR_EXCLUDE_LABELS, SERVICE_DISCOVERY_LABEL_CLUSTER, \
    SERVICE_SELECTOR_RUNTIMES, SERVICE_SELECTOR_CLUSTERS, SERVICE_SELECTOR_EXCLUDE_JOINED_LABELS
from cloudtik.core._private.util.rest_api import rest_api_get_json

REST_ENDPOINT_URL_FORMAT = "http://{}:{}{}"
REST_ENDPOINT_CATALOG = "/v1/catalog"
REST_ENDPOINT_CATALOG_SERVICES = REST_ENDPOINT_CATALOG + "/services"
REST_ENDPOINT_CATALOG_SERVICE = REST_ENDPOINT_CATALOG + "/service"

CONSUL_CLIENT_ADDRESS = "127.0.0.1"
CONSUL_CLIENT_PORT = 8500
CONSUL_REQUEST_TIMEOUT = 5


def consul_api_get(endpoint: str):
    endpoint_url = REST_ENDPOINT_URL_FORMAT.format(
        CONSUL_CLIENT_ADDRESS, CONSUL_CLIENT_PORT, endpoint)
    return rest_api_get_json(endpoint_url, timeout=CONSUL_REQUEST_TIMEOUT)


def get_expressions_of_service_selector(service_selector):
    if not service_selector:
        return None

    services = service_selector.get(SERVICE_SELECTOR_SERVICES)
    tags = service_selector.get(SERVICE_SELECTOR_TAGS)
    labels = service_selector.get(SERVICE_SELECTOR_LABELS)
    runtimes = service_selector.get(SERVICE_SELECTOR_RUNTIMES)
    clusters = service_selector.get(SERVICE_SELECTOR_CLUSTERS)
    exclude_labels = service_selector.get(SERVICE_SELECTOR_EXCLUDE_LABELS)
    exclude_joined_labels = service_selector.get(SERVICE_SELECTOR_EXCLUDE_JOINED_LABELS)

    if not (services or tags or labels or
            runtimes or clusters or
            exclude_labels or exclude_joined_labels):
        return None

    expressions = []
    if services:
        # Any services in the list (OR)
        service_expressions = []
        for service_name in services:
            service_expressions.append('ServiceName == "{}"'.format(service_name))
        expressions.append(" or ".join(service_expressions))

    if tags:
        # Services must contain all the tags (AND)
        tag_expressions = []
        for tag in tags:
            tag_expressions.append('"{}" in ServiceTags'.format(tag))
        expressions.append(" and ".join(tag_expressions))

    if labels:
        # All labels must match (AND)
        label_expressions = []
        for label_name, label_value in labels.items():
            label_expressions.append(
                'ServiceMeta["{}"] matches "{}"'.format(label_name, label_value))
        expressions.append(" and ".join(label_expressions))

    if runtimes:
        # Services of any these runtimes will be included (OR)
        runtime_expressions = []
        for runtime in runtimes:
            runtime_expressions.append(
                'ServiceMeta["cloudtik-runtime"] == "{}"'.format(runtime))
        expressions.append(" or ".join(runtime_expressions))

    if clusters:
        # Services of any these clusters will be included (OR)
        cluster_expressions = []
        for cluster in clusters:
            cluster_expressions.append(
                'ServiceMeta["cloudtik-cluster"] == "{}"'.format(cluster))
        expressions.append(" or ".join(cluster_expressions))

    if exclude_labels:
        # Services with any matched labels will be excluded [NOT (OR)] or [AND NOT]
        exclude_label_expressions = []
        for label_name, label_value in exclude_labels.items():
            exclude_label_expressions.append(
                'ServiceMeta["{}"] not matches "{}"'.format(label_name, label_value))
        expressions.append(" and ".join(exclude_label_expressions))

    if exclude_joined_labels:
        # Services with all matched labels will be excluded [NOT (AND)] or [OR NOT]
        exclude_joined_expressions = []
        for joined_labels in exclude_joined_labels:
            # all the labels must match for each joined labels
            joined_label_expressions = []
            for label_name, label_value in joined_labels.items():
                joined_label_expressions.append(
                    'ServiceMeta["{}"] not matches "{}"'.format(label_name, label_value))
            exclude_joined_expressions.append(" or ".join(joined_label_expressions))
        expressions.append(" and ".join(["( {} )".format(
            expr) for expr in exclude_joined_expressions]))

    return " and ".join(["( {} )".format(expr) for expr in expressions])


def _get_endpoint_with_service_selector(base_endpoint, service_selector):
    expressions = get_expressions_of_service_selector(service_selector)
    if expressions:
        encoded_expressions = quote(expressions)
        query_filter = "filter=" + encoded_expressions
        endpoint = "{}?{}".format(
            base_endpoint, query_filter)
    else:
        endpoint = base_endpoint
    return endpoint


def query_services(service_selector):
    # query all the services with a service selector
    query_endpoint = _get_endpoint_with_service_selector(
        REST_ENDPOINT_CATALOG_SERVICES, service_selector)

    # The response is a dictionary of services
    return consul_api_get(query_endpoint)


def query_service_nodes(service_name, service_selector):
    service_endpoint = "{}/{}".format(
            REST_ENDPOINT_CATALOG_SERVICE, service_name)
    query_endpoint = _get_endpoint_with_service_selector(
        service_endpoint, service_selector)

    # The response is a list of server nodes of this service
    return consul_api_get(query_endpoint)


def get_service_name(service_node):
    return service_node["ServiceName"]


def get_service_address(service_node):
    service_host = service_node.get("ServiceAddress")
    if not service_host:
        service_host = service_node.get("Address")
    port = service_node["ServicePort"]
    return service_host, port


def get_service_cluster(service_node):
    # This is our service implementation specific
    # each service will be labeled with its cluster name
    service_meta = service_node.get(
        "ServiceMeta", {})
    return service_meta.get(SERVICE_DISCOVERY_LABEL_CLUSTER)


def get_service_dns_name(
        service_name, service_tag, service_cluster):
    if service_tag and service_cluster:
        raise ValueError(
            "You can either specify service tag or service cluster. But not both.")
    elif not service_tag and not service_cluster:
        return "{}.service.cloudtik".format(service_name)
    elif service_tag:
        return "{}.{}.service.cloudtik".format(service_tag, service_name)
    else:
        return "cloudtik-c-{}.{}.service.cloudtik".format(service_cluster, service_name)


def get_rfc2782_service_dns_name(
        service_name, service_tag, service_cluster):
    # The service dns name format of RFC 2782 lookup
    # SRV queries must prepend service and protocol values with an underscore (_)
    # or _<service>._<protocol>[.service][.<datacenter>].<domain>
    # or _<service>._<tag>[.service][.<datacenter>].<domain>
    if service_tag and service_cluster:
        raise ValueError(
            "You can either specify service tag or service cluster. But not both.")
    elif not service_tag and not service_cluster:
        return "_{}._tcp.service.cloudtik".format(service_name)
    elif service_tag:
        return "_{}._{}.service.cloudtik".format(service_name, service_tag)
    else:
        return "_{}._cloudtik-c-{}.service.cloudtik".format(service_name, service_cluster)
