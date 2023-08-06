import urllib
from urllib.parse import quote

from cloudtik.core._private.service_discovery.utils import SERVICE_SELECTOR_SERVICES, SERVICE_SELECTOR_TAGS, \
    SERVICE_SELECTOR_LABELS, SERVICE_SELECTOR_EXCLUDE_LABELS

REST_REQUEST_TIMEOUT = 10
REST_ENDPOINT_URL_FORMAT = "http://{}:{}/{}"
REST_ENDPOINT_CATALOG_SERVICES = "v1/catalog/services"

CONSUL_CLIENT_ADDRESS = "127.0.0.1"
CONSUL_CLIENT_PORT = 8500


def request_rest_api(rest_api_ip: str, rest_api_port: int, endpoint: str):
    endpoint_url = REST_ENDPOINT_URL_FORMAT.format(
        rest_api_ip, rest_api_port, endpoint)

    # disable all proxy on 127.0.0.1
    proxy_support = urllib.request.ProxyHandler({"no": "127.0.0.1"})
    opener = urllib.request.build_opener(proxy_support)
    urllib.request.install_opener(opener)

    response = urllib.request.urlopen(
        endpoint_url, timeout=REST_REQUEST_TIMEOUT)
    return response.read()


def get_expressions_of_service_selector(service_selector):
    if not service_selector:
        return None

    services = service_selector.get(SERVICE_SELECTOR_SERVICES)
    tags = service_selector.get(SERVICE_SELECTOR_TAGS)
    labels = service_selector.get(SERVICE_SELECTOR_LABELS)
    exclude_labels = service_selector.get(SERVICE_SELECTOR_EXCLUDE_LABELS)

    if not (services or tags or labels or exclude_labels):
        return None

    expressions = []
    if services:
        # Any services in the list (OR)
        service_expressions = []
        for service_name in services:
            service_expressions.append('ServiceName=="{}"'.format(service_name))
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
        for label_name, label_value in labels.items:
            label_expressions.append(
                'ServiceMeta["{}"]=="{}"'.format(label_name, label_value))
        expressions.append(" and ".join(label_expressions))

    if exclude_labels:
        # Services with label in any exclude labels will not match (OR)
        exclude_label_expressions = []
        for label_name, label_value in labels.items:
            exclude_label_expressions.append(
                'ServiceMeta["{}"]!="{}"'.format(label_name, label_value))
        expressions.append(" and ".join(exclude_label_expressions))

    return " and ".join(["( {} )".format(expr) for expr in expressions])


def query_services(service_selector):
    # query all the services with a service selector
    expressions = get_expressions_of_service_selector(service_selector)
    if expressions:
        encoded_expressions = quote(expressions)
        query_filter = "filter=" + encoded_expressions
        services_endpoint = "{}?{}".format(
            REST_ENDPOINT_CATALOG_SERVICES, query_filter)
    else:
        services_endpoint = REST_ENDPOINT_CATALOG_SERVICES

    return request_rest_api(
        CONSUL_CLIENT_ADDRESS, CONSUL_CLIENT_PORT,
        services_endpoint
    )
