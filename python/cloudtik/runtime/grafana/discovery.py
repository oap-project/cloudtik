import logging

from cloudtik.core._private.core_utils import get_list_for_update
from cloudtik.core._private.runtime_factory import BUILT_IN_RUNTIME_PROMETHEUS
from cloudtik.core._private.service_discovery.utils import SERVICE_SELECTOR_RUNTIMES, deserialize_service_selector
from cloudtik.core._private.util.pull.pull_job import PullJob
from cloudtik.core._private.util.rest_api import rest_api_get_json, rest_api_post_json, rest_api_delete_json, \
    REST_API_AUTH_TYPE, REST_API_AUTH_BASIC, REST_API_AUTH_BASIC_USERNAME, REST_API_AUTH_BASIC_PASSWORD
from cloudtik.runtime.common.service_discovery.consul import query_services, query_service_nodes, get_service_name_of_node, \
    get_service_address_of_node, get_service_cluster_of_node
from cloudtik.runtime.grafana.utils import get_data_source_name, get_prometheus_data_source, \
    GRAFANA_DATA_SOURCE_AUTO_CREATED

logger = logging.getLogger(__name__)


REST_API_ENDPOINT_DATA_SOURCES = "/api/datasources"
REST_API_ENDPOINT_DATA_SOURCES_BY_NAME = REST_API_ENDPOINT_DATA_SOURCES + "/name"
DATA_SOURCE_RUNTIMES = [
    BUILT_IN_RUNTIME_PROMETHEUS,
]


def _get_prometheus_data_source(service_node):
    # How to make sure the name is unique
    # with cluster name?
    service_name = get_service_name_of_node(service_node)
    cluster_name = get_service_cluster_of_node(service_node)
    name = get_data_source_name(service_name, cluster_name)
    service_host, service_port = get_service_address_of_node(service_node)
    url = "http://{}:{}".format(service_host, service_port)
    return get_prometheus_data_source(name, url, is_default=False)


def _is_auto_created(data_source):
    meta = data_source.get("jsonData")
    if not meta:
        return False
    return meta.get(GRAFANA_DATA_SOURCE_AUTO_CREATED, False)


def _get_new_data_sources(data_sources, existing_data_sources):
    # for all in data_sources, if it is not in the existing data sources,
    # we need to add new data sources
    new_data_sources = {}
    for data_source_name, data_source in data_sources.items():
        if data_source_name not in existing_data_sources:
            new_data_sources[data_source_name] = data_source
    return new_data_sources


def _get_delete_data_sources(data_sources, existing_data_sources):
    # for all the existing data sources that not in data_sources
    # we need to delete
    delete_data_sources = set()
    for data_source_name, data_source in existing_data_sources.items():
        if not _is_auto_created(data_source):
            continue
        if data_source_name not in data_sources:
            delete_data_sources.add(data_source_name)
    return delete_data_sources


class DiscoverDataSources(PullJob):
    """Pulling job for discovering data sources through service discovery"""

    def __init__(self,
                 grafana_endpoint=None,
                 service_selector=None):
        if not grafana_endpoint:
            raise RuntimeError("Grafana endpoint is needed for pulling data sources.")

        self.service_selector = deserialize_service_selector(
            service_selector)
        self.grafana_endpoint = grafana_endpoint
        self._apply_data_source_runtime_selector()
        self.auth = {
            REST_API_AUTH_TYPE: REST_API_AUTH_BASIC,
            REST_API_AUTH_BASIC_USERNAME: "cloudtik",
            REST_API_AUTH_BASIC_PASSWORD: "cloudtik"
        }

    def pull(self):
        selected_services = self._query_services()
        data_sources = {}
        for service_name in selected_services:
            service_nodes = self._query_service_nodes(service_name)
            # each node is a data source. if many nodes form a load balancer in a cluster
            # it should be filtered by service selector using service name ,tags or labels
            for service_node in service_nodes:
                data_source = _get_prometheus_data_source(service_node)
                data_source_name = data_source["name"]
                data_sources[data_source_name] = data_source

        self._configure_data_sources(data_sources)

    def _apply_data_source_runtime_selector(self):
        # Currently we only support Prometheus data sources
        # add the runtime label selector to the services
        if self.service_selector is None:
            self.service_selector = {}
        runtimes = get_list_for_update(
            self.service_selector, SERVICE_SELECTOR_RUNTIMES)
        for runtime in DATA_SOURCE_RUNTIMES:
            if runtime not in runtimes:
                runtimes.append(runtime)

    def _query_services(self):
        return query_services(self.service_selector)

    def _query_service_nodes(self, service_name):
        return query_service_nodes(service_name, self.service_selector)

    def _configure_data_sources(self, data_sources):
        # 1. delete data sources was added but now exists
        # 2. add new data sources
        existing_data_sources = self._query_data_sources()
        new_data_sources = _get_new_data_sources(
            data_sources, existing_data_sources)
        delete_data_sources = _get_delete_data_sources(
            data_sources, existing_data_sources)

        self._add_data_sources(new_data_sources)
        self._delete_data_sources(delete_data_sources)

    def _query_data_sources(self):
        endpoint_url = "{}{}".format(
            self.grafana_endpoint, REST_API_ENDPOINT_DATA_SOURCES)
        # The response is a list of data sources
        data_sources = rest_api_get_json(
            endpoint_url, auth=self.auth)
        # filter all the data sources that added by us
        return {
            data_source["name"]: data_source
            for data_source in data_sources}

    def _add_data_sources(self, new_data_sources):
        for data_source_name, data_source in new_data_sources.items():
            self._add_data_source(data_source_name, data_source)

    def _delete_data_sources(self, delete_data_sources):
        for data_source_name in delete_data_sources:
            self._delete_data_source(data_source_name)

    def _add_data_source(self, data_source_name, data_source):
        endpoint_url = "{}{}".format(
            self.grafana_endpoint, REST_API_ENDPOINT_DATA_SOURCES)
        # The response is response object with data_source in it
        # is there an exception when error?
        response_for_add = rest_api_post_json(
            endpoint_url, data_source, auth=self.auth)
        added_data_source = response_for_add.get("datasource")
        if added_data_source:
            logger.info("Data source {} created: {}".format(
                data_source_name, added_data_source))
        else:
            logger.error("Data source {} creation failed: {}".format(
                data_source_name, data_source))

    def _delete_data_source(self, data_source_name):
        # DELETE /api/datasources/name/:datasourceName
        endpoint = "{}/{}".format(
            REST_API_ENDPOINT_DATA_SOURCES_BY_NAME, data_source_name)
        endpoint_url = "{}{}".format(
            self.grafana_endpoint, endpoint)
        # is there an exception when error?
        response_for_delete = rest_api_delete_json(
            endpoint_url, auth=self.auth)
        if response_for_delete:
            logger.info("Data source {} deleted successfully.".format(
                data_source_name))
        else:
            logger.error("Data source {} deletion failed.".format(
                data_source_name))
