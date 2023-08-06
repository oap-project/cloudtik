import logging

from cloudtik.core._private.core_utils import deserialize_config
from cloudtik.core._private.runtime_utils import save_yaml
from cloudtik.core._private.util.pull.pull_job import PullJob
from cloudtik.runtime.common.service_discovery.consul import query_services

logger = logging.getLogger(__name__)


def _get_service_selector_from_str(service_selector_str):
    if not service_selector_str:
        return None
    return deserialize_config(service_selector_str)


def _get_prometheus_data_source(service):
    # TODO
    name = ""  # service["name"]
    url = ""  # "http://{}:{}".format(service["ip"], service["port"])
    prometheus_data_source = {
        "name": name,
        "type": "prometheus",
        "access": "proxy",
        "url": url,
        "isDefault": False,
    }
    return prometheus_data_source


class PullDataSources(PullJob):
    """Pulling job for data sources from service discovery"""

    def __init__(self,
                 grafana_endpoint=None,
                 service_selector=None):
        if not grafana_endpoint:
            raise RuntimeError("Grafana endpoint is needed for pulling data sources.")

        self.service_selector = _get_service_selector_from_str(
            service_selector)
        self.grafana_endpoint = grafana_endpoint
        self.data_sources = None

    def pull(self):
        selected_services = self._query_services()
        data_sources = []
        for service in selected_services:
            data_source = _get_prometheus_data_source(service)
            data_sources.append(data_source)

        self._configure_data_sources(data_sources)

    def _query_services(self):
        return query_services(self.service_selector)

    def _configure_data_sources(self, data_sources):
        # 1. delete data sources was added but now exists
        # 2. add new data sources
        pass
