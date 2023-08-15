import unittest

import pytest

from cloudtik.core._private.service_discovery.utils import SERVICE_SELECTOR_CLUSTERS, SERVICE_SELECTOR_RUNTIMES, \
    SERVICE_SELECTOR_SERVICES, SERVICE_SELECTOR_TAGS, SERVICE_SELECTOR_LABELS, SERVICE_DISCOVERY_LABEL_RUNTIME, \
    SERVICE_SELECTOR_EXCLUDE_LABELS
from cloudtik.runtime.common.service_discovery.workspace import _query_service_registry


class TestServiceDiscovery(unittest.TestCase):
    def test_query_service_registry(self):
        service_registries = {
            "cluster-1.runtime-1": "127.0.0.1:80",
            "cluster-1.runtime-2": "127.0.0.2:80",
            "cluster-2.runtime-1.service-1": "127.0.0.3:80",
            "cluster-2.runtime-3.service-1": "127.0.0.4:80",
        }
        service_selector = {
            SERVICE_SELECTOR_CLUSTERS: ["cluster-1"],
            SERVICE_SELECTOR_RUNTIMES: ["runtime-1"],
        }
        services = _query_service_registry(
            service_registries,
            service_selector,
        )
        assert len(services) == 1
        service_addresses = next(iter(services.values()))
        assert service_addresses[0][0] == "127.0.0.1"

        service_selector = {
            SERVICE_SELECTOR_CLUSTERS: ["cluster-2"],
            SERVICE_SELECTOR_RUNTIMES: ["runtime-1"],
            SERVICE_SELECTOR_SERVICES: ["service-1"],
        }
        services = _query_service_registry(
            service_registries,
            service_selector,
        )
        assert len(services) == 1
        service_addresses = next(iter(services.values()))
        assert service_addresses[0][0] == "127.0.0.3"

        service_selector = {
            SERVICE_SELECTOR_CLUSTERS: ["cluster-1"],
        }
        services = _query_service_registry(
            service_registries,
            service_selector,
        )
        assert len(services) == 2
        assert services["cluster-1.runtime-1"][0][0] == "127.0.0.1"
        assert services["cluster-1.runtime-2"][0][0] == "127.0.0.2"

        service_selector = {
            SERVICE_SELECTOR_TAGS: ["cloudtik-c-cluster-2"],
        }
        services = _query_service_registry(
            service_registries,
            service_selector,
        )
        assert len(services) == 2
        assert services["cluster-2.runtime-1.service-1"][0][0] == "127.0.0.3"
        assert services["cluster-2.runtime-3.service-1"][0][0] == "127.0.0.4"

        service_selector = {
            SERVICE_SELECTOR_LABELS: {SERVICE_DISCOVERY_LABEL_RUNTIME: "runtime-1"},
        }
        services = _query_service_registry(
            service_registries,
            service_selector,
        )
        assert len(services) == 2
        assert services["cluster-1.runtime-1"][0][0] == "127.0.0.1"
        assert services["cluster-2.runtime-1.service-1"][0][0] == "127.0.0.3"

        service_selector = {
            SERVICE_SELECTOR_EXCLUDE_LABELS: {SERVICE_DISCOVERY_LABEL_RUNTIME: "runtime-1"},
        }
        services = _query_service_registry(
            service_registries,
            service_selector,
        )
        assert len(services) == 2
        assert services["cluster-1.runtime-2"][0][0] == "127.0.0.2"
        assert services["cluster-2.runtime-3.service-1"][0][0] == "127.0.0.4"

        service_selector = {
            SERVICE_SELECTOR_CLUSTERS: ["cluster-2"],
            SERVICE_SELECTOR_RUNTIMES: ["runtime-3"],
            SERVICE_SELECTOR_SERVICES: ["service-1"],
            SERVICE_SELECTOR_TAGS: ["cloudtik-c-cluster-2"],
            SERVICE_SELECTOR_LABELS: {SERVICE_DISCOVERY_LABEL_RUNTIME: "runtime-3"},
            SERVICE_SELECTOR_EXCLUDE_LABELS: {SERVICE_DISCOVERY_LABEL_RUNTIME: "runtime-1"},
        }
        services = _query_service_registry(
            service_registries,
            service_selector,
        )
        assert len(services) == 1
        assert services["cluster-2.runtime-3.service-1"][0][0] == "127.0.0.4"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
