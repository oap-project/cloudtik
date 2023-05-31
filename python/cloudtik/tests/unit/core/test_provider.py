import copy

import pytest

from cloudtik.core._private.cluster.cluster_config import _bootstrap_config
from cloudtik.core._private.providers import _get_node_provider, _get_workspace_provider
from cloudtik.core._private.workspace.workspace_operator import _bootstrap_workspace_config
from cloudtik.core.node_provider import NodeProvider
from cloudtik.core.workspace_provider import WorkspaceProvider


class TestExternalNodeProvider(NodeProvider):
    def __init__(self, provider_config, cluster_name):
        NodeProvider.__init__(self, provider_config, cluster_name)


class TestExternalWorkspaceProvider(WorkspaceProvider):
    def __init__(self, provider_config, workspace_name):
        WorkspaceProvider.__init__(self, provider_config, workspace_name)


EXTERNAL_PROVIDER_CONFIG = {
    "cluster_name": "default",
    "provider": {
        "type": "external",
        "provider_class": "cloudtik.tests.unit.core.test_provider.TestExternalNodeProvider",
    },
    "available_node_types": {
        "head.default": {
            "node_config": {
                "instance_type": "test"
            }
        },
        "worker.default": {
            "node_config": {
                "instance_type": "test"
            }
        }
    },
    "head_node_type": "head.default",
    "runtime": {
        "types": []
    }
}

EXTERNAL_WORKSPACE_CONFIG = {
    "workspace_name": "default",
    "provider": {
        "type": "external",
        "provider_class": "cloudtik.tests.unit.core.test_provider.TestExternalWorkspaceProvider",
    }
}


class TestProvider:
    def test_external_node_provider(self):
        config = copy.deepcopy(EXTERNAL_PROVIDER_CONFIG)
        provider = _get_node_provider(
            config["provider"], config["cluster_name"])

        assert provider is not None

        # bootstrap should be successful
        _bootstrap_config(config, no_config_cache=True)

    def test_external_workspace_provider(self):
        config = copy.deepcopy(EXTERNAL_WORKSPACE_CONFIG)
        provider = _get_workspace_provider(
            config["provider"], config["workspace_name"])

        assert provider is not None

        # bootstrap should be successful
        _bootstrap_workspace_config(config, no_config_cache=True)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
