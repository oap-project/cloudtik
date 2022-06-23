import pytest

from cloudtik.tests.integration.basic_test import WorkspaceBasicTest, create_workspace, ClusterFunctionTest, \
    ClusterRuntimeTest, ClusterScaleTest
from cloudtik.tests.integration.constants import GCP_BASIC_WORKSPACE_CONF_FILE, GCP_BASIC_CLUSTER_CONF_FILE, \
    WORKER_NODES_LIST

workspace = None


def setup_module():
    global workspace
    workspace = create_workspace(GCP_BASIC_WORKSPACE_CONF_FILE)


def teardown_module():
    print("\nDelete Workspace")
    workspace.delete()


class TestGCPWorkspaceBasic(WorkspaceBasicTest):
    def setup_class(self):
        self.workspace = workspace


@pytest.mark.parametrize(
    'basic_cluster_fixture',
    [GCP_BASIC_CLUSTER_CONF_FILE],
    indirect=True
)
class TestGCPClusterFunction(ClusterFunctionTest):
    """ Test cloudtik functionality on GCP"""


@pytest.mark.parametrize(
    'worker_nodes_fixture',
    WORKER_NODES_LIST,
    indirect=True
)
@pytest.mark.parametrize(
    'usability_cluster_fixture',
    [GCP_BASIC_CLUSTER_CONF_FILE],
    indirect=True
)
class TestGCPClusterScale(ClusterScaleTest):
    """ Test cloudtik Scale Function on GCP"""


@pytest.mark.parametrize(
    'runtime_cluster_fixture',
    [GCP_BASIC_CLUSTER_CONF_FILE],
    indirect=True
)
class TestGCPClusterRuntime(ClusterRuntimeTest):
    """ Test cloudtik runtime on GCP"""


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-vsx", __file__]))
