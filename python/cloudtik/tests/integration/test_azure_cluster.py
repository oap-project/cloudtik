import pytest

from cloudtik.tests.integration.basic_test import WorkspaceBasicTest, create_workspace, ClusterFunctionTest, \
    ClusterRuntimeTest, ClusterScaleTest
from cloudtik.tests.integration.constants import AZURE_BASIC_WORKSPACE_CONF_FILE, AZURE_BASIC_CLUSTER_CONF_FILE, \
    WORKER_NODES_LIST

workspace = None


def setup_module():
    global workspace
    workspace = create_workspace(AZURE_BASIC_WORKSPACE_CONF_FILE)


def teardown_module():
    print("\nDelete Workspace")
    workspace.delete()


class TestAZUREWorkspaceBasic(WorkspaceBasicTest):
    def setup_class(self):
        self.workspace = workspace


@pytest.mark.parametrize(
    'basic_cluster_fixture',
    [AZURE_BASIC_CLUSTER_CONF_FILE],
    indirect=True
)
class TestAZUREClusterFunction(ClusterFunctionTest):
    """ Test cloudtik functionality on AZURE"""


@pytest.mark.parametrize(
    'worker_nodes_fixture',
    WORKER_NODES_LIST,
    indirect=True
)
@pytest.mark.parametrize(
    'usability_cluster_fixture',
    [AZURE_BASIC_CLUSTER_CONF_FILE],
    indirect=True
)
class TestAZUREClusterScale(ClusterScaleTest):
    """ Test cloudtik Scale Function on AZURE"""


@pytest.mark.parametrize(
    'runtime_cluster_fixture',
    [AZURE_BASIC_CLUSTER_CONF_FILE],
    indirect=True
)
class TestAZUREClusterRuntime(ClusterRuntimeTest):
    """ Test cloudtik runtime on AZURE"""


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-vsx", __file__]))
