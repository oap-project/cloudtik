import pytest

from cloudtik.tests.integration.basic_test import WorkspaceBasicTest, \
    create_workspace, ClusterFunctionTest, \
    ClusterRuntimeTest, ClusterScaleTest
from cloudtik.tests.integration.constants import \
    HUAWEICLOUD_BASIC_WORKSPACE_CONF_FILE, \
    HUAWEICLOUD_BASIC_CLUSTER_CONF_FILE, \
    WORKER_NODES_LIST

workspace = None


def setup_module():
    global workspace
    workspace = create_workspace(HUAWEICLOUD_BASIC_WORKSPACE_CONF_FILE)


def teardown_module():
    print("\nDelete Workspace")
    workspace.delete()


class TestHUAWEICLOUDWorkspaceBasic(WorkspaceBasicTest):
    def setup_class(self):
        self.workspace = workspace


@pytest.mark.parametrize(
    'basic_cluster_fixture',
    [HUAWEICLOUD_BASIC_CLUSTER_CONF_FILE],
    indirect=True
)
class TestHUAWEICLOUDClusterFunction(ClusterFunctionTest):
    """ Test cloudtik functionality on HUAWEICLOUD"""


@pytest.mark.parametrize(
    'worker_nodes_fixture',
    WORKER_NODES_LIST,
    indirect=True
)
@pytest.mark.parametrize(
    'usability_cluster_fixture',
    [HUAWEICLOUD_BASIC_CLUSTER_CONF_FILE],
    indirect=True
)
class TestHUAWEICLOUDClusterScale(ClusterScaleTest):
    """ Test cloudtik Scale Function on HUAWEICLOUD"""


@pytest.mark.parametrize(
    'runtime_cluster_fixture',
    [HUAWEICLOUD_BASIC_CLUSTER_CONF_FILE],
    indirect=True
)
class TestHUAWEICLOUDClusterRuntime(ClusterRuntimeTest):
    """ Test cloudtik runtime on HUAWEICLOUD"""


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-vsx", __file__]))
