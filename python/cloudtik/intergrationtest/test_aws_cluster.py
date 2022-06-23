import pytest

from cloudtik.tests.core.basic_test import WorkspaceBasicTest, create_workspace, ClusterFunctionTest, \
    ClusterRuntimeTest, ClusterScaleTest
from cloudtik.intergrationtest.constants import AWS_BASIC_WORKSPACE_CONF_FILE, AWS_BASIC_CLUSTER_CONF_FILE, \
    WORKER_NODES_LIST

workspace = None


def setup_module():
    global workspace
    workspace = create_workspace(AWS_BASIC_WORKSPACE_CONF_FILE)


def teardown_module():
    print("\nDelete Workspace")
    workspace.delete()


class TestAWSWorkspaceBasic(WorkspaceBasicTest):
    def setup_class(self):
        self.workspace = workspace


@pytest.mark.parametrize(
    'basic_cluster_fixture',
    [AWS_BASIC_CLUSTER_CONF_FILE],
    indirect=True
)
class TestAWSClusterFunction(ClusterFunctionTest):
    """ Test cloudtik functionality on AWS"""


@pytest.mark.parametrize(
    'worker_nodes_fixture',
    WORKER_NODES_LIST,
    indirect=True
)
@pytest.mark.parametrize(
    'usability_cluster_fixture',
    [AWS_BASIC_CLUSTER_CONF_FILE],
    indirect=True
)
class TestAWSClusterScale(ClusterScaleTest):
    """ Test cloudtik Scale Function on AWS"""


@pytest.mark.parametrize(
    'runtime_cluster_fixture',
    [AWS_BASIC_CLUSTER_CONF_FILE],
    indirect=True
)
class TestAWSClusterRuntime(ClusterRuntimeTest):
    """ Test cloudtik runtime on AWS"""


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-vsx", __file__]))
