import pytest

from cloudtik.tests.integration.basic_test import WorkspaceBasicTest, create_workspace, ClusterFunctionTest, \
    ClusterRuntimeTest, ClusterScaleTest
from cloudtik.tests.integration.constants import ALIYUN_BASIC_WORKSPACE_CONF_FILE, ALIYUN_BASIC_CLUSTER_CONF_FILE, \
    WORKER_NODES_LIST

workspace = None


def setup_module():
    global workspace
    workspace = create_workspace(ALIYUN_BASIC_WORKSPACE_CONF_FILE)


def teardown_module():
    print("\nDelete Workspace")
    workspace.delete()


class TestAliyunWorkspaceBasic(WorkspaceBasicTest):
    def setup_class(self):
        self.workspace = workspace


@pytest.mark.parametrize(
    'basic_cluster_fixture',
    [ALIYUN_BASIC_CLUSTER_CONF_FILE],
    indirect=True
)
class TestAliyunClusterFunction(ClusterFunctionTest):
    """ Test cloudtik functionality on Alibaba Cloud"""


@pytest.mark.parametrize(
    'worker_nodes_fixture',
    WORKER_NODES_LIST,
    indirect=True
)
@pytest.mark.parametrize(
    'usability_cluster_fixture',
    [ALIYUN_BASIC_CLUSTER_CONF_FILE],
    indirect=True
)
class TestAliyunClusterScale(ClusterScaleTest):
    """ Test cloudtik Scale Function on Alibaba Cloud"""


@pytest.mark.parametrize(
    'runtime_cluster_fixture',
    [ALIYUN_BASIC_CLUSTER_CONF_FILE],
    indirect=True
)
class TestAliyunClusterRuntime(ClusterRuntimeTest):
    """ Test cloudtik runtime on Alibaba Cloud"""


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-vsx", __file__]))
