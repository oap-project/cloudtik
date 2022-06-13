import pytest

from cloudtik.tests.core.basic_test import WorkspaceBasicTest, create_workspace, ClusterFunctionTest, \
    ClusterRuntimeTest
from cloudtik.tests.core.constants import AZURE_BASIC_WORKSPACE_CONF_FILE, AZURE_BASIC_CLUSTER_CONF_FILE, \
    CLUSTER_TIMEOUT, SCALE_NODES_LIST, SCALE_CPUS_LIST, WORKER_NODES_LIST

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


class TestAZUREClusterFunction(ClusterFunctionTest):
    def setup_class(self):
        self.conf_file = AZURE_BASIC_CLUSTER_CONF_FILE
        super().setup_class(self)


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
class TestAZUREClusterUsability:

    @pytest.mark.parametrize("scale_cpus", SCALE_CPUS_LIST)
    def test_scale_by_cpu(self, usability_cluster_fixture, worker_nodes_fixture, scale_cpus):
        usability_cluster_fixture.scale(num_cpus=scale_cpus)
        usability_cluster_fixture.wait_for_ready(timeout=CLUSTER_TIMEOUT)

    @pytest.mark.parametrize("scale_nodes", SCALE_NODES_LIST)
    def test_scale_by_node(self, usability_cluster_fixture, worker_nodes_fixture, scale_nodes):
        usability_cluster_fixture.scale(nodes=scale_nodes)
        usability_cluster_fixture.wait_for_ready(timeout=CLUSTER_TIMEOUT)


class TestAZUREClusterRuntime(ClusterRuntimeTest):
    def setup_class(self):
        self.conf_file = AZURE_BASIC_CLUSTER_CONF_FILE
        super().setup_class(self)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-vsx", __file__]))
