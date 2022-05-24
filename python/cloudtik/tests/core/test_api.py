import pytest
from conftest import workspace_conf, cluster_conf

from cloudtik.core.api import Workspace, Cluster
import yaml


class TestWorkspace:
    @classmethod
    def setup_class(self):
        config_file = workspace_conf()
        config = yaml.safe_load(open(config_file).read())
        self.workspace = Workspace(config)

    def test_create(self):
        res = self.workspace.create()
        assert res == 0

    def test_update_firewalls(self):
        res = self.workspace.update_firewalls()
        assert res == 0

    def test_list_clusters(self):
        res = self.workspace.list_clusters()
        assert type(res) == dict

    def test_delete(self):
        res = self.workspace.delete()
        assert res == 0


class TestCluster:
    @classmethod
    def setup_class(self):
        config_file = cluster_conf()
        config = yaml.safe_load(open(config_file).read())
        self.cluster = Cluster(config)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
