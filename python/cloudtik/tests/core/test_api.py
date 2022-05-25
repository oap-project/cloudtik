import os
import yaml
import pytest

from conftest import workspace_conf, cluster_conf

from cloudtik.core.api import Workspace, Cluster


class TestWorkspace:
    @classmethod
    def setup_class(self):
        config_file = workspace_conf()
        config = yaml.safe_load(open(config_file).read())
        self.workspace = Workspace(config)

    def test_create(self):
        res = self.workspace.create()
        # TODO The return value of each operation should be standard
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
        self.config_file = cluster_conf()
        config = yaml.safe_load(open(self.config_file).read())
        self.cluster = Cluster(config)

    def test_start(self):
        res = self.cluster.start()

    def test_wait_for_ready(self):
        timeout = 60 * 5
        self.cluster.wait_for_ready(timeout=timeout)

    def test_exec(self):
        res = self.cluster.exec(cmd="uptime", all_nodes=True)

    def test_rsync_up(self):
        res = self.cluster.rsync(source=self.config_file, target="~/test.yaml")

    def test_rsync_down(self):
        tmp_file = "/tmp/cloudtik_bootstrap_config.yaml"
        res = self.cluster.rsync(source="~/cloudtik_bootstrap_config.yaml", target=tmp_file, down=True)
        file_exist = os.path.exists(tmp_file)
        if file_exist:
            os.remove(tmp_file)
        assert file_exist

    def test_get_head_node_ip(self):
        res = self.cluster.get_head_node_ip()
        assert type(res) == str

    def test_get_worker_node_ips(self):
        res = self.cluster.get_worker_node_ips()
        assert type(res) == list

    def test_get_nodes(self):
        res = self.cluster.get_nodes()
        assert type(res) == list

    def test_get_info(self):
        res = self.cluster.get_info()
        assert type(res) == dict

    def test_stop(self):
        res = self.cluster.stop()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-vx", __file__]))
