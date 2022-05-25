import os
import pytest

from cloudtik.core.api import Workspace, Cluster


class TestWorkspace:

    def test_create(self, workspace_config):
        # TODO The return value of each operation should be standard
        res = Workspace(workspace_config).create()

    def test_update_firewalls(self, workspace_config):
        res = Workspace(workspace_config).update_firewalls()

    def test_list_clusters(self, workspace_config):
        res = Workspace(workspace_config).list_clusters()
        assert type(res) == dict

    def test_delete(self, workspace_config):
        res = Workspace(workspace_config).delete()


class TestCluster:

    def test_start(self, cluster_config):
        res = Cluster(cluster_config).start()

    def test_wait_for_ready(self, cluster_config):
        timeout = 60 * 5
        Cluster(cluster_config).wait_for_ready(timeout=timeout)

    def test_exec(self, cluster_config):
        res = Cluster(cluster_config).exec(cmd="uptime", all_nodes=True)

    def test_rsync_up(self, cluster_config):
        res = Cluster(cluster_config).rsync(source=self.config_file, target="~/test.yaml", down=False)

    def test_rsync_down(self, cluster_config):
        tmp_file = "/tmp/cloudtik_bootstrap_config.yaml"
        res = Cluster(cluster_config).rsync(source="~/cloudtik_bootstrap_config.yaml", target=tmp_file, down=True)
        file_exist = os.path.exists(tmp_file)
        if file_exist:
            os.remove(tmp_file)
        assert file_exist

    def test_get_head_node_ip(self, cluster_config):
        res = Cluster(cluster_config).get_head_node_ip()
        assert type(res) == str

    def test_get_worker_node_ips(self, cluster_config):
        res = Cluster(cluster_config).get_worker_node_ips()
        assert type(res) == list

    def test_get_nodes(self, cluster_config):
        res = Cluster(cluster_config).get_nodes()
        assert type(res) == list

    def test_get_info(self, cluster_config):
        res = Cluster(cluster_config).get_info()
        assert type(res) == dict

    def test_stop(self, cluster_config):
        res = Cluster(cluster_config).stop()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-vx", __file__]))
