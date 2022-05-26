import os
import pytest

from cloudtik.core.api import Workspace, Cluster


class TestWorkspace:

    def test_create(self, workspace_config_fixture):
        # TODO The return value of each operation should be standard
        res = Workspace(workspace_config_fixture).create()

    def test_update_firewalls(self, workspace_config_fixture):
        res = Workspace(workspace_config_fixture).update_firewalls()

    def test_list_clusters(self, workspace_config_fixture):
        res = Workspace(workspace_config_fixture).list_clusters()
        assert type(res) == dict

    def test_delete(self, workspace_config_fixture):
        res = Workspace(workspace_config_fixture).delete()


class TestCluster:

    def test_start(self, cluster_config_fixture):
        Cluster(cluster_config_fixture).start()

    def test_wait_for_ready(self, cluster_config_fixture):
        timeout = 60 * 5
        Cluster(cluster_config_fixture).wait_for_ready(timeout=timeout)

    def test_rsync_down(self, cluster_config_fixture):
        tmp_file = "/tmp/cloudtik_bootstrap_config.yaml"
        Cluster(cluster_config_fixture).rsync(source="~/cloudtik_bootstrap_config.yaml", target=tmp_file, down=True)
        file_exist = os.path.exists(tmp_file)
        if file_exist:
            os.remove(tmp_file)
        assert file_exist

    def test_get_head_node_ip(self, cluster_config_fixture):
        res = Cluster(cluster_config_fixture).get_head_node_ip()
        assert type(res) == str

    def test_get_worker_node_ips(self, cluster_config_fixture):
        res = Cluster(cluster_config_fixture).get_worker_node_ips()
        assert type(res) == list

    def test_get_nodes(self, cluster_config_fixture):
        node_infos = Cluster(cluster_config_fixture).get_nodes()
        for node_info in node_infos:
            assert node_info["cloudtik-node-status"] == "up-to-date"

    def test_get_info(self, cluster_config_fixture):
        res = Cluster(cluster_config_fixture).get_info()
        assert type(res) == dict

    def test_stop(self, cluster_config_fixture):
        Cluster(cluster_config_fixture).stop()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-vx", __file__]))
