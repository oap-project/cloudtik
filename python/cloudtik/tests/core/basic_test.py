import os
import pytest
import yaml

from cloudtik.tests.core.constants import CLUSTER_TIMEOUT, \
    AWS_BASIC_CLUSTER_CONF_FILE, TPC_DATAGEN_BENCHMARK, runtime_additional_conf
from cloudtik.core.api import Workspace, Cluster

ROOT_PATH = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))


class WorkspaceBasicTest:
    def setup_class(self):
        self.workspace = Workspace({})

    def test_update_firewalls(self):
        self.workspace.update_firewalls()

    def test_list_clusters(self):
        res = self.workspace.list_clusters()
        assert type(res) == dict


class ClusterBasicTest:
    def setup_class(self):
        self.cluster = start_cluster(self.conf_file)

    def teardown_class(self):
        print("\nTeardown cluster")
        self.cluster.stop()


class ClusterFunctionTest(ClusterBasicTest):

    def test_wait_for_ready(self):
        self.cluster.wait_for_ready(timeout=CLUSTER_TIMEOUT)

    def test_rsync_down(self):
        tmp_file = "/tmp/cloudtik_bootstrap_config.yaml"
        self.cluster.rsync(source="~/cloudtik_bootstrap_config.yaml", target=tmp_file, down=True)
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
        node_infos = self.cluster.get_nodes()
        for node_info in node_infos:
            assert node_info["cloudtik-node-status"] == "up-to-date"

    def test_get_info(self):
        res = self.cluster.get_info()
        assert type(res) == dict


class ClusterRuntimeTest(ClusterBasicTest):
    def setup_class(self):
        self.cluster = start_cluster(self.conf_file, additional_conf=runtime_additional_conf)

    @pytest.mark.parametrize("benchmark", [TPC_DATAGEN_BENCHMARK])
    def test_benchmark(self, benchmark):
        script_file = benchmark["script_file"]
        script_args = benchmark["script_args"]
        cmd_output = self.cluster.submit(script_file=script_file, script_args=[script_args], with_output=True)


def load_conf(conf_file) -> dict:
    conf_file = os.path.join(ROOT_PATH, conf_file)
    conf = yaml.safe_load(open(conf_file).read())
    return conf


def create_workspace(conf_file):
    conf = load_conf(conf_file)
    workspace = Workspace(conf)
    print("\nCreate Workspace {}".format(conf["workspace_name"]))
    workspace.create()
    return workspace


def start_cluster(conf_file, additional_conf=None):
    conf = load_conf(conf_file)
    if additional_conf:
        conf.update(additional_conf)
    if pytest.ssh_proxy_command:
        conf["auth"]["ssh_proxy_command"] = pytest.ssh_proxy_command
    cluster = Cluster(conf)
    print("\nStart cluster {}".format(conf["cluster_name"]))
    cluster.start()
    cluster.wait_for_ready(timeout=CLUSTER_TIMEOUT)
    return cluster
