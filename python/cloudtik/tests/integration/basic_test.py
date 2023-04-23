import os
import pytest
import yaml

from cloudtik.tests.integration.constants import CLUSTER_TIMEOUT, \
    TPC_DATAGEN_BENCHMARK, SCALE_CPUS_LIST, SCALE_NODES_LIST, TPCDS_BENCHMARK, KAFKA_BENCHMARK, PRESTO_BENCHMARK
from cloudtik.core.api import Workspace

ROOT_PATH = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))


class WorkspaceBasicTest:
    def setup_class(self):
        self.workspace = Workspace({})

    def test_list_clusters(self):
        res = self.workspace.list_clusters()
        assert type(res) == dict


class ClusterFunctionTest:
    def test_wait_for_ready(self, basic_cluster_fixture):
        basic_cluster_fixture.wait_for_ready(timeout=CLUSTER_TIMEOUT)

    def test_rsync_down(self, basic_cluster_fixture):
        tmp_file = "/tmp/cloudtik_bootstrap_config.yaml"
        basic_cluster_fixture.rsync(source="~/cloudtik_bootstrap_config.yaml", target=tmp_file, down=True)
        file_exist = os.path.exists(tmp_file)
        if file_exist:
            os.remove(tmp_file)
        assert file_exist

    def test_get_head_node_ip(self, basic_cluster_fixture):
        res = basic_cluster_fixture.get_head_node_ip()
        assert type(res) == str

    def test_get_worker_node_ips(self, basic_cluster_fixture):
        res = basic_cluster_fixture.get_worker_node_ips()
        assert type(res) == list

    def test_get_nodes(self, basic_cluster_fixture):
        node_infos = basic_cluster_fixture.get_nodes()
        for node_info in node_infos:
            assert node_info["cloudtik-node-status"] == "up-to-date"

    def test_get_info(self, basic_cluster_fixture):
        res = basic_cluster_fixture.get_info()
        assert type(res) == dict


class ClusterRuntimeTest:

    @pytest.mark.parametrize("benchmark", [TPC_DATAGEN_BENCHMARK, TPCDS_BENCHMARK, KAFKA_BENCHMARK, PRESTO_BENCHMARK])
    def test_benchmark(self, runtime_cluster_fixture, benchmark):
        script_file = benchmark["script_file"]
        script_args = benchmark["script_args"]
        cmd_output = runtime_cluster_fixture.submit(script_file=script_file, script_args=[script_args],
                                                    with_output=True)
        print(cmd_output)


class ClusterScaleTest:

    @pytest.mark.parametrize("scale_cpus", SCALE_CPUS_LIST)
    def test_scale_by_cpu(self, usability_cluster_fixture, worker_nodes_fixture, scale_cpus):
        usability_cluster_fixture.scale(num_cpus=scale_cpus)
        usability_cluster_fixture.wait_for_ready(timeout=CLUSTER_TIMEOUT)

    @pytest.mark.parametrize("scale_nodes", SCALE_NODES_LIST)
    def test_scale_by_node(self, usability_cluster_fixture, worker_nodes_fixture, scale_nodes):
        usability_cluster_fixture.scale(workers=scale_nodes)
        usability_cluster_fixture.wait_for_ready(timeout=CLUSTER_TIMEOUT)


def load_conf(conf_file) -> dict:
    conf_file = os.path.join(ROOT_PATH, conf_file)
    conf = yaml.safe_load(open(conf_file).read())
    return conf


def create_workspace(conf_file):
    conf = load_conf(conf_file)
    if pytest.allowed_ssh_sources:
        conf["provider"]["allowed_ssh_sources"] = pytest.allowed_ssh_sources
    if conf["provider"]["type"] == "azure":
        conf["provider"]["subscription_id"] = pytest.azure_subscription_id
    if conf["provider"]["type"] == "gcp":
        conf["provider"]["project_id"] = pytest.gcp_project_id
    workspace = Workspace(conf)
    print("\nCreate Workspace {}".format(conf["workspace_name"]))
    workspace.create()
    return workspace
