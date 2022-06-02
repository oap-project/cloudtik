import os
import pytest

CLUSTER_TIMEOUT = 60 * 5

BASIC_WORKSPACE_CONF_FILES = ["example/cluster/aws/example-workspace.yaml",
                              "example/cluster/azure/example-workspace.yaml",
                              "example/cluster/gcp/example-workspace.yaml"]
BASIC_CLUSTER_CONF_FILES = ["example/cluster/aws/example-standard.yaml",
                            "example/cluster/azure/example-standard.yaml",
                            "example/cluster/gcp/example-standard.yaml"]
AWS_BASIC_CLUSTER_CONF_FILES = ["example/cluster/aws/example-standard.yaml"]

TPCDS_BENCHMARK = {
    "script_file": "https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/spark/benchmark/scripts/tpcds-power-test.scala",
    "script_args": "--jars /home/cloudtik/runtime/benchmark-tools/spark-sql-perf/target/scala-2.12/spark-sql-perf_2.12-0.5.1-SNAPSHOT.jar"
}

TPC_DATAGEN_BENCHMARK = {
    "script_file": "https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/spark/benchmark/scripts/tpcds-datagen.scala",
    "script_args": "--jars /home/cloudtik/runtime/benchmark-tools/spark-sql-perf/target/scala-2.12/spark-sql-perf_2.12-0.5.1-SNAPSHOT.jar"
}


KAFKA_BENCHMARK = {
    "script_file": "https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/kafka/benchmark/scripts/kafka-benchmark.sh",
    "script_args": ""
}


@pytest.mark.parametrize(
    'basic_workspace_fixture',
    BASIC_WORKSPACE_CONF_FILES,
    indirect=True
)
class TestWorkspaceBasic:
    def test_update_firewalls(self, basic_workspace_fixture):
        basic_workspace_fixture.update_firewalls()

    def test_list_clusters(self, basic_workspace_fixture):
        res = basic_workspace_fixture.list_clusters()
        assert type(res) == dict


@pytest.mark.parametrize(
    'basic_cluster_fixture',
    BASIC_CLUSTER_CONF_FILES,
    indirect=True
)
class TestClusterBasic:

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


@pytest.mark.parametrize(
    'worker_nodes_fixture',
    [1, 2, 4, 6],
    indirect=True
)
@pytest.mark.parametrize(
    'usability_cluster_fixture',
    AWS_BASIC_CLUSTER_CONF_FILES,
    indirect=True
)
class TestClusterUsability:

    @pytest.mark.parametrize("num_cpus", [2, 6, 8, 14])
    def test_scale_by_cpu(self, usability_cluster_fixture, worker_nodes_fixture, num_cpus):
        usability_cluster_fixture.scale(num_cpus=num_cpus)
        usability_cluster_fixture.wait_for_ready(timeout=CLUSTER_TIMEOUT)

    @pytest.mark.parametrize("nodes", [1, 2, 4, 6])
    def test_scale_by_node(self, usability_cluster_fixture, worker_nodes_fixture, nodes):
        usability_cluster_fixture.scale(nodes=nodes)
        usability_cluster_fixture.wait_for_ready(timeout=CLUSTER_TIMEOUT)


@pytest.mark.parametrize(
    'runtime_cluster_fixture',
    AWS_BASIC_CLUSTER_CONF_FILES,
    indirect=True
)
class TestClusterRuntime:

    @pytest.mark.parametrize("benchmark", [TPC_DATAGEN_BENCHMARK, TPCDS_BENCHMARK, KAFKA_BENCHMARK])
    def test_benchmark(self, runtime_cluster_fixture, benchmark):
        script_file = benchmark["script_file"]
        script_args = benchmark["script_args"]
        runtime_cluster_fixture.submit(script_file=script_file, script_args=[script_args])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-vsx", __file__]))
