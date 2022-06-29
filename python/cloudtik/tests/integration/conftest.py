import pytest
import os

import yaml

from cloudtik.core.api import Cluster
from cloudtik.tests.integration.basic_test import ROOT_PATH


def pytest_configure():
    pytest.ssh_proxy_command = ""
    pytest.allowed_ssh_sources = ""


def pytest_addoption(parser):
    parser.addoption(
        "--ssh_proxy_command", action="store", default=""
    )
    parser.addoption(
        "--allowed_ssh_sources", action="store", default=""
    )


@pytest.fixture(scope='session', autouse=True)
def api_conf_fixture(request):
    ssh_proxy_command = request.config.getoption("--ssh_proxy_command")
    allowed_ssh_sources = request.config.getoption("--allowed_ssh_sources")
    if ssh_proxy_command:
        pytest.ssh_proxy_command = ssh_proxy_command
    if allowed_ssh_sources:
        pytest.allowed_ssh_sources = allowed_ssh_sources.split(",")


def cluster_up_down_opt(conf):
    if pytest.ssh_proxy_command:
        conf["auth"]["ssh_proxy_command"] = pytest.ssh_proxy_command
    cluster = Cluster(conf)
    print("\nStart cluster {}".format(conf["cluster_name"]))
    cluster.start()
    yield cluster
    print("\nTeardown cluster {}".format(conf["cluster_name"]))
    cluster.stop()


@pytest.fixture(scope="class")
def basic_cluster_fixture(request):
    param = request.param
    conf_file = os.path.join(ROOT_PATH, param)
    conf = yaml.safe_load(open(conf_file).read())
    yield from cluster_up_down_opt(conf)


@pytest.fixture(scope="class")
def worker_nodes_fixture(request):
    param = request.param
    return param


@pytest.fixture(scope="class")
def usability_cluster_fixture(request, worker_nodes_fixture):
    param = request.param
    conf_file = os.path.join(ROOT_PATH, param)
    conf = yaml.safe_load(open(conf_file).read())
    conf["available_node_types"]["worker.default"]["min_workers"] = worker_nodes_fixture
    yield from cluster_up_down_opt(conf)


@pytest.fixture(scope="class")
def runtime_cluster_fixture(request):
    param = request.param
    conf_file = os.path.join(ROOT_PATH, param)
    conf = yaml.safe_load(open(conf_file).read())
    conf["setup_commands"] = ["wget -P ~/ https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/benchmarks/spark" \
                             "/scripts/bootstrap-benchmark.sh &&bash ~/bootstrap-benchmark.sh  --tpcds "]
    if not conf.get("runtime", False):
        conf["runtime"] = {}
    conf["runtime"]["types"] = ["ganglia", "metastore", "zookeeper", "spark", "kafka"]
    yield from cluster_up_down_opt(conf)