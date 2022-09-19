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
    parser.addoption(
        "--gcp_project_id", action="store", default=""
    )
    parser.addoption(
        "--azure_subscription_id", action="store", default=""
    )


@pytest.fixture(scope='session', autouse=True)
def api_conf_fixture(request):
    ssh_proxy_command = request.config.getoption("--ssh_proxy_command")
    allowed_ssh_sources = request.config.getoption("--allowed_ssh_sources")
    gcp_project_id = request.config.getoption("--gcp_project_id")
    azure_subscription_id = request.config.getoption("--azure_subscription_id")
    if ssh_proxy_command:
        pytest.ssh_proxy_command = ssh_proxy_command
    if allowed_ssh_sources:
        pytest.allowed_ssh_sources = allowed_ssh_sources.split(",")
    if azure_subscription_id:
        pytest.azure_subscription_id = azure_subscription_id
    if gcp_project_id:
        pytest.gcp_project_id = gcp_project_id


def cluster_up_down_opt(conf):
    if pytest.ssh_proxy_command:
        conf["auth"]["ssh_proxy_command"] = pytest.ssh_proxy_command
    if conf["provider"]["type"] == "azure":
        conf["provider"]["subscription_id"] = pytest.azure_subscription_id
    if conf["provider"]["type"] == "gcp":
        conf["provider"]["project_id"] = pytest.gcp_project_id
    cluster = Cluster(conf)
    try:
        print("\nStart cluster {}".format(conf["cluster_name"]))
        cluster.start()
        yield cluster
    except Exception as e:
        print("\nFailed to start cluster {}".format(conf["cluster_name"]))
        print("Exception: "+str(e))
    finally:
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
    worker_node_type = "worker.default"
    if conf["provider"]["type"] == "gcp":
        worker_node_type = "worker-default"
    conf["available_node_types"][worker_node_type]["min_workers"] = worker_nodes_fixture
    yield from cluster_up_down_opt(conf)


@pytest.fixture(scope="class")
def runtime_cluster_fixture(request):
    param = request.param
    conf_file = os.path.join(ROOT_PATH, param)
    conf = yaml.safe_load(open(conf_file).read())
    if not conf.get("docker", False):
        conf["docker"] = {}
    conf["docker"]["image"] = "cloudtik/spark-runtime-benchmark:nightly"
    if not conf.get("runtime", False):
        conf["runtime"] = {}
    conf["runtime"]["types"] = ["ganglia", "metastore", "zookeeper", "spark", "kafka", "presto"]
    yield from cluster_up_down_opt(conf)