import sys

import pytest
import os

import yaml

from cloudtik.core.api import Workspace, Cluster

ROOT_PATH = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

DEFAULT_CONF_FILE = os.path.join(os.path.dirname(__file__), "api_conf.yaml")


def pytest_configure():
    pytest.api_conf = yaml.safe_load(open(DEFAULT_CONF_FILE).read())


def pytest_addoption(parser):
    parser.addoption(
        "--api_conf", action="store", default=""
    )


@pytest.fixture(scope='session', autouse=True)
def api_conf_fixture(request):
    api_conf_file = request.config.getoption("--api_conf")
    if api_conf_file:
        pytest.api_conf = yaml.safe_load(open(api_conf_file).read())


def cluster_up_down_opt(conf):
    cluster = Cluster(conf)
    print("\nStart cluster{}".format(conf["cluster_name"]))
    cluster.start()
    yield cluster
    print("\nTeardown cluster{}".format(conf["cluster_name"]))
    cluster.stop()


def workspace_up_down_opt(conf):
    workspace = Workspace(conf)
    print("\nCreate Workspace{}".format(conf["cluster_name"]))
    workspace.create()
    yield workspace
    print("\nDelete Workspace{}".format(conf["cluster_name"]))
    workspace.delete()


@pytest.fixture(scope="class")
def basic_cluster_fixture(request):
    param = request.param
    conf_file = os.path.join(ROOT_PATH, param)
    conf = yaml.safe_load(open(conf_file).read())
    print("\nbasic_cluster_opt_fixture start{}".format(conf["cluster_name"]))
    yield from cluster_up_down_opt(conf)
    print("\nbasic_cluster_fixture Teardown cluster{}".format(conf["cluster_name"]))


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
    print("\nbasic_cluster_opt_fixture start{}".format(conf["cluster_name"]))
    yield from cluster_up_down_opt(conf)
    print("\nbasic_cluster_fixture Teardown cluster{}".format(conf["cluster_name"]))
