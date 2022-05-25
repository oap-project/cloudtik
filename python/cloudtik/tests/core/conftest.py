import pytest
import os

import yaml

EXAMPLE_PATH = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))
AWS_CLUSTER_EXAMPLE_PATHS = os.path.join(
    EXAMPLE_PATH, "example/cluster/aws/example-docker.yaml")
AWS_WORKSPACE_EXAMPLE_PATHS = os.path.join(
    EXAMPLE_PATH, "example/cluster/aws/example-workspace.yaml")


def workspace_conf_files():
    return AWS_WORKSPACE_EXAMPLE_PATHS.split(",")


def cluster_conf_files():
    return AWS_CLUSTER_EXAMPLE_PATHS.split(",")


def pytest_addoption(parser):
    parser.addoption(
        "--workspace_conf", action="store", default=AWS_WORKSPACE_EXAMPLE_PATHS
    )
    parser.addoption(
        "--cluster_conf", action="store", default=AWS_CLUSTER_EXAMPLE_PATHS
    )


def pytest_configure(config):
    global AWS_CLUSTER_EXAMPLE_PATHS
    global AWS_WORKSPACE_EXAMPLE_PATHS
    AWS_WORKSPACE_EXAMPLE_PATHS = config.getoption("--workspace_conf")
    AWS_CLUSTER_EXAMPLE_PATHS = config.getoption("--cluster_conf")

@pytest.fixture(scope="class", params=cluster_conf_files())
def cluster_config(request):
    print('setup once per each param', request.param)
    config = yaml.safe_load(open(request.param).read())
    return config

@pytest.fixture(scope="class", params=workspace_conf_files())
def workspace_config(request):
    print('setup once per each param', request.param)
    config = yaml.safe_load(open(request.param).read())
    return config