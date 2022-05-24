import pytest
import os


WORKSPACE_CONFIG_FILES = ["/home/qyao/gitspace/gazelle_aws/cloudtik_conf/workspace-defaults-intel-proxy.yaml"]
EXAMPLE_PATH = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))
AWS_CLUSTER_EXAMPLE_PATH = os.path.join(
    EXAMPLE_PATH, "example/cluster/aws/example-docker.yaml")
AWS_WORKSPACE_EXAMPLE_PATH = os.path.join(
    EXAMPLE_PATH, "example/cluster/aws/example-workspace.yaml")


def workspace_conf():
    return AWS_WORKSPACE_EXAMPLE_PATH


def cluster_conf():
    return AWS_CLUSTER_EXAMPLE_PATH


def pytest_addoption(parser):
    parser.addoption(
        "--workspace_conf", action="store", default=AWS_WORKSPACE_EXAMPLE_PATH
    )
    parser.addoption(
        "--cluster_conf", action="store", default=AWS_CLUSTER_EXAMPLE_PATH
    )


def pytest_configure(config):
    global AWS_CLUSTER_EXAMPLE_PATH
    global AWS_WORKSPACE_EXAMPLE_PATH
    AWS_WORKSPACE_EXAMPLE_PATH = config.getoption("--workspace_conf")
    AWS_CLUSTER_EXAMPLE_PATH = config.getoption("--cluster_conf")
