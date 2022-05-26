import sys

import pytest
import os

import yaml

EXAMPLE_PATH = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))
CLUSTER_EXAMPLE_PATHS = os.path.join(
    EXAMPLE_PATH, "example/cluster")
AWS_CLUSTER_EXAMPLE_PATHS = os.path.join(
    CLUSTER_EXAMPLE_PATHS, "aws/example-docker.yaml")
AWS_WORKSPACE_EXAMPLE_PATHS = os.path.join(
    CLUSTER_EXAMPLE_PATHS, "aws/example-workspace.yaml")

def pytest_addoption(parser):
    parser.addoption(
        "--workspace_conf", action="store", default=AWS_WORKSPACE_EXAMPLE_PATHS
    )
    parser.addoption(
        "--cluster_conf", action="store", default=AWS_CLUSTER_EXAMPLE_PATHS
    )


def pytest_generate_tests(metafunc):
    if 'workspace_config_fixture' in metafunc.fixturenames:
        config_files = set(metafunc.config.option.workspace_conf.split(","))
        configs = []
        for config_file in config_files:
            configs.append(yaml.safe_load(open(config_file).read()))
        metafunc.parametrize("workspace_config_fixture", configs)

    if 'cluster_config_fixture' in metafunc.fixturenames:
        config_files = set(metafunc.config.option.cluster_conf.split(","))
        configs = []
        for config_file in config_files:
            configs.append(yaml.safe_load(open(config_file).read()))
        metafunc.parametrize("cluster_config_fixture", configs)
