import os

from cloudtik.core._private.constants import CLOUDTIK_RUNTIME_ENV_NODE_TYPE, CLOUDTIK_RUNTIME_ENV_NODE_IP
from cloudtik.core._private.utils import load_head_cluster_config, _get_node_type_specific_runtime_config, \
    subscribe_runtime_config


def get_runtime_node_type():
    # Node type should always be set as env
    node_type = os.environ.get(CLOUDTIK_RUNTIME_ENV_NODE_TYPE)
    if not node_type:
        raise RuntimeError(
            "Environment variable {} is not set.".format(CLOUDTIK_RUNTIME_ENV_NODE_TYPE))

    return node_type


def get_runtime_node_ip():
    # Node type should always be set as env
    node_ip = os.environ.get(CLOUDTIK_RUNTIME_ENV_NODE_IP)
    if not node_ip:
        raise RuntimeError(
            "Environment variable {} is not set.".format(CLOUDTIK_RUNTIME_ENV_NODE_IP))

    return node_ip


def get_runtime_config_from_node(head):
    if head:
        config = load_head_cluster_config()
        node_type = config["head_node_type"]
        return _get_node_type_specific_runtime_config(config, node_type)
    else:
        # from worker node, subscribe from head redis
        return subscribe_runtime_config()
