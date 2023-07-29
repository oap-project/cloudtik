import json
import os

from cloudtik.core._private.constants import CLOUDTIK_RUNTIME_ENV_NODE_TYPE, CLOUDTIK_RUNTIME_ENV_NODE_IP, \
    CLOUDTIK_RUNTIME_ENV_SECRETS
from cloudtik.core._private.crypto import AESCipher
from cloudtik.core._private.utils import load_head_cluster_config, _get_node_type_specific_runtime_config, \
    get_runtime_config_key, _get_key_from_kv, decode_cluster_secrets, CLOUDTIK_CLUSTER_NODES_INFO_NODE_TYPE


RUNTIME_NODE_ID = "node_id"
RUNTIME_NODE_IP = "node_ip"
RUNTIME_NODE_SEQ_ID = "node_seq_id"


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


def retrieve_runtime_config(node_type: str = None):
    # Retrieve the runtime config
    runtime_config_key = get_runtime_config_key(node_type)
    encrypted_runtime_config = _get_key_from_kv(runtime_config_key)
    if encrypted_runtime_config is None:
        return None

    # Decrypt
    encoded_secrets = os.environ[CLOUDTIK_RUNTIME_ENV_SECRETS]
    secrets = decode_cluster_secrets(encoded_secrets)
    cipher = AESCipher(secrets)
    runtime_config_str = cipher.decrypt(encrypted_runtime_config)

    # To json object
    if runtime_config_str == "":
        return {}

    return json.loads(runtime_config_str)


def subscribe_runtime_config():
    if CLOUDTIK_RUNTIME_ENV_SECRETS not in os.environ:
        raise RuntimeError("Not able to subscribe runtime config in lack of secrets.")

    node_type = os.environ.get(CLOUDTIK_RUNTIME_ENV_NODE_TYPE)
    if node_type:
        # Try getting node type specific runtime config
        runtime_config = retrieve_runtime_config(node_type)
        if runtime_config is not None:
            return runtime_config

    return retrieve_runtime_config()


def get_runtime_config_from_node(head):
    if head:
        config = load_head_cluster_config()
        node_type = config["head_node_type"]
        return _get_node_type_specific_runtime_config(config, node_type)
    else:
        # from worker node, subscribe from head redis
        return subscribe_runtime_config()


def subscribe_nodes_info():
    if CLOUDTIK_RUNTIME_ENV_NODE_TYPE not in os.environ:
        raise RuntimeError("Not able to subscribe nodes info in lack of node type.")
    node_type = os.environ[CLOUDTIK_RUNTIME_ENV_NODE_TYPE]
    return _retrieve_nodes_info(node_type)


def _retrieve_nodes_info(node_type):
    nodes_info_key = CLOUDTIK_CLUSTER_NODES_INFO_NODE_TYPE.format(node_type)
    nodes_info_str = _get_key_from_kv(nodes_info_key)
    if nodes_info_str is None:
        return None

    return json.loads(nodes_info_str)
