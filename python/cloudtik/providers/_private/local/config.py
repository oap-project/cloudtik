import os
import copy
from typing import Any
from typing import Dict

from cloudtik.core._private.cli_logger import cli_logger
import cloudtik.core._private.utils as utils

unsupported_field_message = ("The field {} is not supported "
                             "for on-premise clusters.")

LOCAL_CLUSTER_NODE_TYPE = "local.node"


def prepare_local(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare local cluster config for ingestion by cluster launcher and scaler.
    """
    config = copy.deepcopy(config)
    for field in "head_node", "available_node_types":
        if config.get(field):
            err_msg = unsupported_field_message.format(field)
            cli_logger.abort(err_msg)
    # We use a config with a single node type for on-prem clusters.
    # Resources internally detected are not overridden
    config["available_node_types"] = {
        LOCAL_CLUSTER_NODE_TYPE: {
            "node_config": {},
            "resources": {}
        }
    }
    config["head_node_type"] = LOCAL_CLUSTER_NODE_TYPE
    if "cloud_simulator_address" in config["provider"]:
        config = prepare_cloud_simulator(config)
    else:
        config = prepare_manual(config)
    return config


def prepare_cloud_simulator(config: Dict[str, Any]) -> Dict[str, Any]:
    config = copy.deepcopy(config)
    # User should explicitly set the max number of workers for the cloud simulator
    # to allocate.
    if "max_workers" not in config:
        cli_logger.abort("The field `max_workers` is required when using an "
                         "automatically managed on-premise cluster.")
    node_type = config["available_node_types"][LOCAL_CLUSTER_NODE_TYPE]
    # The cluster controller no longer uses global `min_workers`.
    # Move `min_workers` to the node_type config.
    node_type["min_workers"] = config.pop("min_workers", 0)
    node_type["max_workers"] = config["max_workers"]
    return config


def prepare_manual(config: Dict[str, Any]) -> Dict[str, Any]:
    config = copy.deepcopy(config)
    if ("worker_ips" not in config["provider"]) or (
            "head_ip" not in config["provider"]):
        cli_logger.abort("Please supply a `head_ip` and list of `worker_ips`. "
                         "Alternatively, supply a `cloud_simulator_address`.")
    num_ips = len(config["provider"]["worker_ips"])
    node_type = config["available_node_types"][LOCAL_CLUSTER_NODE_TYPE]
    # Default to keeping all provided ips in the cluster.
    config.setdefault("max_workers", num_ips)
    # The cluster controller no longer uses global `min_workers`.
    # Move `min_workers` to the node_type config.
    node_type["min_workers"] = config.pop("min_workers", num_ips)
    node_type["max_workers"] = config["max_workers"]
    return config


def get_lock_path(cluster_name: str) -> str:
    return os.path.join(utils.get_user_temp_dir(),
                        "cloudtik-local-{}.lock".format(cluster_name))


def get_state_path(cluster_name: str) -> str:
    return os.path.join(utils.get_user_temp_dir(),
                        "cloudtik-local-{}.state".format(cluster_name))


def get_cloud_simulator_lock_path() -> str:
    return os.path.join(utils.get_user_temp_dir(), "cloudtik-cloud-simulator.lock")


def get_cloud_simulator_state_path() -> str:
    return os.path.join(utils.get_user_temp_dir(), "cloudtik-cloud-simulator.state")


def bootstrap_local(config: Dict[str, Any]) -> Dict[str, Any]:
    return config
