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
    if ("worker_nodes" not in config["provider"]) or (
            "head_node" not in config["provider"]):
        cli_logger.abort("Please supply a `head_node` and list of `worker_nodes`. "
                         "Alternatively, supply a `cloud_simulator_address`.")
    num_workers = len(get_worker_nodes(config["provider"]))
    node_type = config["available_node_types"][LOCAL_CLUSTER_NODE_TYPE]
    # Default to keeping all provided ips in the cluster.
    config.setdefault("max_workers", num_workers)
    # The cluster controller no longer uses global `min_workers`.
    # Move `min_workers` to the node_type config.
    node_type["min_workers"] = config.pop("min_workers", num_workers)
    node_type["max_workers"] = config["max_workers"]

    # Set node type resource from head node
    head_node = get_head_node(config["provider"])
    resources = head_node.get("resources")
    if resources is None:
        cli_logger.warning("Node resources not provided. "
                           "Please supply the resources (CPU and memory) information in head node.")

    # default to a conservative 4 cpu and 8GB if not defined
    cpus = resources.get("CPU", 4)
    memory = resources.get("memory", 1024 * 8)
    node_type["resources"]["CPU"] = cpus
    node_type["resources"]["memory"] = int(memory) * 1024 * 1024
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


def get_head_node(provider_config: Dict[str, Any]):
    return provider_config["head_node"]


def get_head_node_ip(provider_config: Dict[str, Any]):
    head_node = get_head_node(provider_config)
    return head_node["ip"]


def get_head_node_external_ip(provider_config: Dict[str, Any]):
    head_node = get_head_node(provider_config)
    return head_node.get("external_ip")


def get_worker_nodes(provider_config: Dict[str, Any]):
    return provider_config.get("worker_nodes", [])


def get_worker_node_ips(provider_config: Dict[str, Any]):
    worker_nodes = get_worker_nodes(provider_config)
    return [worker_node["ip"] for worker_node in worker_nodes]
