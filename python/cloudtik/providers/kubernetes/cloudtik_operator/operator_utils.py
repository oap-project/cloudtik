import copy
import logging
import os
import time
from typing import Any, Dict, List

from kubernetes.client.rest import ApiException

from cloudtik.core._private import constants
from cloudtik.providers._private._kubernetes import custom_objects_api
from cloudtik.providers._private._kubernetes.config import _get_cluster_selector
from cloudtik.providers._private._kubernetes.node_provider import head_service_selector
from cloudtik.core._private.utils import _get_default_config

CLOUDTIK_API_GROUP = "cloudtik.io"
CLOUDTIK_API_VERSION = "v1"
CLOUDTIK_CLUSTER_PLURAL = "cloudtikclusters"

STATUS_RECOVERING = "Recovering"
STATUS_ERROR = "Error"
STATUS_RUNNING = "Running"
STATUS_UPDATING = "Updating"

SCALER_RETRIES_FIELD = "scalerRetries"

MAX_STATUS_RETRIES = 3
DELAY_BEFORE_STATUS_RETRY = 0.5

OPERATOR_NAMESPACE = os.environ.get("CLOUDTIK_OPERATOR_POD_NAMESPACE", "")
# Operator is namespaced if the above environment variable is set,
# cluster-scoped otherwise:
NAMESPACED_OPERATOR = OPERATOR_NAMESPACE != ""

CLOUDTIK_CONFIG_DIR = os.environ.get("CLOUDTIK_CONFIG_DIR") or os.path.expanduser(
    "~/cloudtik_cluster_configs"
)

CONFIG_SUFFIX = "_config.yaml"

CONFIG_FIELDS = {
    "maxWorkers": "max_workers",
    "upscalingSpeed": "upscaling_speed",
    "idleTimeoutMinutes": "idle_timeout_minutes",
    "headPodType": "head_node_type",
    "podTypes": "available_node_types",
    "setupCommands": "setup_commands",
    "headSetupCommands": "head_setup_commands",
    "workerSetupCommands": "worker_setup_commands",
    "bootstrapCommands": "bootstrap_commands",
    "startCommands": "start_commands",
    "headStartCommands": "head_start_commands",
    "workerStartCommands": "worker_start_commands",
    "stopCommands": "stop_commands",
    "headStopCommands": "head_stop_commands",
    "workerStopCommands": "worker_stop_commands",
}

NODE_TYPE_FIELDS = {
    "minWorkers": "min_workers",
    "maxWorkers": "max_workers",
    "podConfig": "node_config",
    "customResources": "resources",
    "setupCommands": "worker_setup_commands",
    "workerSetupCommands": "worker_setup_commands",
    "bootstrapCommands": "bootstrap_commands",
    "workerStartCommands": "worker_start_commands",
    "workerStopCommands": "worker_stop_commands",
}

NODE_TYPE_NAMES = {
    "headDefault": "head.default",
    "workerDefault": "worker.default",
}

root_logger = logging.getLogger("cloudtik")
root_logger.setLevel(logging.getLevelName("DEBUG"))

logger = logging.getLogger(__name__)


def namespace_dir(namespace: str) -> str:
    """Directory in which to store configs for clusters in a given
    namespace."""
    return os.path.join(CLOUDTIK_CONFIG_DIR, namespace)


def cluster_config_path(cluster_namespace: str, cluster_name: str) -> str:
    """Where to store a cluster's config, given the cluster's name and
    namespace."""
    file_name = cluster_name + CONFIG_SUFFIX
    return config_path(cluster_namespace, file_name)


def controller_config_path(cluster_namespace: str, cluster_name: str) -> str:
    """Where to store controller cluster's config, given the cluster's name and
    namespace."""
    file_name = "cloudtik_" + cluster_name + CONFIG_SUFFIX
    return config_path(cluster_namespace, file_name)


def config_path(cluster_namespace: str, file_name: str) -> str:
    return os.path.join(namespace_dir(cluster_namespace), file_name)


def custom_resource_to_config(cluster_resource: Dict[str, Any]) -> Dict[str, Any]:
    """Convert CloudTikCluster custom resource to a cluster config."""
    config = translate(cluster_resource["spec"], dictionary=CONFIG_FIELDS)
    cluster_name = cluster_resource["metadata"]["name"]
    namespace = cluster_resource["metadata"]["namespace"]
    cluster_owner_reference = get_cluster_owner_reference(
        cluster_resource, cluster_name
    )
    config["cluster_name"] = cluster_name
    config["workspace_name"] = namespace
    config["no_controller_on_head"] = True
    config["available_node_types"] = get_node_types(
        cluster_resource, cluster_name, cluster_owner_reference
    )
    config["provider"] = get_provider_config(
        cluster_resource, cluster_name, namespace, cluster_owner_reference
    )
    config["runtime"] = get_runtime_config(
        cluster_resource
    )
    return config


def get_node_types(
    cluster_resource: Dict[str, Any],
    cluster_name: str,
    cluster_owner_reference: Dict[str, Any],
) -> Dict[str, Any]:
    node_types = {}
    for pod_type in cluster_resource["spec"]["podTypes"]:
        name = pod_type["name"]
        node_type_name = translate_pod_type_name(name)
        pod_type_copy = copy.deepcopy(pod_type)
        pod_type_copy.pop("name")
        node_type = translate(pod_type_copy, dictionary=NODE_TYPE_FIELDS)
        node_config = node_type["node_config"]
        pod = node_config["pod"]
        if "metadata" not in pod:
            pod["metadata"] = {}
        metadata = pod["metadata"]
        metadata.update({"ownerReferences": [cluster_owner_reference]})
        # Prepend cluster name:
        if "generateName" in metadata:
            metadata["generateName"] = f"cloudtik-{cluster_name}-{metadata['generateName']}"
        if name == cluster_resource["spec"]["headPodType"]:
            if "labels" not in metadata:
                metadata["labels"] = {}
        node_types[node_type_name] = node_type
    return node_types


def get_provider_config(
    cluster_resource, cluster_name, namespace, cluster_owner_reference
):
    provider_conf = {"type": "kubernetes", "use_internal_ips": True, "namespace": namespace}

    configure_services(
        provider_conf, cluster_resource,
        cluster_name, cluster_owner_reference)

    configure_cloud(provider_conf, cluster_resource)
    # Signal to autoscaler that the Operator is in use:
    provider_conf["_operator"] = True
    return provider_conf


def get_runtime_config(
    cluster_resource: Dict[str, Any],
) -> Dict[str, Any]:
    if "runtime" not in cluster_resource["spec"]:
        return {}
    return copy.deepcopy(cluster_resource["spec"]["runtime"])


def configure_services(
        provider_config: Dict[str, Any],
        cluster_resource: Dict[str, Any],
        cluster_name, cluster_owner_reference):
    head_service_ports = cluster_resource["spec"].get("headServicePorts", None)
    # Pull the default head service from
    # providers/kubernetes/defaults.yaml
    default_kubernetes_config = _get_default_config({"type": "kubernetes"})

    provider_config["head_service"] = get_head_service(
        cluster_name, cluster_owner_reference,
        head_service_ports, default_kubernetes_config)
    provider_config["head_external_service"] = get_head_external_service(
        cluster_name, cluster_owner_reference,
        default_kubernetes_config)
    provider_config["node_service"] = get_node_service(
        cluster_name, cluster_owner_reference,
        default_kubernetes_config)


def configure_cloud(
    provider_config: Dict[str, Any],
    cluster_resource: Dict[str, Any],
):
    if "cloudConfig" not in cluster_resource["spec"]:
        return
    cloud_config = cluster_resource["spec"]["cloudConfig"]
    configure_cloud_provider(provider_config, cloud_config)
    configure_cloud_storage(provider_config, cloud_config)


def configure_cloud_provider(
    provider_config: Dict[str, Any],
    cloud_config: Dict[str, Any],
):
    if "cloudProvider" not in cloud_config:
        return

    if "cloud_provider" not in provider_config:
        provider_config["cloud_provider"] = {}
    cloud_provider = provider_config["cloud_provider"]

    cloud_provider_config = cloud_config["cloudProvider"]
    for field in cloud_provider_config:
        cloud_provider[field] = copy.deepcopy(cloud_provider_config[field])


def configure_cloud_storage(
    provider_config: Dict[str, Any],
    cloud_config: Dict[str, Any],
):
    if "cloudStorage" not in cloud_config:
        return

    if "storage" not in provider_config:
        provider_config["storage"] = {}
    storage_config = provider_config["storage"]

    cloud_storage = cloud_config["cloudStorage"]
    for field in cloud_storage:
        storage_config[field] = copy.deepcopy(cloud_storage[field])


def get_head_service(
        cluster_name, cluster_owner_reference,
        head_service_ports, default_kubernetes_config):
    # Configure head service for runtimes.
    default_provider_conf = default_kubernetes_config["provider"]
    head_service = copy.deepcopy(default_provider_conf["head_service"])

    # Garbage-collect service upon cluster deletion.
    head_service["metadata"]["ownerReferences"] = [cluster_owner_reference]

    # Allows service to access the head pod.
    # The corresponding label is set on the head pod in
    # KubernetesNodeProvider.create_node().
    head_service["spec"]["selector"] = head_service_selector(cluster_name)

    # Configure custom ports if provided by the user.
    if head_service_ports:
        user_port_dict = port_list_to_dict(head_service_ports)
        default_port_dict = port_list_to_dict(head_service["spec"]["ports"])
        # Update default ports with user specified ones.
        default_port_dict.update(user_port_dict)
        updated_port_list = port_dict_to_list(default_port_dict)
        head_service["spec"]["ports"] = updated_port_list

    return head_service


def get_head_external_service(
        cluster_name, cluster_owner_reference,
        default_kubernetes_config):
    # Configure head external service for SSH access.
    default_provider_conf = default_kubernetes_config["provider"]
    head_external_service = copy.deepcopy(default_provider_conf["head_external_service"])

    # Garbage-collect service upon cluster deletion.
    head_external_service["metadata"]["ownerReferences"] = [cluster_owner_reference]
    return head_external_service


def get_node_service(
        cluster_name, cluster_owner_reference,
        default_kubernetes_config):
    # Configure node service for dns
    default_provider_conf = default_kubernetes_config["provider"]
    node_service = copy.deepcopy(default_provider_conf["node_service"])

    # Configure the service's name
    service_name = f"{cluster_name}"
    node_service["metadata"]["name"] = service_name

    # Garbage-collect service upon cluster deletion.
    node_service["metadata"]["ownerReferences"] = [cluster_owner_reference]
    node_service["spec"]["selector"] = {"cluster": _get_cluster_selector(cluster_name)}

    return node_service


def port_list_to_dict(port_list: List[Dict]) -> Dict:
    """Converts a list of ports with 'name' entries to a dict with name keys.

    Convenience method used when updating default head service ports with user
    specified ports.
    """
    out_dict = {}
    for item in port_list:
        value = copy.deepcopy(item)
        key = value.pop("name")
        out_dict[key] = value
    return out_dict


def port_dict_to_list(port_dict: Dict) -> List[Dict]:
    """Inverse of port_list_to_dict."""
    out_list = []
    for key, value in port_dict.items():
        out_list.append({"name": key, **value})
    return out_list


def get_cluster_owner_reference(
    cluster_resource: Dict[str, Any], cluster_name: str
) -> Dict[str, Any]:
    return {
        "apiVersion": cluster_resource["apiVersion"],
        "kind": cluster_resource["kind"],
        "blockOwnerDeletion": True,
        "controller": True,
        "name": cluster_name,
        "uid": cluster_resource["metadata"]["uid"],
    }


def translate(
    configuration: Dict[str, Any], dictionary: Dict[str, str]
) -> Dict[str, Any]:
    return {
        dictionary[field]: configuration[field]
        for field in dictionary
        if field in configuration
    }


def translate_pod_type_name(pod_type_name: str):
    node_type_name = NODE_TYPE_NAMES.get(pod_type_name)
    if node_type_name is not None:
        return node_type_name
    return pod_type_name


def set_status(cluster_name: str, cluster_namespace: str, status: str) -> None:
    """Sets status.phase field for a CloudTikCluster with the given name and
    namespace.

    Just in case, handles the 409 error that would arise if the CloudTikCluster
    API object is modified between retrieval and patch.

    Args:
        cluster_name: Name of the cluster.
        cluster_namespace: Namespace in which the cluster is running.
        status: String to set for the CloudTikCluster object's status.phase field.

    """
    for _ in range(MAX_STATUS_RETRIES - 1):
        try:
            _set_status(cluster_name, cluster_namespace, status)
            return
        except ApiException as e:
            if e.status == 409:
                logger.info(
                    "Caught a 409 error while setting CloudTikCluster status. Retrying..."
                )
                time.sleep(DELAY_BEFORE_STATUS_RETRY)
                continue
            else:
                raise
    # One more try
    _set_status(cluster_name, cluster_namespace, status)


def _set_status(cluster_name: str, cluster_namespace: str, phase: str) -> None:
    cluster_cr = custom_objects_api().get_namespaced_custom_object(
        namespace=cluster_namespace,
        group=CLOUDTIK_API_GROUP,
        version=CLOUDTIK_API_VERSION,
        plural=CLOUDTIK_CLUSTER_PLURAL,
        name=cluster_name,
    )
    status = cluster_cr.get("status", {})
    scaler_retries = status.get(SCALER_RETRIES_FIELD, 0)
    if phase == STATUS_RECOVERING:
        scaler_retries += 1
    cluster_cr["status"] = {
        "phase": phase,
        SCALER_RETRIES_FIELD: scaler_retries,
    }
    custom_objects_api().patch_namespaced_custom_object_status(
        namespace=cluster_namespace,
        group=CLOUDTIK_API_GROUP,
        version=CLOUDTIK_API_VERSION,
        plural=CLOUDTIK_CLUSTER_PLURAL,
        name=cluster_name,
        body=cluster_cr,
    )


def infer_head_port(cluster_config: Dict[str, Any]) -> str:
    """Infer head redis port. If no port argument is provided, return the default port.
    The port is used by the Operator to initialize the controller.

    Args:
        cluster_config: The cluster config dict

    Returns:
        The head redis port.

    """
    return str(constants.CLOUDTIK_DEFAULT_PORT)

