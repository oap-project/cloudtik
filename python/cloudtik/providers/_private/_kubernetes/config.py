import copy
import logging
import math
import os
import re
import time
from typing import Any, Dict, Optional
import random
import string

from kubernetes import client
from kubernetes.client.rest import ApiException

from cloudtik.core._private.cli_logger import cli_logger, cf
from cloudtik.core._private.providers import _get_node_provider
from cloudtik.core._private.utils import is_use_internal_ip, get_running_head_node, binary_to_hex, hex_to_binary, \
    get_runtime_service_ports, _is_use_managed_cloud_storage, _is_use_internal_ip
from cloudtik.core.tags import CLOUDTIK_TAG_CLUSTER_NAME, CLOUDTIK_TAG_NODE_KIND, NODE_KIND_HEAD, \
    CLOUDTIK_GLOBAL_VARIABLE_KEY, CLOUDTIK_GLOBAL_VARIABLE_KEY_PREFIX
from cloudtik.core.workspace_provider import Existence
from cloudtik.providers._private._kubernetes import auth_api, core_api, log_prefix
from cloudtik.core._private.constants import CLOUDTIK_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION, \
    CLOUDTIK_KUBERNETES_SSH_DEFAULT_PORT
from cloudtik.providers._private._kubernetes.utils import _get_node_info, to_label_selector, \
    KUBERNETES_WORKSPACE_NAME_MAX, check_kubernetes_name_format, _get_head_service_account_name, \
    _get_worker_service_account_name, KUBERNETES_HEAD_SERVICE_ACCOUNT_CONFIG_KEY, \
    KUBERNETES_WORKER_SERVICE_ACCOUNT_CONFIG_KEY, _get_service_account, \
    cleanup_orphan_pvcs, get_key_pair_path_for_kubernetes, KUBERNETES_HEAD_SERVICE_CONFIG_KEY, \
    KUBERNETES_HEAD_EXTERNAL_SERVICE_CONFIG_KEY, KUBERNETES_NODE_SERVICE_CONFIG_KEY

logger = logging.getLogger(__name__)

MEMORY_SIZE_UNITS = {
    "K": 2**10,
    "M": 2**20,
    "G": 2**30,
    "T": 2**40,
    "P": 2**50
}

CLOUDTIK_COMPONENT_LABEL = "cluster.cloudtik.io/component"
CLOUDTIK_HEAD_POD_NAME_PREFIX = "cloudtik-{}-head-"
CLOUDTIK_WORKER_POD_NAME_PREFIX = "cloudtik-{}-worker-"

CLOUDTIK_HEAD_POD_LABEL = "cloudtik-{}-head"
CLOUDTIK_WORKER_POD_LABEL = "cloudtik-{}-worker"

CLOUDTIK_HEAD_SERVICE_NAME_FORMAT = "cloudtik-{}-head"
CLOUDTIK_HEAD_EXTERNAL_SERVICE_NAME_FORMAT = "cloudtik-{}-external"
CLOUDTIK_NODE_SERVICE_NAME_FORMAT = "{}"

CLOUDTIK_CLUSTER_POD_LABEL = "cloudtik-{}"

CLOUDTIK_HEAD_HOSTNAME = "head-{}"
CLOUDTIK_WORKER_HOSTNAME = "worker-{}"

CONFIG_NAME_IMAGE = "image"

KUBERNETES_HEAD_ROLE_NAME = "cloudtik-role"
KUBERNETES_HEAD_ROLE_BINDING_NAME = "cloudtik-role-binding"

KUBERNETES_NAMESPACE = "kubernetes.namespace"
KUBERNETES_HEAD_SERVICE_ACCOUNT = "kubernetes.head.service_account"
KUBERNETES_WORKER_SERVICE_ACCOUNT = "kubernetes.worker.service_account"
KUBERNETES_HEAD_ROLE = "kubernetes.head.role"
KUBERNETES_HEAD_ROLE_BINDING = "kubernetes.head.role_binding"

KUBERNETES_HEAD_ROLE_CONFIG_KEY = "head_role"
KUBERNETES_HEAD_ROLE_BINDING_CONFIG_KEY = "head_role_binding"

KUBERNETES_WORKSPACE_NUM_CREATION_STEPS = 5
KUBERNETES_WORKSPACE_NUM_DELETION_STEPS = 5
KUBERNETES_WORKSPACE_TARGET_RESOURCES = 5

KUBERNETES_RESOURCE_OP_MAX_POLLS = 12
KUBERNETES_RESOURCE_OP_POLL_INTERVAL = 5

KUBERNETES_CONTAINER_SSH_PORT_SPEC = {'containerPort': 22, 'name': 'cloudtik-ssh'}

KUBERNETES_HEAD_EXTERNAL_SERVICE_SSH_PORT_SPEC = {
    "name": "cloudtik-ssh-port",
    "protocol": "TCP",
    "port": CLOUDTIK_KUBERNETES_SSH_DEFAULT_PORT,
    "targetPort": "cloudtik-ssh"
}


def head_service_selector(cluster_name: str) -> Dict[str, str]:
    """Selector for Operator-configured head service.
    """
    return {CLOUDTIK_COMPONENT_LABEL: CLOUDTIK_HEAD_POD_LABEL.format(cluster_name)}


def _get_head_service_name(cluster_name):
    return CLOUDTIK_HEAD_SERVICE_NAME_FORMAT.format(cluster_name)


def _get_head_external_service_name(cluster_name):
    return CLOUDTIK_HEAD_EXTERNAL_SERVICE_NAME_FORMAT.format(cluster_name)


def _get_node_service_name(cluster_name):
    return CLOUDTIK_NODE_SERVICE_NAME_FORMAT.format(cluster_name)


def _get_head_component_selector(cluster_name):
    return CLOUDTIK_HEAD_POD_LABEL.format(cluster_name)


def _get_worker_component_selector(cluster_name):
    return CLOUDTIK_WORKER_POD_LABEL.format(cluster_name)


def _get_cluster_selector(cluster_name):
    return CLOUDTIK_CLUSTER_POD_LABEL.format(cluster_name)


def get_random_hostname(hostname_format):
    suffix = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))
    return hostname_format.format(suffix)


def get_head_hostname():
    return get_random_hostname(CLOUDTIK_HEAD_HOSTNAME)


def get_worker_hostname():
    return get_random_hostname(CLOUDTIK_WORKER_HOSTNAME)


def _add_service_name_to_service_port(spec, svc_name):
    """Goes recursively through the ingress manifest and adds the
    right serviceName next to every servicePort definition.
    """
    if isinstance(spec, dict):
        dict_keys = list(spec.keys())
        for k in dict_keys:
            spec[k] = _add_service_name_to_service_port(spec[k], svc_name)

            if k == "serviceName" and spec[k] != svc_name:
                raise ValueError(
                    "The value of serviceName must be set to "
                    "${CLOUDTIK_POD_NAME}. It is automatically replaced "
                    "when using the scaler.")

    elif isinstance(spec, list):
        spec = [
            _add_service_name_to_service_port(item, svc_name) for item in spec
        ]

    elif isinstance(spec, str):
        # The magic string ${CLOUDTIK_POD_NAME} is replaced with
        # the true service name, which is equal to the worker pod name.
        if "${CLOUDTIK_POD_NAME}" in spec:
            spec = spec.replace("${CLOUDTIK_POD_NAME}", svc_name)
    return spec


class InvalidNamespaceError(ValueError):
    def __init__(self, field_name, namespace):
        self.message = ("Namespace of {} config doesn't match provided "
                        "namespace '{}'. Either set it to {} or remove the "
                        "field".format(field_name, namespace, namespace))

    def __str__(self):
        return self.message


def using_existing_msg(resource_type, name):
    return "Using existing {} '{}'".format(resource_type, name)


def updating_existing_msg(resource_type, name):
    return "Updating existing {} '{}'".format(resource_type, name)


def not_found_msg(resource_type, name):
    return "{} '{}' not found, attempting to create it".format(
        resource_type, name)


def not_checking_msg(resource_type, name):
    return "Not checking if {} '{}' exists".format(resource_type, name)


def creating_msg(resource_type, name):
    return "Creating {} '{}'...".format(
        resource_type, name)


def created_msg(resource_type, name):
    return "Successfully created {} '{}'".format(resource_type, name)


def not_provided_msg(resource_type):
    return "No {} config provided, must already exist".format(resource_type)


def get_workspace_namespace_name(workspace_name: str):
    # The namespace is workspace name
    return workspace_name


def get_workspace_namespace(workspace_name: str):
    # Check namespace exists
    namespace_name = get_workspace_namespace_name(workspace_name)
    namespace_object = _get_namespace(namespace_name)
    if namespace_object is None:
        return None
    return namespace_name


def _get_head_service_account(config, namespace):
    name = _get_head_service_account_name(config["provider"])
    return _get_service_account(namespace, name)


def _get_worker_service_account(config, namespace):
    name = _get_worker_service_account_name(config["provider"])
    return _get_service_account(namespace, name)


def _get_head_role(config, namespace):
    name = _get_head_role_name(config["provider"])
    return _get_role(namespace, name)


def _get_head_role_binding(config, namespace):
    name = _get_head_role_binding_name(config["provider"])
    return _get_role_binding(namespace, name)


def create_kubernetes_workspace(config):
    # create a copy of the input config to modify
    config = copy.deepcopy(config)

    # create workspace
    config = _create_workspace(config)

    return config


def delete_kubernetes_workspace(config, delete_managed_storage: bool = False):
    workspace_name = config["workspace_name"]
    namespace = get_workspace_namespace(workspace_name)
    if namespace is None:
        cli_logger.print("The workspace: {} doesn't exist!".format(config["workspace_name"]))
        return

    current_step = 1
    total_steps = KUBERNETES_WORKSPACE_NUM_DELETION_STEPS

    try:
        with cli_logger.group("Deleting workspace: {}", workspace_name):
            with cli_logger.group(
                    "Deleting cloud provider configurations",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _delete_configurations_for_cloud_provider(config, workspace_name, delete_managed_storage)

            with cli_logger.group(
                    "Deleting role binding",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _delete_head_role_binding(workspace_name, config["provider"])

            with cli_logger.group(
                    "Deleting role",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _delete_head_role(workspace_name, config["provider"])

            with cli_logger.group(
                    "Deleting service accounts",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _delete_service_accounts(workspace_name, config["provider"])

            with cli_logger.group(
                    "Deleting namespace",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _delete_namespace(workspace_name)

    except Exception as e:
        cli_logger.error(
            "Failed to delete workspace {}. {}".format(workspace_name, str(e)))
        raise e

    cli_logger.print(
        "Successfully deleted workspace: {}.",
        cf.bold(workspace_name))


def check_kubernetes_workspace_existence(config):
    workspace_name = config["workspace_name"]
    existing_resources = 0
    target_resources = KUBERNETES_WORKSPACE_TARGET_RESOURCES
    cloud_existence = Existence.NOT_EXIST
    """
         Do the work - order of operation
         1.) Check namespace
         2.) Check service accounts (2)
         3.) Check role
         4.) Check role binding
    """
    namespace = get_workspace_namespace(workspace_name)
    if namespace is not None:
        existing_resources += 1

        # Resources depending on namespace
        if _get_head_service_account(config, namespace) is not None:
            existing_resources += 1
        if _get_worker_service_account(config, namespace) is not None:
            existing_resources += 1
        if _get_head_role(config, namespace) is not None:
            existing_resources += 1
        if _get_head_role_binding(config, namespace) is not None:
            existing_resources += 1

    # The namespace may not exist
    namespace_name = get_workspace_namespace_name(workspace_name)
    cloud_existence = _check_existence_for_cloud_provider(config, namespace_name)

    if existing_resources == 0:
        if cloud_existence is not None:
            if cloud_existence == Existence.STORAGE_ONLY:
                return Existence.STORAGE_ONLY
            elif cloud_existence == Existence.NOT_EXIST:
                return Existence.NOT_EXIST
            return Existence.IN_COMPLETED

        return Existence.NOT_EXIST
    elif existing_resources == target_resources:
        if cloud_existence is not None:
            if cloud_existence == Existence.COMPLETED:
                return Existence.COMPLETED
            return Existence.IN_COMPLETED

        return Existence.COMPLETED
    else:
        return Existence.IN_COMPLETED


def check_kubernetes_workspace_integrity(config):
    existence = check_kubernetes_workspace_existence(config)
    return True if existence == Existence.COMPLETED else False


def list_kubernetes_clusters(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    head_nodes = get_workspace_head_nodes(config)
    clusters = {}
    for head_node in head_nodes:
        cluster_name = get_cluster_name_from_head(head_node)
        if cluster_name:
            clusters[cluster_name] = _get_node_info(head_node)
    return clusters


def get_kubernetes_workspace_info(config):
    namespace = config["workspace_name"]
    provider_config = config["provider"]

    head_service_account_name = _get_head_service_account_name(provider_config)
    worker_service_account_name = _get_worker_service_account_name(provider_config)
    role_name = _get_head_role_name(provider_config)
    role_binding_name = _get_head_role_binding_name(provider_config)

    info = {
        KUBERNETES_NAMESPACE: namespace,
        KUBERNETES_HEAD_SERVICE_ACCOUNT: head_service_account_name,
        KUBERNETES_WORKER_SERVICE_ACCOUNT: worker_service_account_name,
        KUBERNETES_HEAD_ROLE: role_name,
        KUBERNETES_HEAD_ROLE_BINDING: role_binding_name,
    }

    cloud_provider = _get_cloud_provider_config(provider_config)
    if cloud_provider is not None:
        cloud_provider_type = cloud_provider["type"]

        if cloud_provider_type == "aws":
            from cloudtik.providers._private._kubernetes.aws_eks.config import get_info_for_aws
            get_info_for_aws(config, namespace, cloud_provider, info)
        elif cloud_provider_type == "gcp":
            from cloudtik.providers._private._kubernetes.gcp_gke.config import get_info_for_gcp
            get_info_for_gcp(config, namespace, cloud_provider, info)

    return info


def validate_kubernetes_config(provider_config: Dict[str, Any], workspace_name: str):
    if len(workspace_name) > KUBERNETES_WORKSPACE_NAME_MAX or \
            not check_kubernetes_name_format(workspace_name):
        raise RuntimeError("{} workspace name is between 1 and {} characters, "
                           "and can only contain lowercase alphanumeric "
                           "characters and '-' or '.'".format(provider_config["type"], KUBERNETES_WORKSPACE_NAME_MAX))


def prepare_kubernetes_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare kubernetes cluster config.
    """
    if not is_use_internal_ip(config):
        ssh_start_cmd = " sudo /etc/init.d/ssh start"
        config["head_start_commands"] = [ssh_start_cmd] + config.get("head_start_commands", [])
    return config


def bootstrap_kubernetes_workspace(config):
    return config


def bootstrap_kubernetes_for_api(config):
    workspace_name = config.get("workspace_name", "")
    if workspace_name == "":
        raise ValueError(f"Workspace name is not specified.")

    _configure_namespace_from_workspace(config)
    return config


def bootstrap_kubernetes(config):
    workspace_name = config.get("workspace_name", "")
    if workspace_name == "":
        config = bootstrap_kubernetes_default(config)
    else:
        config = bootstrap_kubernetes_from_workspace(config)
    return config


def bootstrap_kubernetes_default(config):
    if config["provider"].get("_operator"):
        namespace = config["provider"]["namespace"]
    else:
        namespace = _configure_namespace(config["provider"])

    _configure_pods(config)
    _configure_services(config)
    _create_or_update_services(namespace, config["provider"])

    if not config["provider"].get("_operator"):
        # These steps are unnecessary when using the Operator.
        _configure_head_service_account(namespace, config["provider"])
        _configure_head_role(namespace, config["provider"])
        _configure_head_role_binding(namespace, config["provider"])

    configure_for_ssh(config)
    return config


def bootstrap_kubernetes_from_workspace(config):
    namespace = _configure_namespace_from_workspace(config)

    _configure_cloud_provider(config, namespace)
    _configure_pods(config)
    _configure_services(config)
    _create_or_update_services(namespace, config["provider"])

    configure_for_ssh(config)
    return config


def configure_for_ssh(config):
    if is_use_internal_ip(config):
        return
    auth_config = config["auth"]
    ssh_private_key = auth_config.get("ssh_private_key", None)
    ssh_public_key = auth_config.get("ssh_public_key", None)
    if not auth_config.get("ssh_port", None):
        auth_config["ssh_port"] = CLOUDTIK_KUBERNETES_SSH_DEFAULT_PORT
    if ssh_public_key and ssh_private_key:
        return
    else:
        key_pair_file_path = get_key_pair_path_for_kubernetes(config)
        private_key_generation_cmd = f"test -e {key_pair_file_path}||ssh-keygen -t rsa -q -N '' -m PEM -f {key_pair_file_path}"
        os.system(private_key_generation_cmd)
        auth_config["ssh_private_key"] = key_pair_file_path
        auth_config["ssh_public_key"] = f"{key_pair_file_path}.pub"

        cli_logger.print("Private key not specified in config, generated and using "
                         "{}".format(key_pair_file_path))


def cleanup_kubernetes_cluster(config, cluster_name, namespace):
    # Delete services associated with the cluster
    _delete_services(config)

    # Clean up the PVCs if there are any not deleted by Pod deletion
    cleanup_orphan_pvcs(cluster_name, namespace)


def _configure_namespace_from_workspace(config):
    if config["provider"].get("_operator"):
        namespace = config["provider"]["namespace"]
    else:
        workspace_name = config["workspace_name"]
        # We don't check namespace existence here since operator may not have the permission to list namespaces
        namespace = get_workspace_namespace_name(workspace_name)
        if namespace is None:
            raise RuntimeError("The workspace namespace {} doesn't exist.".format(workspace_name))

        config["provider"]["namespace"] = namespace
    return namespace


def post_prepare_kubernetes(config: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Configure pod container resource should be done before fill resources
        # Because fill resources will use the pod resources as a source of truth to resources
        # and set the resources field in the node type for other usage
        _configure_pod_container_resources(config)
        config = fill_resources_kubernetes(config)
    except Exception as exc:
        cli_logger.warning(
            "Failed to detect node resources. Make sure you have properly configured the Kubernetes credentials: {}.",
            str(exc))

    # Set use_managed_cloud_storage value from cloud_provider if it is set
    provider_config = config["provider"]
    if "cloud_provider" in provider_config:
        provider_config["use_managed_cloud_storage"] = _is_use_managed_cloud_storage(
            provider_config["cloud_provider"])
    return config


def fill_resources_kubernetes(config):
    """Fills CPU and GPU resources by reading pod spec of each available node
    type.

    For each node type and each of CPU/GPU, looks at container's resources
    and limits, takes min of the two. The result is rounded up, as we do
    not currently support fractional CPU.
    """
    if "available_node_types" not in config:
        return config
    node_types = copy.deepcopy(config["available_node_types"])
    for node_type in node_types:
        node_config = node_types[node_type]["node_config"]
        pod = node_config["pod"]
        container_data = pod["spec"]["containers"][0]

        autodetected_resources = get_autodetected_resources(container_data)
        node_type_config = config["available_node_types"][node_type]
        if "resources" not in node_type_config:
            node_type_config["resources"] = {}
        autodetected_resources.update(
            node_type_config["resources"])
        node_type_config["resources"] = autodetected_resources
        logger.debug(
            "Updating the resources of node type {} to include {}.".format(
                node_type, autodetected_resources))
    return config


def get_autodetected_resources(container_data):
    container_resources = container_data.get("resources", None)
    if container_resources is None:
        return {"CPU": 0, "GPU": 0}

    node_type_resources = {
        resource_name.upper(): get_resource(container_resources, resource_name)
        for resource_name in ["cpu", "gpu"]
    }

    memory_limits = get_resource(container_resources, "memory")
    node_type_resources["memory"] = int(
        memory_limits *
        (1 - CLOUDTIK_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION))

    return node_type_resources


def get_resource(container_resources, resource_name):
    # float("inf") means there's no limit set
    # consider limit first if it is specified for this resource type
    limit = _get_resource(
        container_resources, resource_name, field_name="limits")
    if limit != float("inf"):
        return int(limit)

    # if no limit specified, check requests
    request = _get_resource(
        container_resources, resource_name, field_name="requests")
    return 0 if request == float("inf") else int(request)


def _get_resource(container_resources, resource_name, field_name):
    """Returns the resource quantity.

    The amount of resource is rounded up to nearest integer.
    Returns float("inf") if the resource is not present.

    Args:
        container_resources (dict): Container's resource field.
        resource_name (str): One of 'cpu', 'gpu' or memory.
        field_name (str): One of 'requests' or 'limits'.

    Returns:
        Union[int, float]: Detected resource quantity.
    """
    if field_name not in container_resources:
        # No limit/resource field.
        return float("inf")
    resources = container_resources[field_name]
    # Look for keys containing the resource_name. For example,
    # the key 'nvidia.com/gpu' contains the key 'gpu'.
    matching_keys = [key for key in resources if resource_name in key.lower()]
    if len(matching_keys) == 0:
        return float("inf")
    if len(matching_keys) > 1:
        # Should have only one match -- mostly relevant for gpu.
        raise ValueError(f"Multiple {resource_name} types not supported.")
    # E.g. 'nvidia.com/gpu' or 'cpu'.
    resource_key = matching_keys.pop()
    resource_quantity = resources[resource_key]
    if resource_name == "memory":
        return _parse_memory_resource(resource_quantity)
    else:
        return _parse_cpu_or_gpu_resource(resource_quantity)


def _parse_cpu_or_gpu_resource(resource):
    resource_str = str(resource)
    if resource_str[-1] == "m":
        # For example, '500m' rounds up to 1.
        return math.ceil(int(resource_str[:-1]) / 1000)
    else:
        return int(resource_str)


def _parse_memory_resource(resource):
    resource_str = str(resource)
    try:
        return int(resource_str)
    except ValueError:
        pass
    memory_size = re.sub(r"([KMGTP]+)", r" \1", resource_str)
    number, unit_index = [item.strip() for item in memory_size.split()]
    unit_index = unit_index[0]
    return float(number) * MEMORY_SIZE_UNITS[unit_index]


def _configure_namespace(provider_config):
    namespace_field = "namespace"
    if namespace_field not in provider_config:
        raise ValueError("Must specify namespace in Kubernetes config.")

    namespace = provider_config[namespace_field]
    field_selector = "metadata.name={}".format(namespace)
    try:
        namespaces = core_api().list_namespace(
            field_selector=field_selector).items
    except ApiException:
        logger.warning(log_prefix +
                       not_checking_msg(namespace_field, namespace))
        return namespace

    if len(namespaces) > 0:
        assert len(namespaces) == 1
        cli_logger.print(log_prefix + using_existing_msg(namespace_field, namespace))
        return namespace

    cli_logger.print(log_prefix + not_found_msg(namespace_field, namespace))
    namespace_config = client.V1Namespace(
        metadata=client.V1ObjectMeta(name=namespace))
    core_api().create_namespace(namespace_config)
    cli_logger.print(log_prefix + created_msg(namespace_field, namespace))
    return namespace


def _configure_head_service_account(namespace, provider_config):
    account_field = KUBERNETES_HEAD_SERVICE_ACCOUNT_CONFIG_KEY
    if account_field not in provider_config:
        cli_logger.print(log_prefix + not_provided_msg(account_field))
        return

    account = provider_config[account_field]
    if "namespace" not in account["metadata"]:
        account["metadata"]["namespace"] = namespace
    elif account["metadata"]["namespace"] != namespace:
        raise InvalidNamespaceError(account_field, namespace)

    name = account["metadata"]["name"]
    field_selector = "metadata.name={}".format(name)
    accounts = core_api().list_namespaced_service_account(
        namespace, field_selector=field_selector).items
    if len(accounts) > 0:
        assert len(accounts) == 1
        cli_logger.print(log_prefix + using_existing_msg(account_field, name))
        return

    cli_logger.print(log_prefix + not_found_msg(account_field, name))
    core_api().create_namespaced_service_account(namespace, account)
    cli_logger.print(log_prefix + created_msg(account_field, name))


def _configure_head_role(namespace, provider_config):
    role_field = KUBERNETES_HEAD_ROLE_CONFIG_KEY
    if role_field not in provider_config:
        cli_logger.print(log_prefix + not_provided_msg(role_field))
        return

    role = provider_config[role_field]
    if "namespace" not in role["metadata"]:
        role["metadata"]["namespace"] = namespace
    elif role["metadata"]["namespace"] != namespace:
        raise InvalidNamespaceError(role_field, namespace)

    name = role["metadata"]["name"]
    field_selector = "metadata.name={}".format(name)
    accounts = auth_api().list_namespaced_role(
        namespace, field_selector=field_selector).items
    if len(accounts) > 0:
        assert len(accounts) == 1
        cli_logger.print(log_prefix + using_existing_msg(role_field, name))
        return

    cli_logger.print(log_prefix + not_found_msg(role_field, name))
    auth_api().create_namespaced_role(namespace, role)
    cli_logger.print(log_prefix + created_msg(role_field, name))


def _configure_head_role_binding(namespace, provider_config):
    binding_field = KUBERNETES_HEAD_ROLE_BINDING_CONFIG_KEY
    if binding_field not in provider_config:
        cli_logger.print(log_prefix + not_provided_msg(binding_field))
        return

    binding = provider_config[binding_field]
    if "namespace" not in binding["metadata"]:
        binding["metadata"]["namespace"] = namespace
    elif binding["metadata"]["namespace"] != namespace:
        raise InvalidNamespaceError(binding_field, namespace)
    for subject in binding["subjects"]:
        if "namespace" not in subject:
            subject["namespace"] = namespace
        elif subject["namespace"] != namespace:
            raise InvalidNamespaceError(
                binding_field + " subject '{}'".format(subject["name"]),
                namespace)

    name = binding["metadata"]["name"]
    field_selector = "metadata.name={}".format(name)
    accounts = auth_api().list_namespaced_role_binding(
        namespace, field_selector=field_selector).items
    if len(accounts) > 0:
        assert len(accounts) == 1
        cli_logger.print(log_prefix + using_existing_msg(binding_field, name))
        return

    cli_logger.print(log_prefix + not_found_msg(binding_field, name))
    auth_api().create_namespaced_role_binding(namespace, binding)
    cli_logger.print(log_prefix + created_msg(binding_field, name))


def _create_or_update_services(namespace, provider_config):
    _create_or_update_head_service(namespace, provider_config)
    _create_or_update_head_external_service(namespace, provider_config)
    _create_or_update_node_service(namespace, provider_config)


def _create_or_update_head_service(namespace, provider_config):
    service_field = KUBERNETES_HEAD_SERVICE_CONFIG_KEY
    if service_field not in provider_config:
        cli_logger.print(log_prefix + not_provided_msg(service_field))
        return
    _create_or_update_service(namespace, provider_config, service_field)


def _create_or_update_head_external_service(namespace, provider_config):
    # Check whether we need to create the service based on use_internal_ip flag
    if _is_use_internal_ip(provider_config):
        return

    service_field = KUBERNETES_HEAD_EXTERNAL_SERVICE_CONFIG_KEY
    if service_field not in provider_config:
        cli_logger.print(log_prefix + not_provided_msg(service_field))
        return
    _create_or_update_service(namespace, provider_config, service_field)


def _create_or_update_node_service(namespace, provider_config):
    service_field = KUBERNETES_NODE_SERVICE_CONFIG_KEY
    if service_field not in provider_config:
        cli_logger.print(log_prefix + not_provided_msg(service_field))
        raise RuntimeError("No node service specified in configuration.")
    _create_or_update_service(namespace, provider_config, service_field)


def _create_or_update_service(namespace, provider_config, service_field):
    service = provider_config[service_field]
    if "namespace" not in service["metadata"]:
        service["metadata"]["namespace"] = namespace
    elif service["metadata"]["namespace"] != namespace:
        raise InvalidNamespaceError(service_field, namespace)

    name = service["metadata"]["name"]
    field_selector = "metadata.name={}".format(name)
    service_list = core_api().list_namespaced_service(
        namespace, field_selector=field_selector).items
    if len(service_list) > 0:
        assert len(service_list) == 1
        existing_service = service_list[0]
        if service == existing_service:
            cli_logger.print(log_prefix + using_existing_msg("service", name))
            return
        else:
            cli_logger.print(log_prefix + updating_existing_msg("service", name))
            core_api().patch_namespaced_service(name, namespace, service)
    else:
        cli_logger.print(log_prefix + not_found_msg("service", name))
        core_api().create_namespaced_service(namespace, service)
        cli_logger.print(log_prefix + created_msg("service", name))


def _configure_pods(config):
    # Update the node config with image
    _configure_pod_image(config)

    # Update the generateName pod name and labels with the cluster name
    _configure_pod_name_and_labels(config)
    _configure_pod_service_account(config)
    _configure_pod_hostname(config)

    # Configure the head pod container ports
    _configure_pod_container_ports(config)


def _configure_pod_name_and_labels(config):
    if "available_node_types" not in config:
        return config

    cluster_name = config["cluster_name"]
    node_types = config["available_node_types"]
    head_node_type = config["head_node_type"]
    for node_type in node_types:
        node_config = node_types[node_type]["node_config"]
        pod = node_config["pod"]
        if "metadata" not in pod:
            pod["metadata"] = {}
        metadata = pod["metadata"]
        if node_type == head_node_type:
            _configure_pod_name_and_labels_for_head(metadata, cluster_name)
        else:
            _configure_pod_name_and_labels_for_worker(metadata, cluster_name)


def _configure_pod_name_and_labels_for_head(metadata, cluster_name):
    _configure_pod_name(metadata, cluster_name, CLOUDTIK_HEAD_POD_NAME_PREFIX)
    if "labels" not in metadata:
        metadata["labels"] = {}
    labels = metadata["labels"]
    labels["cluster"] = _get_cluster_selector(cluster_name)
    labels["component"] = _get_head_component_selector(cluster_name)


def _configure_pod_name_and_labels_for_worker(metadata, cluster_name):
    _configure_pod_name(metadata, cluster_name, CLOUDTIK_WORKER_POD_NAME_PREFIX)
    if "labels" not in metadata:
        metadata["labels"] = {}
    labels = metadata["labels"]
    labels["cluster"] = _get_cluster_selector(cluster_name)
    labels["component"] = _get_worker_component_selector(cluster_name)


def _configure_pod_name(metadata, cluster_name, name_pattern):
    metadata["generateName"] = name_pattern.format(cluster_name)


def _configure_pod_hostname(config):
    if "available_node_types" not in config:
        return config

    cluster_name = config["cluster_name"]
    node_types = config["available_node_types"]
    for node_type in node_types:
        node_config = node_types[node_type]["node_config"]
        spec = node_config["pod"]["spec"]
        spec["subdomain"] = cluster_name


def _configure_pod_container_ports(config):
    if "available_node_types" not in config:
        return

    runtime_config = config.get("runtime", {})
    service_ports = get_runtime_service_ports(runtime_config)

    node_types = config["available_node_types"]
    head_node_type = config["head_node_type"]
    node_config = node_types[head_node_type]["node_config"]
    pod = node_config["pod"]
    container_data = pod["spec"]["containers"][0]

    if "ports" not in container_data:
        container_data["ports"] = []

    ports = container_data["ports"]
    for port_name in service_ports:
        port_config = service_ports[port_name]
        container_port = {
            "containerPort": port_config["port"],
            "name": port_name,
        }
        ports.append(container_port)

    if not is_use_internal_ip(config):
        # ssh port
        ports.append(KUBERNETES_CONTAINER_SSH_PORT_SPEC)

    container_data["ports"] = ports


def _configure_pod_container_resources(config):
    if "available_node_types" not in config:
        return

    node_types = config["available_node_types"]
    for node_type in node_types:
        node_config = node_types[node_type]["node_config"]
        if "resources" in node_config:
            resources = node_config["resources"]
            containers = node_config["pod"]["spec"]["containers"]
            for container in containers:
                _configure_container_resources(resources, container)


def _configure_container_resources(resources, container):
    cpu = resources.get("cpu")
    memory = resources.get("memory")
    if cpu is not None or memory is not None:
        if "resources" not in container:
            container["resources"] = {}
        container_resources = container["resources"]
        if "requests" not in container_resources:
            container_resources["requests"] = {}
        if "limits" not in container_resources:
            container_resources["limits"] = {}

        if cpu is not None:
            container_resources["requests"]["cpu"] = cpu
            container_resources["limits"]["cpu"] = cpu

        if memory is not None:
            container_resources["requests"]["memory"] = memory
            container_resources["limits"]["memory"] = memory


def _configure_services(config):
    _configure_services_name_and_selector(config)
    _configure_head_service_ports(config)
    _configure_head_external_service_ports(config)


def _configure_services_name_and_selector(config):
    _configure_head_service_name_and_selector(config)
    _configure_head_external_service_name_and_selector(config)
    _configure_node_service_name_and_selector(config)


def _configure_head_service_name_and_selector(config):
    cluster_name = config["cluster_name"]
    _configure_service_name_and_selector_for_head(
        config,
        KUBERNETES_HEAD_SERVICE_CONFIG_KEY,
        _get_head_service_name(cluster_name)
    )


def _configure_head_external_service_name_and_selector(config):
    cluster_name = config["cluster_name"]
    _configure_service_name_and_selector_for_head(
        config,
        KUBERNETES_HEAD_EXTERNAL_SERVICE_CONFIG_KEY,
        _get_head_external_service_name(cluster_name)
    )


def _configure_service_name_and_selector_for_head(
        config, service_field, service_name):
    provider_config = config["provider"]
    if service_field not in provider_config:
        return

    cluster_name = config["cluster_name"]
    service = provider_config[service_field]

    if "metadata" not in service:
        service["metadata"] = {}
    service["metadata"]["name"] = service_name

    if "spec" not in service:
        service["spec"] = {}
    if "selector" not in service["spec"]:
        service["spec"]["selector"] = {}

    service["spec"]["selector"]["component"] = _get_head_component_selector(cluster_name)


def _configure_node_service_name_and_selector(config):
    provider_config = config["provider"]
    service_field = KUBERNETES_NODE_SERVICE_CONFIG_KEY
    if service_field not in provider_config:
        return

    cluster_name = config["cluster_name"]
    service = provider_config[service_field]

    if "metadata" not in service:
        service["metadata"] = {}
    service["metadata"]["name"] = _get_node_service_name(cluster_name)

    if "spec" not in service:
        service["spec"] = {}
    if "selector" not in service["spec"]:
        service["spec"]["selector"] = {}
    service["spec"]["selector"]["cluster"] = _get_cluster_selector(cluster_name)


def _configure_head_service_ports(config):
    provider_config = config["provider"]
    service_field = KUBERNETES_HEAD_SERVICE_CONFIG_KEY
    if service_field not in provider_config:
        return
    service = provider_config[service_field]

    runtime_config = config.get("runtime", {})
    runtime_service_ports = get_runtime_service_ports(runtime_config)
    _configure_service_ports_for_runtime(service, runtime_service_ports)


def _configure_service_ports_for_runtime(service, service_ports):
    if "spec" not in service:
        service["spec"] = {}
    if "ports" not in service["spec"]:
        service["spec"]["ports"] = []

    ports = service["spec"]["ports"]
    for port_name in service_ports:
        port_config = service_ports[port_name]
        port = {
            "name": "{}-svc-port".format(port_name),
            "protocol": port_config["protocol"],
            "port": port_config["port"],
            "targetPort": port_name,
        }
        ports.append(port)
    service["spec"]["ports"] = ports


def _configure_head_external_service_ports(config):
    provider_config = config["provider"]
    service_field = KUBERNETES_HEAD_EXTERNAL_SERVICE_CONFIG_KEY
    if service_field not in provider_config:
        return

    service = provider_config[service_field]
    ssh_port = copy.deepcopy(KUBERNETES_HEAD_EXTERNAL_SERVICE_SSH_PORT_SPEC)
    ssh_port["port"] = config["auth"].get("ssh_port", CLOUDTIK_KUBERNETES_SSH_DEFAULT_PORT)
    service["spec"]["ports"] = [ssh_port]


def _configure_pod_service_account(config):
    if "available_node_types" not in config:
        return config

    provider_config = config["provider"]

    head_service_account_name = _get_head_service_account_name(provider_config)
    worker_service_account_name = _get_worker_service_account_name(provider_config)
    node_types = config["available_node_types"]
    head_node_type = config["head_node_type"]
    for node_type in node_types:
        node_type_config = node_types[node_type]
        node_config = node_type_config["node_config"]
        pod = node_config["pod"]
        pod_spec = pod["spec"]
        if node_type == head_node_type:
            # If service account name is not configured, configure it
            if "serviceAccountName" not in pod_spec:
                pod_spec["serviceAccountName"] = head_service_account_name
        else:
            # If service account name is not configured, configure it
            if "serviceAccountName" not in pod_spec:
                pod_spec["serviceAccountName"] = worker_service_account_name


def _configure_pod_image(config):
    if "available_node_types" not in config:
        return config

    provider_config = config["provider"]
    node_types = config["available_node_types"]
    for node_type in node_types:
        node_type_config = node_types[node_type]
        image = get_node_type_image(provider_config, node_type_config)
        if image is not None:
            _configure_pod_image_for_node_type(node_type_config, image)


def get_node_type_image(provider_config, node_type_config):
    if CONFIG_NAME_IMAGE in node_type_config:
        return node_type_config[CONFIG_NAME_IMAGE]

    return provider_config.get(CONFIG_NAME_IMAGE)


def _configure_pod_image_for_node_type(node_type_config, image):
    node_config = node_type_config["node_config"]
    pod = node_config["pod"]
    container_data = pod["spec"]["containers"][0]

    # The image spec in the container will take high priority than external config
    if CONFIG_NAME_IMAGE not in container_data:
        container_data[CONFIG_NAME_IMAGE] = image


def get_cluster_name_from_head(head_node) -> Optional[str]:
    labels = head_node.metadata.labels
    if labels is not None and CLOUDTIK_TAG_CLUSTER_NAME in labels:
        return labels[CLOUDTIK_TAG_CLUSTER_NAME]
    return None


def get_workspace_head_nodes(config):
    return _get_workspace_head_nodes(
        config["provider"], config["workspace_name"])


def _get_workspace_head_nodes(provider_config, workspace_name):
    namespace = get_workspace_namespace_name(workspace_name)
    if namespace is None:
        raise RuntimeError(f"Kubernetes namespace for workspace doesn't exist: {workspace_name}")

    head_node_tags = {
        CLOUDTIK_TAG_NODE_KIND: NODE_KIND_HEAD,
    }

    field_selector = ",".join([
        "status.phase!=Failed",
        "status.phase!=Unknown",
        "status.phase!=Succeeded",
        "status.phase!=Terminating",
    ])

    label_selector = to_label_selector(head_node_tags)
    pod_list = core_api().list_namespaced_pod(
        namespace,
        field_selector=field_selector,
        label_selector=label_selector)

    # Don't return pods marked for deletion,
    # i.e. pods with non-null metadata.DeletionTimestamp.
    return [
        pod for pod in pod_list.items
        if pod.metadata.deletion_timestamp is None
    ]


def publish_kubernetes_global_variables(
        cluster_config: Dict[str, Any], global_variables: Dict[str, Any]):
    # Add prefix to the variables
    global_variables_prefixed = {}
    for name in global_variables:
        prefixed_name = CLOUDTIK_GLOBAL_VARIABLE_KEY.format(name)
        value = global_variables[name]
        hex_encoded_value = binary_to_hex(value.encode())
        global_variables_prefixed[prefixed_name] = hex_encoded_value

    provider = _get_node_provider(cluster_config["provider"], cluster_config["cluster_name"])
    head_node_id = get_running_head_node(cluster_config, provider)
    provider.set_node_tags(head_node_id, global_variables_prefixed)


def subscribe_kubernetes_global_variables(
        provider_config: Dict[str, Any], workspace_name: str, cluster_config: Dict[str, Any]):
    global_variables = {}
    head_nodes = _get_workspace_head_nodes(
        provider_config, workspace_name)
    for head in head_nodes:
        labels = head.metadata.labels
        if labels is None:
            continue

        for key, value in labels.items():
            if key.startswith(CLOUDTIK_GLOBAL_VARIABLE_KEY_PREFIX):
                global_variable_name = key[len(CLOUDTIK_GLOBAL_VARIABLE_KEY_PREFIX):]
                hex_decoded_value = hex_to_binary(value).decode()
                global_variables[global_variable_name] = hex_decoded_value

    return global_variables


def _create_workspace(config):
    workspace_name = config["workspace_name"]

    current_step = 1
    total_steps = KUBERNETES_WORKSPACE_NUM_CREATION_STEPS

    try:
        with cli_logger.group("Creating workspace: {}", workspace_name):
            with cli_logger.group(
                    "Creating namespace",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _create_namespace(workspace_name)

            with cli_logger.group(
                    "Creating service accounts",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _create_service_accounts(workspace_name, config["provider"])

            with cli_logger.group(
                    "Creating role",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _create_head_role(workspace_name, config["provider"])

            with cli_logger.group(
                    "Creating role binding",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _create_head_role_binding(workspace_name, config["provider"])

            with cli_logger.group(
                    "Creating cloud provider configurations",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _create_configurations_for_cloud_provider(config, workspace_name)

    except Exception as e:
        cli_logger.error("Failed to create workspace with the name {}. "
                         "You need to delete and try create again. {}", workspace_name, str(e))
        raise e

    cli_logger.print(
        "Successfully created workspace: {}.",
        cf.bold(workspace_name))

    return config


def _get_namespace(namespace: str):
    field_selector = "metadata.name={}".format(namespace)

    cli_logger.verbose("Getting the namespace: {}.", namespace)
    namespaces = core_api().list_namespace(
        field_selector=field_selector).items

    if len(namespaces) > 0:
        assert len(namespaces) == 1
        cli_logger.verbose("Successfully get the namespace: {}.", namespace)
        return namespaces[0]

    cli_logger.verbose("Failed to get the namespace: {}.", namespace)
    return None


def _create_namespace(workspace_name: str):
    namespace_field = "namespace"
    namespace = workspace_name

    # Check existence
    namespace_object = _get_namespace(namespace)
    if namespace_object is not None:
        cli_logger.print(log_prefix + using_existing_msg(namespace_field, namespace))
        return

    cli_logger.print(log_prefix + creating_msg(namespace_field, namespace))
    namespace_config = client.V1Namespace(
        metadata=client.V1ObjectMeta(name=namespace))
    core_api().create_namespace(namespace_config)
    cli_logger.print(log_prefix + created_msg(namespace_field, namespace))
    return namespace


def _create_service_accounts(namespace, provider_config):
    current_step = 1
    total_steps = 2

    with cli_logger.group(
            "Creating head service account",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_head_service_account(namespace, provider_config)

    with cli_logger.group(
            "Creating worker service account",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_worker_service_account(namespace, provider_config)


def _create_head_service_account(namespace, provider_config):
    service_account_name = _get_head_service_account_name(provider_config)
    _create_service_account(
        namespace,
        provider_config,
        KUBERNETES_HEAD_SERVICE_ACCOUNT_CONFIG_KEY,
        service_account_name
    )


def _create_worker_service_account(namespace, provider_config):
    service_account_name = _get_worker_service_account_name(provider_config)
    _create_service_account(
        namespace,
        provider_config,
        KUBERNETES_WORKER_SERVICE_ACCOUNT_CONFIG_KEY,
        service_account_name
    )


def _create_service_account(namespace, provider_config,
                            account_field, name):
    if account_field not in provider_config:
        cli_logger.print(log_prefix + not_provided_msg(account_field))
        return

    service_account_object = _get_service_account(namespace, name)
    if service_account_object is not None:
        cli_logger.print(log_prefix + using_existing_msg(account_field, name))
        return

    account = provider_config[account_field]
    if "metadata" not in account:
        account["metadata"] = {}
    if "namespace" not in account["metadata"]:
        account["metadata"]["namespace"] = namespace
    elif account["metadata"]["namespace"] != namespace:
        raise InvalidNamespaceError(account_field, namespace)
    if "name" not in account["metadata"]:
        account["metadata"]["name"] = name

    cli_logger.print(log_prefix + creating_msg(account_field, name))
    core_api().create_namespaced_service_account(namespace, account)
    cli_logger.print(log_prefix + created_msg(account_field, name))


def _get_head_role_name(provider_config):
    role_field = KUBERNETES_HEAD_ROLE_CONFIG_KEY
    name = provider_config.get(role_field, {}).get("metadata", {}).get("name")
    if name is None or name == "":
        return KUBERNETES_HEAD_ROLE_NAME
    return name


def _get_role(namespace, name):
    field_selector = "metadata.name={}".format(name)
    cli_logger.verbose("Getting the role: {} {}.", namespace, name)
    roles = auth_api().list_namespaced_role(
        namespace, field_selector=field_selector).items
    if len(roles) > 0:
        assert len(roles) == 1
        cli_logger.verbose("Successfully get the role: {} {}.", namespace, name)
        return roles[0]

    cli_logger.verbose("Failed to get the role: {} {}.", namespace, name)
    return None


def _create_head_role(namespace, provider_config):
    role_field = KUBERNETES_HEAD_ROLE_CONFIG_KEY
    if role_field not in provider_config:
        cli_logger.print(log_prefix + not_provided_msg(role_field))
        return

    name = _get_head_role_name(provider_config)
    role_object = _get_role(namespace, name)
    if role_object is not None:
        cli_logger.print(log_prefix + using_existing_msg(role_field, name))
        return

    role = provider_config[role_field]
    if "metadata" not in role:
        role["metadata"] = {}
    if "namespace" not in role["metadata"]:
        role["metadata"]["namespace"] = namespace
    elif role["metadata"]["namespace"] != namespace:
        raise InvalidNamespaceError(role_field, namespace)
    if "name" not in role["metadata"]:
        role["metadata"]["name"] = name

    cli_logger.print(log_prefix + creating_msg(role_field, name))
    auth_api().create_namespaced_role(namespace, role)
    cli_logger.print(log_prefix + created_msg(role_field, name))


def _get_head_role_binding_name(provider_config):
    binding_field = KUBERNETES_HEAD_ROLE_BINDING_CONFIG_KEY
    name = provider_config.get(binding_field, {}).get("metadata", {}).get("name")
    if name is None or name == "":
        return KUBERNETES_HEAD_ROLE_BINDING_NAME
    return name


def _get_role_binding(namespace, name):
    field_selector = "metadata.name={}".format(name)

    cli_logger.verbose("Getting the role binding: {} {}.", namespace, name)
    role_bindings = auth_api().list_namespaced_role_binding(
        namespace, field_selector=field_selector).items
    if len(role_bindings) > 0:
        assert len(role_bindings) == 1
        cli_logger.verbose("Successfully get the role binding: {} {}.", namespace, name)
        return role_bindings[0]

    cli_logger.verbose("Failed to get the role binding: {} {}", namespace, name)
    return None


def _create_head_role_binding(namespace, provider_config):
    binding_field = KUBERNETES_HEAD_ROLE_BINDING_CONFIG_KEY
    if binding_field not in provider_config:
        cli_logger.print(log_prefix + not_provided_msg(binding_field))
        return

    name = _get_head_role_binding_name(provider_config)
    role_binding_object = _get_role_binding(namespace, name)
    if role_binding_object is not None:
        cli_logger.print(log_prefix + using_existing_msg(binding_field, name))
        return

    service_account_name = _get_head_service_account_name(provider_config)
    role_name = _get_head_role_name(provider_config)
    binding = provider_config[binding_field]
    if "metadata" not in binding:
        binding["metadata"] = {}
    if "namespace" not in binding["metadata"]:
        binding["metadata"]["namespace"] = namespace
    elif binding["metadata"]["namespace"] != namespace:
        raise InvalidNamespaceError(binding_field, namespace)
    for subject in binding["subjects"]:
        if "namespace" not in subject:
            subject["namespace"] = namespace
        elif subject["namespace"] != namespace:
            raise InvalidNamespaceError(
                binding_field + " subject '{}'".format(subject["name"]),
                namespace)
        if "name" not in subject:
            subject["name"] = service_account_name
    if name not in binding["roleRef"]:
        binding["roleRef"]["name"] = role_name
    if "name" not in binding["metadata"]:
        binding["metadata"]["name"] = name

    cli_logger.print(log_prefix + creating_msg(binding_field, name))
    auth_api().create_namespaced_role_binding(namespace, binding)
    cli_logger.print(log_prefix + created_msg(binding_field, name))


def _delete_namespace(namespace):
    namespace_object = _get_namespace(namespace)
    if namespace_object is None:
        cli_logger.print(log_prefix + "Namespace: {} doesn't exist.".format(namespace))
        return

    cli_logger.print(log_prefix + "Deleting namespace: {}".format(namespace))
    core_api().delete_namespace(namespace)
    wait_for_namespace_deleted(namespace)
    cli_logger.print(log_prefix + "Successfully deleted namespace: {}".format(namespace))


def wait_for_namespace_deleted(namespace):
    field_selector = "metadata.name={}".format(namespace)
    for _ in range(KUBERNETES_RESOURCE_OP_MAX_POLLS):
        namespaces = core_api().list_namespace(
            field_selector=field_selector).items
        if len(namespaces) == 0:
            return

        # wait for deletion
        cli_logger.verbose("Waiting for namespace delete operation to finish...")
        time.sleep(KUBERNETES_RESOURCE_OP_POLL_INTERVAL)

    raise RuntimeError("Namespace deletion doesn't completed after {} seconds.".format(
        KUBERNETES_RESOURCE_OP_MAX_POLLS * KUBERNETES_RESOURCE_OP_POLL_INTERVAL))


def _delete_service_accounts(namespace, provider_config):
    current_step = 1
    total_steps = 2

    with cli_logger.group(
            "Deleting head service account",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_head_service_account(namespace, provider_config)

    with cli_logger.group(
            "Deleting worker service account",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_worker_service_account(namespace, provider_config)


def _delete_head_service_account(namespace, provider_config):
    name = _get_head_service_account_name(provider_config)
    _delete_service_account(namespace, name)


def _delete_worker_service_account(namespace, provider_config):
    name = _get_worker_service_account_name(provider_config)
    _delete_service_account(namespace, name)


def _delete_service_account(namespace, name):
    service_account_object = _get_service_account(namespace, name)
    if service_account_object is None:
        cli_logger.print(log_prefix + "Service account: {} doesn't exist.".format(name))
        return

    cli_logger.print(log_prefix + "Deleting service account: {}".format(name))
    core_api().delete_namespaced_service_account(name, namespace)
    cli_logger.print(log_prefix + "Successfully deleted service account: {}".format(name))


def _delete_head_role(namespace, provider_config):
    name = _get_head_role_name(provider_config)

    role_object = _get_role(namespace, name)
    if role_object is None:
        cli_logger.print(log_prefix + "Role: {} doesn't exist.".format(name))
        return

    cli_logger.print(log_prefix + "Deleting role: {}".format(name))
    auth_api().delete_namespaced_role(name, namespace)
    cli_logger.print(log_prefix + "Successfully deleted role: {}".format(name))


def _delete_head_role_binding(namespace, provider_config):
    name = _get_head_role_binding_name(provider_config)

    role_binding_object = _get_role_binding(namespace, name)
    if role_binding_object is None:
        cli_logger.print(log_prefix + "Role Binding: {} doesn't exist.".format(name))
        return

    cli_logger.print(log_prefix + "Deleting role binding: {}".format(name))
    auth_api().delete_namespaced_role_binding(name, namespace)
    cli_logger.print(log_prefix + "Successfully deleted role binding: {}".format(name))


def _delete_services(config):
    _delete_head_service(config)
    _delete_head_external_service(config)
    _delete_node_service(config)


def _delete_head_service(config):
    provider_config = config["provider"]
    if "namespace" not in provider_config:
        raise ValueError("Must specify namespace in Kubernetes config.")

    namespace = provider_config["namespace"]
    cluster_name = config["cluster_name"]

    service_name = _get_head_service_name(cluster_name)
    _delete_service(namespace, service_name)


def _delete_head_external_service(config):
    if is_use_internal_ip(config):
        return

    provider_config = config["provider"]
    if "namespace" not in provider_config:
        raise ValueError("Must specify namespace in Kubernetes config.")

    namespace = provider_config["namespace"]
    cluster_name = config["cluster_name"]

    service_name = _get_head_external_service_name(cluster_name)
    _delete_service(namespace, service_name)


def _delete_node_service(config):
    provider_config = config["provider"]
    if "namespace" not in provider_config:
        raise ValueError("Must specify namespace in Kubernetes config.")

    namespace = provider_config["namespace"]
    cluster_name = config["cluster_name"]

    service_name = _get_node_service_name(cluster_name)
    _delete_service(namespace, service_name)


def _get_service(namespace, name):
    field_selector = "metadata.name={}".format(name)

    cli_logger.verbose("Getting the service: {} {}.", namespace, name)
    service_list = core_api().list_namespaced_service(
        namespace, field_selector=field_selector).items
    if len(service_list) > 0:
        assert len(service_list) == 1
        cli_logger.verbose("Successfully get the service: {} {}.", namespace, name)
        return service_list[0]

    cli_logger.verbose("Failed to get the service: {} {}", namespace, name)
    return None


def _delete_service(namespace: str, name: str):
    service_object = _get_service(namespace, name)
    if service_object is None:
        cli_logger.print(log_prefix + "Service: {} doesn't exist.".format(name))
        return

    cli_logger.print(log_prefix + "Deleting service: {}".format(name))
    core_api().delete_namespaced_service(name, namespace)
    cli_logger.print(log_prefix + "Successfully deleted service: {}".format(name))


def get_head_external_service_address(provider_config, namespace, cluster_name):
    service_name = _get_head_external_service_name(cluster_name)
    service = core_api().read_namespaced_service(namespace=namespace, name=service_name)
    ingress = service.status.load_balancer.ingress[0]
    if ingress.hostname:
        return ingress.hostname
    else:
        return ingress.ip


def with_kubernetes_environment_variables(provider_config, node_type_config: Dict[str, Any], node_id: str):
    config_dict = {}

    storage_config = provider_config.get("storage", {})

    if "aws_s3_storage" in storage_config:
        from cloudtik.providers._private._kubernetes.aws_eks.config import with_aws_environment_variables
        with_aws_environment_variables(provider_config, config_dict)

    if "gcp_cloud_storage" in storage_config:
        from cloudtik.providers._private._kubernetes.gcp_gke.config import with_gcp_environment_variables
        with_gcp_environment_variables(provider_config, config_dict)

    if "azure_cloud_storage" in storage_config:
        from cloudtik.providers._private._azure.utils import export_azure_cloud_storage_config
        export_azure_cloud_storage_config(provider_config, config_dict)

    return config_dict


def _get_cloud_provider_config(provider_config):
    if "cloud_provider" not in provider_config:
        return None
    cloud_provider = provider_config["cloud_provider"]

    if "type" not in cloud_provider:
        raise RuntimeError("Missing 'type' key for cloud provider configuration.")

    return cloud_provider


def _create_configurations_for_cloud_provider(config, namespace):
    provider_config = config["provider"]
    cloud_provider = _get_cloud_provider_config(provider_config)
    if cloud_provider is None:
        cli_logger.print("No cloud provider configured. Skipped cloud provider configurations.")
        return

    cloud_provider_type = cloud_provider["type"]

    cli_logger.print("Configuring {} cloud provider for Kubernetes.", cloud_provider_type)
    if cloud_provider_type == "aws":
        from cloudtik.providers._private._kubernetes.aws_eks.config import create_configurations_for_aws
        create_configurations_for_aws(config, namespace, cloud_provider)
    elif cloud_provider_type == "gcp":
        from cloudtik.providers._private._kubernetes.gcp_gke.config import create_configurations_for_gcp
        create_configurations_for_gcp(config, namespace, cloud_provider)
    else:
        cli_logger.print("No integration for {} cloud provider. Configuration skipped.", cloud_provider_type)


def _delete_configurations_for_cloud_provider(config, namespace,
                                              delete_managed_storage: bool = False):
    provider_config = config["provider"]
    cloud_provider = _get_cloud_provider_config(provider_config)
    if cloud_provider is None:
        cli_logger.print("No cloud provider configured. Skipped cloud provider configurations.")
        return
    cloud_provider_type = cloud_provider["type"]

    cli_logger.print("Configuring {} cloud provider for Kubernetes.", cloud_provider_type)
    if cloud_provider_type == "aws":
        from cloudtik.providers._private._kubernetes.aws_eks.config import delete_configurations_for_aws
        delete_configurations_for_aws(
            config, namespace, cloud_provider, delete_managed_storage)
    elif cloud_provider_type == "gcp":
        from cloudtik.providers._private._kubernetes.gcp_gke.config import delete_configurations_for_gcp
        delete_configurations_for_gcp(
            config, namespace, cloud_provider, delete_managed_storage)
    else:
        cli_logger.print("No integration for {} cloud provider. Configuration skipped.", cloud_provider_type)


def _configure_cloud_provider(config: Dict[str, Any], namespace):
    cloud_provider = _get_cloud_provider_config(config["provider"])
    if cloud_provider is None:
        cli_logger.print("No cloud provider configured. Skipped cloud provider configurations.")
        return
    cloud_provider_type = cloud_provider["type"]
    if cloud_provider_type == "aws":
        from cloudtik.providers._private._kubernetes.aws_eks.config import configure_kubernetes_for_aws
        configure_kubernetes_for_aws(config, namespace, cloud_provider)
    elif cloud_provider_type == "gcp":
        from cloudtik.providers._private._kubernetes.gcp_gke.config import configure_kubernetes_for_gcp
        configure_kubernetes_for_gcp(config, namespace, cloud_provider)
    else:
        cli_logger.verbose("No integration for {} cloud provider. Configuration skipped.", cloud_provider_type)


def _check_existence_for_cloud_provider(config: Dict[str, Any], namespace):
    provider_config = config["provider"]
    cloud_provider = _get_cloud_provider_config(provider_config)
    if cloud_provider is None:
        cli_logger.verbose("No cloud provider configured for Kubernetes.")
        return None
    cloud_provider_type = cloud_provider["type"]

    cli_logger.verbose("Getting existence for {} cloud provider for Kubernetes.", cloud_provider_type)
    if cloud_provider_type == "aws":
        from cloudtik.providers._private._kubernetes.aws_eks.config import check_existence_for_aws
        existence = check_existence_for_aws(config, namespace, cloud_provider)
    elif cloud_provider_type == "gcp":
        from cloudtik.providers._private._kubernetes.gcp_gke.config import check_existence_for_gcp
        existence = check_existence_for_gcp(config, namespace, cloud_provider)
    else:
        cli_logger.verbose("No integration for {} cloud provider.", cloud_provider_type)
        return None

    cli_logger.verbose("The existence status for {} cloud provider: {}.", cloud_provider_type, existence)
    return existence
