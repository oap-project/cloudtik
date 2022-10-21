import copy
import json
import logging
import time
import uuid
import subprocess
from pathlib import Path
import random

from typing import Any, Dict, Optional

from cloudtik.core.tags import CLOUDTIK_TAG_NODE_KIND, NODE_KIND_HEAD, CLOUDTIK_TAG_CLUSTER_NAME
from cloudtik.core._private.cli_logger import cli_logger, cf
from cloudtik.core._private.utils import check_cidr_conflict, is_use_internal_ip, _is_use_working_vpc, is_use_working_vpc, is_use_peering_vpc, \
    is_managed_cloud_storage, is_use_managed_cloud_storage, _is_use_managed_cloud_storage, update_nested_dict
from cloudtik.core.workspace_provider import Existence, CLOUDTIK_MANAGED_CLOUD_STORAGE, \
    CLOUDTIK_MANAGED_CLOUD_STORAGE_URI

from azure.mgmt.compute import ComputeManagementClient
from azure.core.exceptions import ResourceNotFoundError

from azure.storage.blob import BlobServiceClient
from azure.storage.filedatalake import DataLakeServiceClient

from cloudtik.providers._private._azure.utils import _get_node_info, get_azure_sdk_function, get_credential, \
    construct_resource_client, construct_network_client, construct_storage_client, _construct_storage_client, \
    construct_authorization_client, construct_manage_server_identity_client, construct_compute_client, \
    _construct_compute_client, _construct_resource_client, export_azure_cloud_storage_config, \
    get_azure_cloud_storage_config, get_azure_cloud_storage_config_for_update
from cloudtik.providers._private.utils import StorageTestingError

AZURE_RESOURCE_NAME_PREFIX = "cloudtik"
AZURE_MSI_NAME = AZURE_RESOURCE_NAME_PREFIX + "-msi-user-identity"
AZURE_WORKSPACE_RESOURCE_GROUP_NAME = AZURE_RESOURCE_NAME_PREFIX + "-{}-resource-group"
AZURE_WORKSPACE_VNET_NAME = AZURE_RESOURCE_NAME_PREFIX + "-{}-vnet"
AZURE_WORKSPACE_SUBNET_NAME = AZURE_RESOURCE_NAME_PREFIX + "-{}-{}-subnet"
AZURE_WORKSPACE_VNET_PEERING_NAME = AZURE_RESOURCE_NAME_PREFIX + "-{}-virtual-network-peering"
AZURE_WORKSPACE_STORAGE_ACCOUNT_NAME = AZURE_RESOURCE_NAME_PREFIX + "-{}-storage-account"
AZURE_WORKSPACE_STORAGE_CONTAINER_NAME = AZURE_RESOURCE_NAME_PREFIX + "-{}"
AZURE_WORKSPACE_NETWORK_SECURITY_GROUP_NAME = AZURE_RESOURCE_NAME_PREFIX + "-{}-network-security-group"
AZURE_WORKSPACE_PUBLIC_IP_ADDRESS_NAME = AZURE_RESOURCE_NAME_PREFIX + "-{}-public-ip-address"
AZURE_WORKSPACE_NAT_NAME = AZURE_RESOURCE_NAME_PREFIX + "-{}-nat"
AZURE_WORKSPACE_SECURITY_RULE_NAME = AZURE_RESOURCE_NAME_PREFIX + "-{}-security-rule-{}"
AZURE_WORKSPACE_WORKER_USI_NAME = AZURE_RESOURCE_NAME_PREFIX + "-{}-worker-user-assigned-identity"
AZURE_WORKSPACE_HEAD_USI_NAME = AZURE_RESOURCE_NAME_PREFIX + "-{}-user-assigned-identity"


AZURE_WORKSPACE_VERSION_TAG_NAME = "cloudtik-workspace-version"
AZURE_WORKSPACE_VERSION_CURRENT = "1"

AZURE_WORKSPACE_NUM_CREATION_STEPS = 9
AZURE_WORKSPACE_NUM_DELETION_STEPS = 9
AZURE_WORKSPACE_TARGET_RESOURCES = 12

AZURE_MANAGED_STORAGE_TYPE = "azure.managed.storage.type"
AZURE_MANAGED_STORAGE_ACCOUNT = "azure.managed.storage.account"
AZURE_MANAGED_STORAGE_CONTAINER = "azure.managed.storage.container"

logger = logging.getLogger(__name__)


def post_prepare_azure(config: Dict[str, Any]) -> Dict[str, Any]:
    try:
        config = fill_available_node_types_resources(config)
    except Exception as exc:
        cli_logger.warning(
            "Failed to detect node resources. Make sure you have properly configured the Azure credentials: {}.",
            str(exc))
        raise
    return config


def fill_available_node_types_resources(
        cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    """Fills out missing "resources" field for available_node_types."""
    if "available_node_types" not in cluster_config:
        return cluster_config
    cluster_config = copy.deepcopy(cluster_config)

    # Get instance information from cloud provider
    provider_config = cluster_config["provider"]
    subscription_id = provider_config["subscription_id"]
    vm_location = provider_config["location"]

    credential = get_credential(provider_config)
    compute_client = ComputeManagementClient(credential, subscription_id)

    vmsizes = compute_client.virtual_machine_sizes.list(vm_location)
    instances_dict = {
        instance.name: {"memory": instance.memory_in_mb, "cpu": instance.number_of_cores}
        for instance in vmsizes
    }

    # Update the instance information to node type
    available_node_types = cluster_config["available_node_types"]
    for node_type in available_node_types:
        instance_type = available_node_types[node_type]["node_config"]["azure_arm_parameters"]["vmSize"]
        if instance_type in instances_dict:
            cpus = instances_dict[instance_type]["cpu"]
            detected_resources = {"CPU": cpus}

            memory_total = instances_dict[instance_type]["memory"]
            memory_total_in_bytes = int(memory_total) * 1024 * 1024
            detected_resources["memory"] = memory_total_in_bytes

            detected_resources.update(
                available_node_types[node_type].get("resources", {}))
            if detected_resources != \
                    available_node_types[node_type].get("resources", {}):
                available_node_types[node_type][
                    "resources"] = detected_resources
                logger.debug("Updating the resources of {} to {}.".format(
                    node_type, detected_resources))
        else:
            raise ValueError("Instance type " + instance_type +
                             " is not available in Azure location: " +
                             vm_location + ".")
    return cluster_config


def get_workspace_resource_group_name(workspace_name):
    return AZURE_WORKSPACE_RESOURCE_GROUP_NAME.format(workspace_name)


def get_workspace_virtual_network_name(workspace_name):
    return AZURE_WORKSPACE_VNET_NAME.format(workspace_name)


def get_workspace_vnet_peering_name(workspace_name):
    return AZURE_WORKSPACE_VNET_PEERING_NAME.format(workspace_name)


def get_workspace_subnet_name(workspace_name, isPrivate=True):
    return AZURE_WORKSPACE_VNET_NAME.format(workspace_name, 'private') if isPrivate \
        else AZURE_WORKSPACE_VNET_NAME.format(workspace_name, 'public')


def get_workspace_storage_account_name(workspace_name):
    return AZURE_WORKSPACE_STORAGE_ACCOUNT_NAME.format(workspace_name)


def get_workspace_network_security_group_name(workspace_name):
    return AZURE_WORKSPACE_NETWORK_SECURITY_GROUP_NAME.format(workspace_name)


def get_workspace_public_ip_address_name(workspace_name):
    return AZURE_WORKSPACE_PUBLIC_IP_ADDRESS_NAME.format(workspace_name)


def get_workspace_nat_name(workspace_name):
    return AZURE_WORKSPACE_NAT_NAME.format(workspace_name)


def get_workspace_security_rule_name(workspace_name, suffix):
    return AZURE_WORKSPACE_SECURITY_RULE_NAME.format(workspace_name, suffix)


def get_workspace_worker_user_assigned_identity_name(workspace_name):
    return AZURE_WORKSPACE_WORKER_USI_NAME.format(workspace_name)


def get_workspace_head_user_assigned_identity_name(workspace_name):
    return AZURE_WORKSPACE_HEAD_USI_NAME.format(workspace_name)


def get_workspace_storage_container_name(workspace_name):
    return AZURE_WORKSPACE_STORAGE_CONTAINER_NAME.format(workspace_name)


def check_azure_workspace_existence(config):
    use_working_vpc = is_use_working_vpc(config)
    use_peering_vpc = is_use_peering_vpc(config)
    workspace_name = config["workspace_name"]
    managed_cloud_storage = is_managed_cloud_storage(config)
    network_client = construct_network_client(config)
    resource_client = construct_resource_client(config)

    existing_resources = 0
    target_resources = AZURE_WORKSPACE_TARGET_RESOURCES
    if managed_cloud_storage:
        target_resources += 1
    if use_peering_vpc:
        target_resources += 1

    """
         Do the work - order of operation
         Check resource group
         Check vpc
         Check network security group
         Check public IP address
         Check NAT gateway
         Check private subnet
         Check public subnet
         Check virtual network peering if needed
         Check role assignments
         Check user assigned identities
         Check cloud storage if need
    """
    resource_group_existence = False
    cloud_storage_existence = False
    resource_group_name = get_resource_group_name(config, resource_client, use_working_vpc)
    if resource_group_name is not None:
        existing_resources += 1
        resource_group_existence = True

        # Below resources are all depends on resource group
        virtual_network_name = get_virtual_network_name(config, resource_client, network_client, use_working_vpc)
        if virtual_network_name is not None:
            existing_resources += 1

            # Below resources are all depends on the virtual network
            if get_network_security_group(config, network_client, resource_group_name) is not None:
                existing_resources += 1

            if get_public_ip_address(config, network_client, resource_group_name) is not None:
                existing_resources += 1

            if get_nat_gateway(config, network_client, resource_group_name) is not None:
                existing_resources += 1

            private_subnet_name = get_workspace_subnet_name(workspace_name, isPrivate=True)
            if get_subnet(network_client, resource_group_name, virtual_network_name, private_subnet_name) is not None:
                existing_resources += 1

            public_subnet_name = get_workspace_subnet_name(workspace_name, isPrivate=False)
            if get_subnet(network_client, resource_group_name, virtual_network_name, public_subnet_name) is not None:
                existing_resources += 1

            if use_peering_vpc:
                virtual_network_peering_name = get_workspace_vnet_peering_name(workspace_name)
                if get_virtual_network_peering(network_client, resource_group_name, virtual_network_name, virtual_network_peering_name) is not None:
                    existing_resources += 1

        if get_head_user_assigned_identity(config, resource_group_name) is not None:
            existing_resources += 1
            if get_head_role_assignment_for_contributor(config, resource_group_name) is not None:
                existing_resources += 1

            if get_head_role_assignment_for_storage_blob_data_owner(config, resource_group_name) is not None:
                existing_resources += 1

        if get_worker_user_assigned_identity(config, resource_group_name) is not None:
            existing_resources += 1
            if get_worker_role_assignment_for_storage_blob_data_owner(config, resource_group_name) is not None:
                existing_resources += 1

        if managed_cloud_storage:
            if get_container_for_storage_account(config, resource_group_name) is not None:
                existing_resources += 1
                cloud_storage_existence = True

    if existing_resources == 0 or (
            existing_resources == 1 and resource_group_existence):
        return Existence.NOT_EXIST
    elif existing_resources == target_resources:
        return Existence.COMPLETED
    else:
        if existing_resources == 2 and cloud_storage_existence:
            return Existence.STORAGE_ONLY
        return Existence.IN_COMPLETED


def check_azure_workspace_integrity(config):
    existence = check_azure_workspace_existence(config)
    return True if existence == Existence.COMPLETED else False


def get_azure_workspace_info(config):
    workspace_name = config["workspace_name"]
    azure_cloud_storage = get_workspace_azure_storage(config, workspace_name)
    info = {}
    if azure_cloud_storage is not None:
        storage_uri = "abfs://{container}@{storage_account}.dfs.core.windows.net".format(
            container=azure_cloud_storage.get("azure.container"),
            storage_account=azure_cloud_storage.get("azure.storage.account")
        )
        managed_cloud_storage = {AZURE_MANAGED_STORAGE_TYPE: azure_cloud_storage.get("azure.storage.type"),
                                 AZURE_MANAGED_STORAGE_ACCOUNT: azure_cloud_storage.get("azure.storage.account"),
                                 AZURE_MANAGED_STORAGE_CONTAINER: azure_cloud_storage.get("azure.container"),
                                 CLOUDTIK_MANAGED_CLOUD_STORAGE_URI: storage_uri}
        info[CLOUDTIK_MANAGED_CLOUD_STORAGE] = managed_cloud_storage

    return info


def get_resource_group_name(config, resource_client, use_working_vpc):
    return _get_resource_group_name(
        config.get("workspace_name"), resource_client, use_working_vpc)


def get_virtual_network_name(config, resource_client, network_client, use_working_vpc):
    if use_working_vpc:
        virtual_network_name = get_working_node_virtual_network_name(resource_client, network_client)
    else:
        virtual_network = get_workspace_virtual_network(config, network_client)
        virtual_network_name = None if virtual_network is None else virtual_network.name

    return virtual_network_name


def update_azure_workspace_firewalls(config):
    resource_client = construct_resource_client(config)
    network_client = construct_network_client(config)
    workspace_name = config["workspace_name"]
    use_working_vpc = is_use_working_vpc(config)
    resource_group_name = get_resource_group_name(config, resource_client, use_working_vpc)

    if resource_group_name is None:
        cli_logger.print("Workspace: {} doesn't exist!".format(config["workspace_name"]))
        return

    current_step = 1
    total_steps = 1

    try:

        with cli_logger.group(
                "Updating workspace firewalls",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _create_or_update_network_security_group(config, network_client, resource_group_name)
            
    except Exception as e:
        cli_logger.error(
            "Failed to update the firewalls of workspace {}. {}", workspace_name, str(e))
        raise e

    cli_logger.print(
        "Successfully updated the firewalls of workspace: {}.",
        cf.bold(workspace_name))


def delete_azure_workspace(config, delete_managed_storage: bool = False):
    resource_client = construct_resource_client(config)
    workspace_name = config["workspace_name"]
    use_working_vpc = is_use_working_vpc(config)
    use_peering_vpc = is_use_peering_vpc(config)
    managed_cloud_storage = is_managed_cloud_storage(config)
    resource_group_name = get_resource_group_name(config, resource_client, use_working_vpc)

    if resource_group_name is None:
        cli_logger.print("Workspace: {} doesn't exist!".format(config["workspace_name"]))
        return

    current_step = 1
    total_steps = AZURE_WORKSPACE_NUM_DELETION_STEPS
    if managed_cloud_storage and delete_managed_storage:
        total_steps += 1
    if use_peering_vpc:
        total_steps += 1

    try:
        # delete network resources
        with cli_logger.group("Deleting workspace: {}", workspace_name):
            # Delete the resources in a reverse way of creating

            if managed_cloud_storage and delete_managed_storage:
                with cli_logger.group(
                        "Deleting Azure storage account",
                        _numbered=("[]", current_step, total_steps)):
                    current_step += 1
                    _delete_workspace_cloud_storage(config, resource_group_name)

            # delete role_assignments
            with cli_logger.group(
                    "Deleting role assignments for managed identity",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _delete_role_assignments(config, resource_group_name)

            # delete user_assigned_identities
            with cli_logger.group(
                    "Deleting user assigned identities",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _delete_user_assigned_identities(config, resource_group_name)

            current_step = _delete_network_resources(
                config, resource_client, resource_group_name, current_step, total_steps)

            # delete resource group
            with cli_logger.group(
                    "Deleting resource group",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _delete_resource_group(config, resource_client)
    except Exception as e:
        cli_logger.error(
            "Failed to delete workspace {}. {}", workspace_name, str(e))
        raise e

    cli_logger.print(
        "Successfully deleted workspace: {}.",
        cf.bold(workspace_name))


def _delete_network_resources(config, resource_client, resource_group_name, current_step, total_steps):
    use_working_vpc = is_use_working_vpc(config)
    use_peering_vpc = is_use_peering_vpc(config)
    network_client = construct_network_client(config)
    virtual_network_name = get_virtual_network_name(config, resource_client, network_client, use_working_vpc)

    """
         Do the work - order of operation
         Delete virtual network peering if needed
         Delete public subnet
         Delete private subnet 
         Delete NAT gateway
         Delete public IP address
         Delete network security group
         Delete vpc
    """

    # delete vpc peering connection
    if use_peering_vpc:
        with cli_logger.group(
                "Deleting virtual network peering connections",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _delete_vnet_peering_connections(config, resource_client, network_client)

    # delete public subnets
    with cli_logger.group(
            "Deleting public subnet",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_subnet(config, network_client, resource_group_name, virtual_network_name, is_private=False)

    # delete private subnets
    with cli_logger.group(
            "Deleting private subnet",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_subnet(config, network_client, resource_group_name, virtual_network_name, is_private=True)

    # delete NAT gateway
    with cli_logger.group(
            "Deleting NAT gateway",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_nat(config, network_client, resource_group_name)

    # delete public IP address
    with cli_logger.group(
            "Deleting public IP address",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_public_ip_address(config, network_client, resource_group_name)

    # delete network security group
    with cli_logger.group(
            "Deleting network security group",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_network_security_group(config, network_client, resource_group_name)

    # delete virtual network
    with cli_logger.group(
            "Deleting VPC",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_vnet(config, resource_client, network_client)

    return current_step


def get_container_for_storage_account(config, resource_group_name):
    workspace_name = config["workspace_name"]
    container_name = get_workspace_storage_container_name(workspace_name)
    storage_client = construct_storage_client(config)
    storage_account = get_storage_account(config)
    if storage_account is None:
        return None

    cli_logger.verbose("Getting the workspace container: {}.".format(container_name))
    containers = list(storage_client.blob_containers.list(
        resource_group_name=resource_group_name, account_name=storage_account.name))
    workspace_containers = [container for container in containers
                                  if container.name == container_name]

    if len(workspace_containers) > 0:
        container = workspace_containers[0]
        cli_logger.verbose("Successfully get the workspace container: {}.".format(container.name))
        return container

    cli_logger.verbose("Failed to get the container in storage account: {}", storage_account.name)
    return None


def get_storage_account(config):
    workspace_name = config["workspace_name"]
    storage_client = construct_storage_client(config)
    storage_account_name = get_workspace_storage_account_name(workspace_name)

    cli_logger.verbose("Getting the workspace storage account: {}.".format(storage_account_name))
    storage_accounts = list(storage_client.storage_accounts.list())
    workspace_storage_accounts = [storage_account for storage_account in storage_accounts
                                 for key, value in storage_account.tags.items()
                                 if key == "Name" and value == storage_account_name]

    if len(workspace_storage_accounts) > 0:
        storage_account = workspace_storage_accounts[0]
        cli_logger.verbose("Successfully get the workspace storage account: {}.".format(storage_account.name))
        return storage_account

    cli_logger.verbose("Failed to get the storage account for workspace")
    return None


def has_storage_account(config, resource_group_name) -> bool:
    storage_client = construct_storage_client(config)
    storage_accounts = list(storage_client.storage_accounts.list_by_resource_group(
        resource_group_name=resource_group_name))
    if len(storage_accounts) > 0:
        return True

    return False


def _get_container(provider_config, resource_group_name, storage_account_name, container_name):
    storage_client = _construct_storage_client(provider_config)
    container = storage_client.blob_containers.get(
        resource_group_name=resource_group_name,
        account_name=storage_account_name,
        container_name=container_name)
    return container


def _delete_workspace_cloud_storage(config, resource_group_name):
    storage_client = construct_storage_client(config)
    storage_account = get_storage_account(config)
    if storage_account is None:
        cli_logger.print("The storage account doesn't exist.")
        return

    """ Delete storage account """
    cli_logger.print("Deleting the storage account: {}...".format(storage_account.name))
    try:
        storage_client.storage_accounts.delete(
            resource_group_name=resource_group_name,
            account_name=storage_account.name)
        cli_logger.print("Successfully deleted the storage account: {}.".format(storage_account.name))
    except Exception as e:
        cli_logger.error(
            "Failed to delete the storage account: {}. {}", storage_account.name, str(e))
        raise e


def get_head_role_assignment_for_storage_blob_data_owner(config, resource_group_name):
    role_assignment_name = get_role_assignment_name_for_storage_blob_data_owner(config, "head")
    return get_role_assignment_for_storage_blob_data_owner(
        config, resource_group_name, role_assignment_name)


def get_worker_role_assignment_for_storage_blob_data_owner(config, resource_group_name):
    role_assignment_name = get_role_assignment_name_for_storage_blob_data_owner(config, "worker")
    return get_role_assignment_for_storage_blob_data_owner(
        config, resource_group_name, role_assignment_name)


def get_role_assignment_name_for_storage_blob_data_owner(config, role_type):
    workspace_name = config["workspace_name"]
    subscription_id = config["provider"].get("subscription_id")
    role_assignment_name = str(uuid.uuid3(uuid.UUID(subscription_id),
                                          workspace_name + role_type + "storage_blob_data_owner"))
    return role_assignment_name


def get_role_assignment_for_storage_blob_data_owner(config, resource_group_name, role_assignment_name):
    authorization_client = construct_authorization_client(config)
    subscription_id = config["provider"].get("subscription_id")
    scope = "subscriptions/{subscriptionId}/resourcegroups/{resourceGroupName}".format(
        subscriptionId=subscription_id,
        resourceGroupName=resource_group_name
    )

    cli_logger.verbose("Getting the existing role assignment for Storage Blob Data Owner: {}.",
                       role_assignment_name)

    try:
        role_assignment = authorization_client.role_assignments.get(
            scope=scope,
            role_assignment_name=role_assignment_name,
        )
        cli_logger.verbose("Successfully get the role assignment for Storage Blob Data Owner: {}.".
                           format(role_assignment_name))
        return role_assignment
    except Exception as e:
        cli_logger.error("Failed to get the role assignment. {}", str(e))
        return None


def get_head_role_assignment_for_contributor(config, resource_group_name):
    workspace_name = config["workspace_name"]
    subscription_id = config["provider"].get("subscription_id")
    role_assignment_name = str(uuid.uuid3(uuid.UUID(subscription_id), workspace_name + "contributor"))
    cli_logger.verbose("Getting the existing role assignment for Contributor: {}.", role_assignment_name)

    authorization_client = construct_authorization_client(config)
    scope = "subscriptions/{subscriptionId}/resourcegroups/{resourceGroupName}".format(
        subscriptionId=subscription_id,
        resourceGroupName=resource_group_name
    )
    try:
        role_assignment = authorization_client.role_assignments.get(
            scope=scope,
            role_assignment_name=role_assignment_name,
        )
        cli_logger.verbose("Successfully get the role assignment for Contributor: {}.".
                           format(role_assignment_name))
        return role_assignment_name
    except Exception as e:
        cli_logger.error(
            "Failed to get the role assignment for Contributor. {}", str(e))
        return None


def _delete_head_role_assignment_for_storage_blob_data_owner(config, resource_group_name):
    _delete_role_assignment_for_storage_blob_data_owner(
        config, resource_group_name, "head")


def _delete_worker_role_assignment_for_storage_blob_data_owner(config, resource_group_name):
    _delete_role_assignment_for_storage_blob_data_owner(
        config, resource_group_name, "worker")


def _delete_role_assignment_for_storage_blob_data_owner(config, resource_group_name, role_type):
    role_assignment_name = get_role_assignment_name_for_storage_blob_data_owner(config, role_type)
    role_assignment = get_role_assignment_for_storage_blob_data_owner(
        config, resource_group_name, role_assignment_name)
    if role_assignment is None:
        cli_logger.print("The role assignment {} doesn't exist.".format(role_assignment_name))
        return

    """ Delete the role_assignment """
    cli_logger.print("Deleting the role assignment for Storage Blob Data Owner: {}...".format(
        role_assignment_name))
    try:
        authorization_client = construct_authorization_client(config)
        subscription_id = config["provider"].get("subscription_id")
        scope = "subscriptions/{subscriptionId}/resourcegroups/{resourceGroupName}".format(
            subscriptionId=subscription_id,
            resourceGroupName=resource_group_name
        )

        authorization_client.role_assignments.delete(
            scope=scope,
            role_assignment_name=role_assignment_name
        )
        cli_logger.print("Successfully deleted the role assignment for Storage Blob Data Owner: {}.".format(
            role_assignment_name))
    except Exception as e:
        cli_logger.error(
            "Failed to delete the role assignment for Storage Blob Data Owner: {}. {}", role_assignment_name, str(e))
        raise e


def _delete_head_role_assignment_for_contributor(config, resource_group_name):
    role_assignment_name = get_head_role_assignment_for_contributor(config, resource_group_name)
    if role_assignment_name is None:
        cli_logger.print("The role assignment doesn't exist.")
        return

    authorization_client = construct_authorization_client(config)
    subscription_id = config["provider"].get("subscription_id")
    scope = "subscriptions/{subscriptionId}/resourcegroups/{resourceGroupName}".format(
        subscriptionId=subscription_id,
        resourceGroupName=resource_group_name
    )

    """ Delete the role_assignment"""
    cli_logger.print("Deleting the role assignment for Contributor: {}...".format(role_assignment_name))
    try:
        authorization_client.role_assignments.delete(
            scope=scope,
            role_assignment_name=role_assignment_name
        )
        cli_logger.print("Successfully deleted the role assignment for Contributor: {}.".format(role_assignment_name))
    except Exception as e:
        cli_logger.error(
            "Failed to delete the role assignment for Contributor: {}. {}", role_assignment_name, str(e))
        raise e


def _delete_role_assignments(config, resource_group_name):
    current_step = 1
    total_steps = 3

    with cli_logger.group(
            "Deleting Contributor role assignment for head",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_head_role_assignment_for_contributor(config, resource_group_name)

    with cli_logger.group(
            "Deleting Storage Blob Data Owner role assignment for head",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_head_role_assignment_for_storage_blob_data_owner(config, resource_group_name)

    with cli_logger.group(
            "Deleting Storage Blob Data Owner role assignment for worker",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_worker_role_assignment_for_storage_blob_data_owner(config, resource_group_name)


def get_head_user_assigned_identity(config, resource_group_name):
    user_assigned_identity_name = _get_head_user_assigned_identity_name(config)
    return get_user_assigned_identity(config, resource_group_name, user_assigned_identity_name)


def get_worker_user_assigned_identity(config, resource_group_name):
    user_assigned_identity_name = _get_worker_user_assigned_identity_name(config)
    return get_user_assigned_identity(config, resource_group_name, user_assigned_identity_name)


def get_user_assigned_identity(config, resource_group_name, user_assigned_identity_name):
    msi_client = construct_manage_server_identity_client(config)
    cli_logger.verbose("Getting the existing user assigned identity: {}.".format(user_assigned_identity_name))
    try:
        user_assigned_identity = msi_client.user_assigned_identities.get(
            resource_group_name,
            user_assigned_identity_name
        )
        cli_logger.verbose("Successfully get the user assigned identity: {}.".format(user_assigned_identity_name))
        return user_assigned_identity
    except ResourceNotFoundError as e:
        cli_logger.verbose_error(
            "Failed to get the user assigned identity: {}. {}", user_assigned_identity_name, str(e))
        return None


def _delete_user_assigned_identities(config, resource_group_name):
    current_step = 1
    total_steps = 2

    with cli_logger.group(
            "Deleting user assigned identity for head",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_user_assigned_identity_for_head(config, resource_group_name)

    with cli_logger.group(
            "Deleting user assigned identity for worker",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_user_assigned_identity_for_worker(config, resource_group_name)


def _delete_user_assigned_identity_for_head(config, resource_group_name):
    user_assigned_identity_name = _get_head_user_assigned_identity_name(config)
    _delete_user_assigned_identity(config, resource_group_name, user_assigned_identity_name)


def _delete_user_assigned_identity_for_worker(config, resource_group_name):
    worker_user_assigned_identity_name = _get_worker_user_assigned_identity_name(config)
    _delete_user_assigned_identity(config, resource_group_name, worker_user_assigned_identity_name)


def _delete_user_assigned_identity(config, resource_group_name, user_assigned_identity_name):
    user_assigned_identity = get_user_assigned_identity(config, resource_group_name, user_assigned_identity_name)
    msi_client = construct_manage_server_identity_client(config)
    if user_assigned_identity is None:
        cli_logger.print("The user assigned identity doesn't exist: {}.".format(user_assigned_identity_name))
        return

    """ Delete the user_assigned_identity """
    cli_logger.print("Deleting the user assigned identity: {}...".format(user_assigned_identity.name))
    try:
        msi_client.user_assigned_identities.delete(
            resource_group_name=resource_group_name,
            resource_name=user_assigned_identity.name
        )
        cli_logger.print("Successfully deleted the user assigned identity: {}.".format(user_assigned_identity.name))
    except Exception as e:
        cli_logger.error(
            "Failed to delete the user assigned identity: {}. {}", user_assigned_identity.name, str(e))
        raise e


def get_network_security_group(config, network_client, resource_group_name):
    network_security_group_name = get_workspace_network_security_group_name(config["workspace_name"])

    cli_logger.verbose("Getting the existing network security group: {}.".format(network_security_group_name))
    try:
        network_client.network_security_groups.get(
            resource_group_name,
            network_security_group_name
        )
        cli_logger.verbose("Successfully get the network security group: {}.".format(network_security_group_name))
        return network_security_group_name
    except ResourceNotFoundError as e:
        cli_logger.verbose_error("Failed to get the network security group: {}. {}",
                                 network_security_group_name, str(e))
        return None


def _delete_network_security_group(config, network_client, resource_group_name):
    network_security_group_name = get_network_security_group(config, network_client, resource_group_name)
    if network_security_group_name is None:
        cli_logger.print("The network security group doesn't exist.")
        return

    # Delete the network security group
    cli_logger.print("Deleting the network security group: {}...".format(network_security_group_name))
    try:
        network_client.network_security_groups.begin_delete(
            resource_group_name=resource_group_name,
            network_security_group_name=network_security_group_name
        ).result()
        cli_logger.print("Successfully deleted the network security group: {}.".format(network_security_group_name))
    except Exception as e:
        cli_logger.error("Failed to delete the network security group: {}. {}", network_security_group_name, str(e))
        raise e


def get_public_ip_address(config, network_client, resource_group_name):
    public_ip_address_name = get_workspace_public_ip_address_name(config["workspace_name"])

    cli_logger.verbose("Getting the existing public IP address: {}.".format(public_ip_address_name))
    try:
        network_client.public_ip_addresses.get(
            resource_group_name,
            public_ip_address_name
        )
        cli_logger.verbose("Successfully get the public IP address: {}.".format(public_ip_address_name))
        return public_ip_address_name
    except ResourceNotFoundError as e:
        cli_logger.verbose_error("Failed to get the public IP address: {}. {}",
                                 public_ip_address_name, str(e))
        return None


def _delete_public_ip_address(config, network_client, resource_group_name):
    public_ip_address_name = get_public_ip_address(config, network_client, resource_group_name)
    if public_ip_address_name is None:
        cli_logger.print("The public IP address doesn't exist.")
        return

    # Delete the public IP address
    cli_logger.print("Deleting the public IP address: {}...".format(public_ip_address_name))
    try:
        network_client.public_ip_addresses.begin_delete(
            resource_group_name=resource_group_name,
            public_ip_address_name=public_ip_address_name
        ).result()
        cli_logger.print("Successfully deleted the public IP address: {}.".format(public_ip_address_name))
    except Exception as e:
        cli_logger.error("Failed to delete the public IP address: {}. {}", public_ip_address_name, str(e))
        raise e


def get_nat_gateway(config, network_client, resource_group_name):
    nat_gateway_name = get_workspace_nat_name(config["workspace_name"])

    cli_logger.verbose("Getting the existing NAT gateway: {}.".format(nat_gateway_name))
    try:
        network_client.nat_gateways.get(
            resource_group_name,
            nat_gateway_name
        )
        cli_logger.verbose("Successfully get the NAT gateway: {}.".format(nat_gateway_name))
        return nat_gateway_name
    except ResourceNotFoundError as e:
        cli_logger.verbose_error("Failed to get the NAT gateway: {}. {}", nat_gateway_name, str(e))
        return None


def _delete_nat(config, network_client, resource_group_name):
    nat_gateway_name = get_nat_gateway(config, network_client, resource_group_name)
    if nat_gateway_name is None:
        cli_logger.print("The Nat Gateway doesn't exist.")
        return

    """ Delete the Nat Gateway """
    cli_logger.print("Deleting the Nat Gateway: {}...".format(nat_gateway_name))
    try:
        network_client.nat_gateways.begin_delete(
            resource_group_name=resource_group_name,
            nat_gateway_name=nat_gateway_name
        ).result()
        cli_logger.print("Successfully deleted the Nat Gateway: {}.".format(nat_gateway_name))
    except Exception as e:
        cli_logger.error("Failed to delete the Nat Gateway: {}. {}", nat_gateway_name, str(e))
        raise e


def _delete_vnet(config, resource_client, network_client):
    use_working_vpc = is_use_working_vpc(config)
    if use_working_vpc:
        cli_logger.print("Will not delete the current node virtual network.")
        return

    resource_group_name = get_resource_group_name(config, resource_client, use_working_vpc)
    virtual_network_name = get_virtual_network_name(config, resource_client, network_client, use_working_vpc)
    if virtual_network_name is None:
        cli_logger.print("The virtual network: {} doesn't exist.".
                         format(virtual_network_name))
        return

    # Delete the virtual network
    cli_logger.print("Deleting the virtual network: {}...".format(virtual_network_name))
    try:
        network_client.virtual_networks.begin_delete(
            resource_group_name=resource_group_name,
            virtual_network_name=virtual_network_name
        ).result()
        cli_logger.print("Successfully deleted the virtual network: {}.".format(virtual_network_name))
    except Exception as e:
        cli_logger.error("Failed to delete the virtual network: {}. {}", virtual_network_name, str(e))
        raise e


def _delete_resource_group(config, resource_client):
    use_working_vpc = is_use_working_vpc(config)
    if use_working_vpc:
        cli_logger.print("Will not delete the current node resource group.")
        return

    resource_group = _get_workspace_resource_group(config["workspace_name"], resource_client)
    if resource_group is None:
        cli_logger.print("The resource group doesn't exist. Skip deletion.")
        return

    resource_group_name = resource_group.name
    if has_storage_account(config, resource_group_name):
        cli_logger.print("The resource group {} has remaining storage accounts. Will not be deleted.".
                         format(resource_group_name))
        return

    # Delete the resource group
    cli_logger.print("Deleting the resource group: {}...".format(resource_group_name))

    try:
        resource_client.resource_groups.begin_delete(
            resource_group_name
        ).result()
        cli_logger.print("Successfully deleted the resource group: {}.".format(resource_group_name))
    except Exception as e:
        cli_logger.error("Failed to delete the resource group: {}. {}", resource_group_name, str(e))
        raise e


def create_azure_workspace(config):
    config = copy.deepcopy(config)
    config = _create_workspace(config)
    return config


def _create_workspace(config):
    workspace_name = config["workspace_name"]
    managed_cloud_storage = is_managed_cloud_storage(config)
    use_peering_vpc = is_use_peering_vpc(config)
    current_step = 1
    total_steps = AZURE_WORKSPACE_NUM_CREATION_STEPS
    if managed_cloud_storage:
        total_steps += 2
    if use_peering_vpc:
        total_steps += 1

    resource_client = construct_resource_client(config)

    try:
        with cli_logger.group("Creating workspace: {}", workspace_name):
            # create resource group
            with cli_logger.group(
                    "Creating resource group",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                resource_group_name = _create_resource_group(config, resource_client)

            # create network resources
            current_step = _create_network_resources(config, resource_group_name, current_step, total_steps)

            # create user_assigned_identities
            with cli_logger.group(
                    "Creating user assigned identities",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _create_user_assigned_identities(config, resource_group_name)

            # create role assignments
            with cli_logger.group(
                    "Creating role assignments for managed identity",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _create_role_assignments(config, resource_group_name)

            if managed_cloud_storage:
                with cli_logger.group(
                        "Creating storage account",
                        _numbered=("[]", current_step, total_steps)):
                    current_step += 1
                    _create_storage_account(config, resource_group_name)

                with cli_logger.group(
                        "Creating container for storage account",
                        _numbered=("[]", current_step, total_steps)):
                    current_step += 1
                    _create_container_for_storage_account(config, resource_group_name)

    except Exception as e:
        cli_logger.error("Failed to create workspace with the name {}. "
                         "You need to delete and try create again. {}", workspace_name, str(e))
        raise e

    cli_logger.print(
        "Successfully created workspace: {}.",
        cf.bold(workspace_name))

    return config


def _create_resource_group(config, resource_client):
    workspace_name = config["workspace_name"]
    use_working_vpc = is_use_working_vpc(config)

    if use_working_vpc:
        # No need to create new resource group
        resource_group_name = get_working_node_resource_group_name(resource_client)
        if resource_group_name is None:
            cli_logger.abort("Failed to get the resource group for the current machine. "
                             "Please make sure your current machine is an Azure virtual machine "
                             "to use use_internal_ips=True with use_working_vpc=True.")
        else:
            cli_logger.print("Will use the current node resource group: {}.", resource_group_name)
    else:
        # Need to create a new resource_group
        resource_group = _get_workspace_resource_group(workspace_name, resource_client)
        if resource_group is None:
            resource_group = create_resource_group(config, resource_client)
        else:
            cli_logger.print("Resource group {} for workspace already exists. Skip creation.", resource_group.name)
        resource_group_name = resource_group.name
    return resource_group_name


def get_azure_instance_metadata():
    # This function only be used on Azure instances.
    try:
        subprocess.run(
            "curl -sL -H \"metadata:true\" \"http://169.254.169.254/metadata/instance?api-version=2020-09-01\""
            " > /tmp/azure_instance_metadata.json", shell=True)
        with open('/tmp/azure_instance_metadata.json', 'r', encoding='utf8') as fp:
            metadata = json.load(fp)
        subprocess.run("rm -rf /tmp/azure_instance_metadata.json", shell=True)
        return metadata
    except Exception as e:
        cli_logger.verbose_error("Failed to get instance metadata: {}", str(e))
        return None


def get_working_node_resource_group_name(resource_client):
    resource_group = get_working_node_resource_group(resource_client)
    return None if resource_group is None else resource_group.name


def get_working_node_resource_group(resource_client):
    metadata = get_azure_instance_metadata()
    if metadata is None:
        cli_logger.error("Failed to get the metadata of the working node. "
                         "Please check whether the working node is a Azure instance or not!")
        return None
    resource_group_name = metadata.get("compute", {}).get("resourceGroupName", "")
    try:
        resource_group = resource_client.resource_groups.get(
            resource_group_name
        )
        cli_logger.verbose(
            "Successfully get the resource group: {} for working node.".format(resource_group_name))
        return resource_group
    except ResourceNotFoundError as e:
        cli_logger.verbose_error(
            "The resource group for working node is not found: {}", str(e))
        return None


def get_virtual_network_name_by_subnet(resource_client, network_client, resource_group_name, subnet):
    subnet_address_prefix = subnet['address'] + "/" + subnet['prefix']
    virtual_networks_resources = list(resource_client.resources.list_by_resource_group(
        resource_group_name=resource_group_name, filter="resourceType eq 'Microsoft.Network/virtualNetworks'"))
    virtual_network_names = [virtual_network_resources.name for virtual_network_resources in virtual_networks_resources]
    virtual_networks = [network_client.virtual_networks.get(
        resource_group_name=resource_group_name, virtual_network_name=virtual_network_name)
        for virtual_network_name in virtual_network_names]

    for virtual_network in virtual_networks:
        for subnet in virtual_network.subnets:
            if subnet.address_prefix == subnet_address_prefix:
                return virtual_network.name

    return None


def get_working_node_virtual_network_name(resource_client, network_client):
    metadata = get_azure_instance_metadata()
    if metadata is None:
        cli_logger.error("Failed to get the metadata of the working node. "
                         "Please check whether the working node is a Azure instance or not!")
        return None
    resource_group_name = metadata.get("compute", {}).get("resourceGroupName", "")
    interfaces = metadata.get("network", {}).get("interface", "")
    subnet = interfaces[0]["ipv4"]["subnet"][0]
    virtual_network_name = get_virtual_network_name_by_subnet(resource_client, network_client, resource_group_name, subnet)
    if virtual_network_name is not None:
        cli_logger.print("Successfully get the VirtualNetworkName for working node.")

    return virtual_network_name


def get_virtual_network(resource_group_name, virtual_network_name, network_client):
    try:
        virtual_network = network_client.virtual_networks.get(
            resource_group_name=resource_group_name,
            virtual_network_name=virtual_network_name
        )
        cli_logger.verbose("Successfully get the VirtualNetwork: {}.".
                                 format(virtual_network.name))
        return virtual_network
    except ResourceNotFoundError as e:
        cli_logger.verbose_error(
            "The virtual network {} is not found: {}", virtual_network_name, str(e))
        return None


def get_workspace_virtual_network(config, network_client):
    workspace_name = config["workspace_name"]
    resource_group_name = get_workspace_resource_group_name(workspace_name)
    virtual_network_name = get_workspace_virtual_network_name(workspace_name)
    cli_logger.verbose("Getting the VirtualNetworkName for workspace: {}...".
                       format(virtual_network_name))

    virtual_network = get_virtual_network(resource_group_name, virtual_network_name, network_client)
    return virtual_network


def _get_resource_group_name(
        workspace_name, resource_client, use_working_vpc):
    resource_group = _get_resource_group(workspace_name, resource_client, use_working_vpc)
    return None if resource_group is None else resource_group.name


def _get_resource_group(
        workspace_name, resource_client, use_working_vpc):
    if use_working_vpc:
        resource_group = get_working_node_resource_group(resource_client)
    else:
        resource_group = _get_workspace_resource_group(workspace_name, resource_client)

    return resource_group


def _get_workspace_resource_group(workspace_name, resource_client):
    resource_group_name = get_workspace_resource_group_name(workspace_name)
    cli_logger.verbose("Getting the resource group name for workspace: {}...".
                       format(resource_group_name))

    try:
        resource_group = resource_client.resource_groups.get(
            resource_group_name
        )
        cli_logger.verbose(
            "Successfully get the resource group name: {} for workspace.".format(resource_group_name))
        return resource_group
    except ResourceNotFoundError as e:
        cli_logger.verbose_error(
            "The resource group for workspace is not found: {}", str(e))
        return None


def create_resource_group(config, resource_client):
    resource_group_name = get_workspace_resource_group_name(config["workspace_name"])

    assert "location" in config["provider"], (
        "Provider config must include location field")
    params = {"location": config["provider"]["location"]}
    cli_logger.print("Creating workspace resource group: {} on Azure...", resource_group_name)
    # create resource group
    try:
        resource_group = resource_client.resource_groups.create_or_update(
            resource_group_name=resource_group_name, parameters=params)
        cli_logger.print("Successfully created workspace resource group: {}.",
                         get_workspace_resource_group_name(config["workspace_name"]))
        return resource_group
    except Exception as e:
        cli_logger.error(
            "Failed to create workspace resource group. {}", str(e))
        raise e


def _create_head_role_assignment_for_storage_blob_data_owner(config, resource_group_name):
    user_assigned_identity = get_head_user_assigned_identity(config, resource_group_name)
    _create_role_assignment_for_storage_blob_data_owner(
        config, resource_group_name, user_assigned_identity, "head")


def _create_worker_role_assignment_for_storage_blob_data_owner(config, resource_group_name):
    user_assigned_identity = get_worker_user_assigned_identity(config, resource_group_name)
    _create_role_assignment_for_storage_blob_data_owner(
        config, resource_group_name, user_assigned_identity, "worker")


def _create_role_assignment_for_storage_blob_data_owner(
        config, resource_group_name, user_assigned_identity, role_type):
    role_assignment_name = get_role_assignment_name_for_storage_blob_data_owner(config, role_type)
    cli_logger.print("Creating workspace role assignment for Storage Blob Data Owner: {} on Azure...",
                     role_assignment_name)

    authorization_client = construct_authorization_client(config)
    subscription_id = config["provider"].get("subscription_id")
    scope = "subscriptions/{subscriptionId}/resourcegroups/{resourceGroupName}".format(
        subscriptionId=subscription_id,
        resourceGroupName=resource_group_name
    )
    # Create role assignment for Storage Blob Data Owner
    try:
        role_assignment = authorization_client.role_assignments.create(
            scope=scope,
            role_assignment_name=role_assignment_name,
            parameters={
                "role_definition_id": "/providers/Microsoft.Authorization/roleDefinitions/b7e6dc6d-f1e8-4753-8033-0f276bb0955b",
                "principal_id": user_assigned_identity.principal_id,
                "principalType": "ServicePrincipal"
            }
        )
        cli_logger.print("Successfully created workspace role assignment for Storage Blob Data Owner: {}.".
                         format(role_assignment_name))
    except Exception as e:
        cli_logger.error(
            "Failed to create workspace role assignment for Storage Blob Data Owner. {}", str(e))
        raise e


def _create_head_role_assignment_for_contributor(config, resource_group_name):
    workspace_name = config["workspace_name"]
    subscription_id = config["provider"].get("subscription_id")
    role_assignment_name = str(uuid.uuid3(uuid.UUID(subscription_id), workspace_name + "contributor"))

    cli_logger.print("Creating workspace role assignment: {} on Azure...", role_assignment_name)

    authorization_client = construct_authorization_client(config)
    scope = "subscriptions/{subscriptionId}/resourcegroups/{resourceGroupName}".format(
        subscriptionId=subscription_id,
        resourceGroupName=resource_group_name
    )
    user_assigned_identity = get_head_user_assigned_identity(config, resource_group_name)
    # Create role assignment
    try:
        role_assignment = authorization_client.role_assignments.create(
            scope=scope,
            role_assignment_name=role_assignment_name,
            parameters={
                "role_definition_id": "/subscriptions/{}/providers/Microsoft.Authorization/roleDefinitions/b24988ac-6180-42a0-ab88-20f7382dd24c".format(
                    subscription_id),
                "principal_id": user_assigned_identity.principal_id,
                "principalType": "ServicePrincipal"
            }
        )
        cli_logger.print("Successfully created workspace role assignment: {}.".
                         format(role_assignment_name))
    except Exception as e:
        cli_logger.error(
            "Failed to create workspace role assignment. {}", str(e))
        raise e


def _create_role_assignments(config, resource_group_name):
    current_step = 1
    total_steps = 3

    # create Contributor role assignment for head
    with cli_logger.group(
            "Creating Contributor role assignment for head ",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_head_role_assignment_for_contributor(config, resource_group_name)

    # create Storage Blob Data Owner role assignment for head
    with cli_logger.group(
            "Creating Storage Blob Data Owner role assignment for head",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_head_role_assignment_for_storage_blob_data_owner(config, resource_group_name)

    # create Storage Blob Data Owner role assignment for worker
    with cli_logger.group(
            "Creating Storage Blob Data Owner role assignment for worker",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_worker_role_assignment_for_storage_blob_data_owner(config, resource_group_name)


def _create_container_for_storage_account(config, resource_group_name):
    workspace_name = config["workspace_name"]
    container_name = get_workspace_storage_container_name(workspace_name)
    storage_account = get_storage_account(config)
    if storage_account is None:
        cli_logger.abort("No storage account is found. You need to make sure storage account has been created.")
    account_name = storage_account.name

    container = get_container_for_storage_account(config, resource_group_name)
    if container is not None:
        cli_logger.print("Storage container for the workspace already exists. Skip creation.")
        return

    storage_client = construct_storage_client(config)

    cli_logger.print("Creating container for storage account: {} on Azure...", account_name)
    # Create container for storage account
    try:
        blob_container = storage_client.blob_containers.create(
            resource_group_name=resource_group_name,
            account_name=account_name,
            container_name=container_name,
            blob_container={},
        )
        cli_logger.print("Successfully created container for storage account: {}.".
                         format(account_name))
    except Exception as e:
        cli_logger.error(
            "Failed to create container for storage account. {}", str(e))
        raise e


def _create_storage_account(config, resource_group_name):
    storage_account = get_storage_account(config)
    if storage_account is not None:
        cli_logger.print("Storage account for the workspace already exists. Skip creation.")
        return

    workspace_name = config["workspace_name"]
    provider_config = config["provider"]
    location = provider_config["location"]
    subscription_id = provider_config.get("subscription_id")
    # Default is "TLS1_1", some environment requires "TLS1_2"
    # can be specified with storage options
    storage_account_options = provider_config.get("storage_account_options")
    use_working_vpc = is_use_working_vpc(config)
    resource_client = construct_resource_client(config)
    resource_group = _get_resource_group(workspace_name, resource_client, use_working_vpc)

    storage_suffix = str(uuid.uuid3(uuid.UUID(subscription_id), resource_group.id))[-12:]
    account_name = 'storage{}'.format(storage_suffix)
    storage_client = construct_storage_client(config)

    cli_logger.print("Creating workspace storage account: {} on Azure...", account_name)
    # Create storage account
    try:
        parameters = {
                "sku": {
                    "name": "Premium_LRS",
                    "tier": "Premium"
                },
                "kind": "BlockBlobStorage",
                "location": location,
                "allowBlobPublicAccess": False,
                "allowSharedKeyAccess": True,
                "isHnsEnabled": True,
                "encryption": {
                    "services": {
                        "file": {
                            "enabled": True
                        },
                        "blob": {
                             "enabled": True
                        }
                    },
                    "key_source": "Microsoft.Storage"
                },
                "tags": {
                    "Name": get_workspace_storage_account_name(workspace_name)
                }
            }

        if storage_account_options is not None:
            update_nested_dict(parameters, storage_account_options)

        poller = storage_client.storage_accounts.begin_create(
            resource_group_name=resource_group_name,
            account_name=account_name,
            parameters=parameters
        )
        # Long-running operations return a poller object; calling poller.result()
        # waits for completion.
        account_result = poller.result()
        cli_logger.print("Successfully created storage account: {}.".
                         format(account_result.name))
    except Exception as e:
        cli_logger.error(
            "Failed to create storage account. {}", str(e))
        raise e


def _create_user_assigned_identities(config, resource_group_name):
    current_step = 1
    total_steps = 2

    with cli_logger.group(
            "Creating user assigned identity for head",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_user_assigned_identity_for_head(config, resource_group_name)

    with cli_logger.group(
            "Creating user assigned identity for worker",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_user_assigned_identity_for_worker(config, resource_group_name)


def _create_user_assigned_identity_for_head(config, resource_group_name):
    user_assigned_identity_name = _get_head_user_assigned_identity_name(config)
    _create_user_assigned_identity(config, resource_group_name, user_assigned_identity_name)


def _create_user_assigned_identity_for_worker(config, resource_group_name):
    worker_user_assigned_identity_name = _get_worker_user_assigned_identity_name(config)
    _create_user_assigned_identity(config, resource_group_name, worker_user_assigned_identity_name)


def _create_user_assigned_identity(config, resource_group_name, user_assigned_identity_name):
    location = config["provider"]["location"]
    msi_client = construct_manage_server_identity_client(config)

    cli_logger.print("Creating workspace user assigned identity: {} on Azure...", user_assigned_identity_name)
    # Create identity
    try:
        msi_client.user_assigned_identities.create_or_update(
            resource_group_name,
            user_assigned_identity_name,
            parameters={
                "location": location
            }
        )
        time.sleep(20)
        cli_logger.print("Successfully created workspace user assigned identity: {}.".
                         format(user_assigned_identity_name))
    except Exception as e:
        cli_logger.error(
            "Failed to create workspace user assigned identity. {}", str(e))
        raise e


def _configure_peering_vnet_cidr_block(resource_client, network_client):
    current_resource_group_name = get_working_node_resource_group_name(resource_client)
    current_virtual_network_name = get_working_node_virtual_network_name(resource_client, network_client)
    current_vnet_peering_connections = list(network_client.virtual_network_peerings.list(current_resource_group_name, current_virtual_network_name))
    current_remote_peering_cidr_blocks = [current_vnet_peering_connection.remote_address_space.address_prefixes
                                          for current_vnet_peering_connection in current_vnet_peering_connections ]
    existing_vnet_cidr_blocks = []
    for current_remote_peering_cidr_block in current_remote_peering_cidr_blocks:
        existing_vnet_cidr_blocks += current_remote_peering_cidr_block

    current_virtual_network = get_virtual_network(current_resource_group_name, current_virtual_network_name, network_client)
    existing_vnet_cidr_blocks += current_virtual_network.address_space.address_prefixes

    for  i in range(0, 256):
        tmp_cidr_block = "10.{}.0.0/16".format(i)

        if check_cidr_conflict(tmp_cidr_block, existing_vnet_cidr_blocks):
            cli_logger.print("Successfully found cidr block for peering vnet.")
            return tmp_cidr_block

    cli_logger.abort("Failed to find non-conflicted cidr block for peering vnet.")


def _create_vnet(config, resource_client, network_client):
    workspace_name = config["workspace_name"]
    use_working_vpc = is_use_working_vpc(config)

    if use_working_vpc:
        # No need to create new virtual network
        virtual_network_name = get_working_node_virtual_network_name(resource_client, network_client)
        if virtual_network_name is None:
            cli_logger.abort("Only when the working node is an Azure instance"
                             " can use use_internal_ips=True with use_working_vpc=True.")
        else:
            cli_logger.print("Will use the current node virtual network: {}.", virtual_network_name)
    else:

        # Need to create a new virtual network
        if get_workspace_virtual_network(config, network_client) is None:
            virtual_network = create_virtual_network(config, resource_client, network_client)
            virtual_network_name = virtual_network.name
        else:
            cli_logger.abort("There is a existing virtual network with the same name: {}, "
                             "if you want to create a new workspace with the same name, "
                             "you need to execute workspace delete first!".format(workspace_name))
    return virtual_network_name


def create_virtual_network(config, resource_client, network_client):
    workspace_name = config["workspace_name"]
    virtual_network_name = get_workspace_virtual_network_name(workspace_name)
    use_working_vpc = is_use_working_vpc(config)
    resource_group_name = get_resource_group_name(config, resource_client, use_working_vpc)
    assert "location" in config["provider"], (
        "Provider config must include location field")

    # choose a random subnet, skipping most common value of 0
    random.seed(virtual_network_name)
    cidr_block = "10.{}.0.0/16".format(random.randint(1, 254))
    if is_use_peering_vpc(config):
        cidr_block = _configure_peering_vnet_cidr_block(resource_client, network_client)

    params = {
        "address_space": {
            "address_prefixes": [
                cidr_block
            ]
        },
        "location": config["provider"]["location"],
        "tags": {
            AZURE_WORKSPACE_VERSION_TAG_NAME: AZURE_WORKSPACE_VERSION_CURRENT
        }
    }
    cli_logger.print("Creating workspace virtual network: {} on Azure...", virtual_network_name)
    # create virtual network
    try:
        virtual_network = network_client.virtual_networks.begin_create_or_update(
            resource_group_name=resource_group_name,
            virtual_network_name=virtual_network_name,
            parameters=params).result()
        cli_logger.print("Successfully created workspace virtual network: {}.",
                         virtual_network_name)
        return virtual_network
    except Exception as e:
        cli_logger.error(
            "Failed to create workspace virtual network. {}", str(e))
        raise e


def get_virtual_network_peering(network_client, resource_group_name, virtual_network_name, virtual_network_peering_name):
    cli_logger.verbose("Getting the existing virtual network peering: {} ", virtual_network_peering_name)
    if virtual_network_name is None:
        cli_logger.verbose_error("Failed to get the virtual network peering: {} because virtual network {} not existed!",
                                 virtual_network_peering_name, virtual_network_name)
        return None
    try:
        virtual_network_peering = network_client.virtual_network_peerings.get(
            resource_group_name=resource_group_name,
            virtual_network_name=virtual_network_name,
            virtual_network_peering_name=virtual_network_peering_name
        )
        cli_logger.verbose("Successfully get the virtual network peering: {}.", virtual_network_peering_name)
        return virtual_network_peering
    except ResourceNotFoundError as e:
        cli_logger.verbose_error("Failed to get the virtual network peering: {}. {}", virtual_network_peering_name, str(e))
        return None


def _delete_vnet_peering_connection(network_client, resource_group_name, virtual_network_name, virtual_network_peering_name):
    if get_virtual_network_peering(network_client, resource_group_name, virtual_network_name, virtual_network_peering_name) is None:
        cli_logger.print("The virtual_network_peering \"{}\" is not found.", virtual_network_peering_name)
    else:
        try:
            network_client.virtual_network_peerings.begin_delete(
                resource_group_name=resource_group_name,
                virtual_network_name=virtual_network_name,
                virtual_network_peering_name=virtual_network_peering_name
            ).result()
            cli_logger.print("Successfully deleted virtual network peering: {} .",
                             virtual_network_peering_name)
        except Exception as e:
            cli_logger.error("Failed to delete the virtual network peering: {}. {}",
                             virtual_network_peering_name, str(e))
            raise e


def _delete_vnet_peering_connections(config, resource_client, network_client):
    workspace_name = config["workspace_name"]
    virtual_network_peering_name = get_workspace_vnet_peering_name(workspace_name)
    use_working_vpc = is_use_working_vpc(config)
    resource_group_name = get_resource_group_name(config, resource_client, use_working_vpc)
    virtual_network_name = get_virtual_network_name(config, resource_client, network_client, use_working_vpc)
    current_resource_group_name = get_working_node_resource_group_name(resource_client)
    current_virtual_network_name = get_working_node_virtual_network_name(resource_client, network_client)

    current_step = 1
    total_steps = 2

    with cli_logger.group(
            "Deleting working virtual network peering",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_vnet_peering_connection(network_client, resource_group_name, virtual_network_name,
                                        virtual_network_peering_name)

    with cli_logger.group(
            "Deleting workspace virtual network peering",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_vnet_peering_connection(network_client, current_resource_group_name, current_virtual_network_name,
                                        virtual_network_peering_name)


def get_subnet(network_client, resource_group_name, virtual_network_name, subnet_name):
    cli_logger.verbose("Getting the existing subnet: {}.", subnet_name)
    if virtual_network_name is None:
        cli_logger.verbose_error("Failed to get the subnet: {} because virtual network not existed!",
                                 subnet_name, virtual_network_name)
        return None
    try:
        subnet = network_client.subnets.get(
            resource_group_name=resource_group_name,
            virtual_network_name=virtual_network_name,
            subnet_name=subnet_name
        )
        cli_logger.verbose("Successfully get the subnet: {}.", subnet_name)
        return subnet
    except ResourceNotFoundError as e:
        cli_logger.verbose_error("Failed to get the subnet: {}. {}", subnet_name, str(e))
        return None


def _delete_subnet(config, network_client, resource_group_name, virtual_network_name, is_private=True):
    if is_private:
        subnet_attribute = "private"
    else:
        subnet_attribute = "public"

    workspace_name = config["workspace_name"]
    subnet_name = get_workspace_subnet_name(workspace_name, isPrivate=is_private)

    if get_subnet(network_client, resource_group_name, virtual_network_name, subnet_name) is None:
        cli_logger.print("The {} subnet \"{}\" is not found for workspace.",
                         subnet_attribute, subnet_name)
        return

    """ Delete custom subnet """
    cli_logger.print("Deleting {} subnet: {}...", subnet_attribute, subnet_name)
    try:
        network_client.subnets.begin_delete(
            resource_group_name=resource_group_name,
            virtual_network_name=virtual_network_name,
            subnet_name=subnet_name
        ).result()
        cli_logger.print("Successfully deleted {} subnet: {}.",
                         subnet_attribute, subnet_name)
    except Exception as e:
        cli_logger.error("Failed to delete the {} subnet: {}. {}",
                         subnet_attribute, subnet_name, str(e))
        raise e


def _configure_azure_subnet_cidr(network_client, resource_group_name, virtual_network_name):
    virtual_network = network_client.virtual_networks.get(
        resource_group_name=resource_group_name, virtual_network_name=virtual_network_name)
    ip = virtual_network.address_space.address_prefixes[0].split("/")[0].split(".")
    subnets = virtual_network.subnets
    cidr_block = None

    if len(subnets) == 0:
        existed_cidr_blocks = []
    else:
        existed_cidr_blocks = [subnet.address_prefix for subnet in subnets]

    # choose a random subnet, skipping most common value of 0
    random.seed(virtual_network_name)
    while cidr_block is None:
        tmp_cidr_block = ip[0] + "." + ip[1] + "." + str(random.randint(1, 254)) + ".0/24"
        if check_cidr_conflict(tmp_cidr_block, existed_cidr_blocks):
            cidr_block = tmp_cidr_block

    return cidr_block


def _create_vnet_peering_connection(network_client, subscription_id, current_resource_group_name,
                                    current_virtual_network_name , remote_resource_group_name, remote_virtual_network_name, virtual_network_peering_name):
    # Create virtual network peering
    cli_logger.print("Creating virtual network peering: {}... ", virtual_network_peering_name)
    try:
        network_client.virtual_network_peerings.begin_create_or_update(
            current_resource_group_name,
            current_virtual_network_name,
            virtual_network_peering_name,
            {
                "allow_virtual_network_access": True,
                "allow_forwarded_traffic": True,
                "allow_gateway_transit": False,
                "use_remote_gateways": False,
                "remote_virtual_network": {
                    "id": "/subscriptions/{}/resourceGroups/{}/providers/Microsoft.Network/virtualNetworks/{}"
                        .format(subscription_id, remote_resource_group_name, remote_virtual_network_name)
                }
            }
        ).result()
        cli_logger.print("Successfully created virtual network peering {} for virtual network {}.",
                         virtual_network_peering_name, current_virtual_network_name)
    except Exception as e:
        cli_logger.error("Failed to create virtual network peering. {}", str(e))
        raise e


def _create_vnet_peering_connections(config, resource_client, network_client, resource_group_name, virtual_network_name):
    subscription_id = config["provider"].get("subscription_id")
    workspace_name = config["workspace_name"]
    virtual_network_peering_name = get_workspace_vnet_peering_name(workspace_name)
    current_resource_group_name = get_working_node_resource_group_name(resource_client)
    current_virtual_network_name = get_working_node_virtual_network_name(resource_client, network_client)

    current_step = 1
    total_steps = 2

    with cli_logger.group(
            "Creating working virtual network peering",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_vnet_peering_connection(network_client, subscription_id, current_resource_group_name,
                                        current_virtual_network_name, resource_group_name,
                                        virtual_network_name, virtual_network_peering_name)

    with cli_logger.group(
            "Creating workspace virtual network peering",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_vnet_peering_connection(network_client, subscription_id, resource_group_name,
                                        virtual_network_name, current_resource_group_name,
                                        current_virtual_network_name, virtual_network_peering_name)


def _create_and_configure_subnets(config, network_client, resource_group_name, virtual_network_name, is_private=True):
    subscription_id = config["provider"].get("subscription_id")
    workspace_name = config["workspace_name"]
    subnet_attribute = "private" if is_private else "public"
    subnet_name = get_workspace_subnet_name(workspace_name, isPrivate=is_private)

    cidr_block = _configure_azure_subnet_cidr(network_client, resource_group_name, virtual_network_name)
    nat_gateway_name = get_workspace_nat_name(workspace_name)
    network_security_group_name = get_workspace_network_security_group_name(workspace_name)
    if is_private:
        subnet_parameters = {
            "address_prefix": cidr_block,
            "nat_gateway": {
                "id": "/subscriptions/{}/resourceGroups/{}/providers/Microsoft.Network/natGateways/{}"
                    .format(subscription_id, resource_group_name, nat_gateway_name)
            },
            "network_security_group": {
                "id": "/subscriptions/{}/resourceGroups/{}/providers/Microsoft.Network/networkSecurityGroups/{}"
                    .format(subscription_id, resource_group_name, network_security_group_name)
            }
        }
    else:
        subnet_parameters = {
            "address_prefix": cidr_block,
            "network_security_group": {
                "id": "/subscriptions/{}/resourceGroups/{}/providers/Microsoft.Network/networkSecurityGroups/{}"
                    .format(subscription_id, resource_group_name, network_security_group_name)
            }
        }

    # Create subnet
    cli_logger.print("Creating subnet for the virtual network: {} with CIDR: {}...".
                     format(virtual_network_name, cidr_block))
    try:
        network_client.subnets.begin_create_or_update(
            resource_group_name=resource_group_name,
            virtual_network_name=virtual_network_name,
            subnet_name=subnet_name,
            subnet_parameters=subnet_parameters
        ).result()
        cli_logger.print("Successfully created {} subnet: {}.".format(subnet_attribute, subnet_name))
    except Exception as e:
        cli_logger.error("Failed to create subnet. {}", str(e))
        raise e


def _create_nat(config, network_client, resource_group_name, public_ip_address_name):
    subscription_id = config["provider"].get("subscription_id")
    workspace_name = config["workspace_name"]
    nat_gateway_name = get_workspace_nat_name(workspace_name)

    cli_logger.print("Creating NAT gateway: {}... ".format(nat_gateway_name))
    try:
        network_client.nat_gateways.begin_create_or_update(
            resource_group_name=resource_group_name,
            nat_gateway_name=nat_gateway_name,
            parameters={
                "location": config["provider"]["location"],
                "sku": {
                    "name": "Standard"
                },
                "public_ip_addresses": [
                    {
                        "id": "/subscriptions/{}/resourceGroups/{}/providers/Microsoft.Network/publicIPAddresses/{}"
                            .format(subscription_id, resource_group_name, public_ip_address_name)
                    }
                ],
            }
        ).result()
        cli_logger.print("Successfully created NAT gateway: {}.".
                         format(nat_gateway_name))
    except Exception as e:
        cli_logger.error("Failed to create NAT gateway. {}", str(e))
        raise e


def _create_public_ip_address(config, network_client, resource_group_name):
    workspace_name = config["workspace_name"]
    public_ip_address_name = get_workspace_public_ip_address_name(workspace_name)
    location = config["provider"]["location"]

    cli_logger.print("Creating public IP address: {}... ".format(public_ip_address_name))
    try:
        network_client.public_ip_addresses.begin_create_or_update(
            resource_group_name,
            public_ip_address_name,
            {
                'location': location,
                'public_ip_allocation_method': 'Static',
                'idle_timeout_in_minutes': 4,
                "sku": {
                    "name": "Standard"
                }
            }
        ).result()
        cli_logger.print("Successfully created public IP address: {}.".
                         format(public_ip_address_name))
    except Exception as e:
        cli_logger.error("Failed to create public IP address. {}", str(e))
        raise e

    return public_ip_address_name


def _create_or_update_network_security_group(config, network_client, resource_group_name):
    workspace_name = config["workspace_name"]
    location = config["provider"]["location"]
    security_rules = config["provider"].get("securityRules", [])
    network_security_group_name = get_workspace_network_security_group_name(workspace_name)

    for i in range(0, len(security_rules)):
        security_rules[i]["name"] = get_workspace_security_rule_name(workspace_name, i)

    cli_logger.print("Creating or updating network security group: {}... ".format(network_security_group_name))
    try:
        network_client.network_security_groups._create_or_update_initial(
            resource_group_name=resource_group_name,
            network_security_group_name=network_security_group_name,
            parameters={
                "location": location,
                "securityRules": security_rules
            }
        )
        cli_logger.print("Successfully created or updated network security group: {}.".
                         format(network_security_group_name))
    except Exception as e:
        cli_logger.error("Failed to create or updated network security group. {}", str(e))
        raise e

    return network_security_group_name


def _create_network_resources(config, resource_group_name, current_step, total_steps):
    network_client = construct_network_client(config)
    resource_client = construct_resource_client(config)

    # create virtual network
    with cli_logger.group(
            "Creating virtual network",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        virtual_network_name = _create_vnet(config, resource_client, network_client)

    # create network security group
    with cli_logger.group(
            "Creating network security group",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_or_update_network_security_group(config, network_client, resource_group_name)

    # create public subnet
    with cli_logger.group(
            "Creating public subnet",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_and_configure_subnets(
            config, network_client, resource_group_name, virtual_network_name, is_private=False)

    # create public IP address
    with cli_logger.group(
            "Creating public IP address for NAT gateway",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        public_ip_address_name = _create_public_ip_address(config, network_client, resource_group_name,)

    # create NAT gateway
    with cli_logger.group(
            "Creating NAT gateway",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_nat(config, network_client, resource_group_name, public_ip_address_name)

    # create private subnet
    with cli_logger.group(
            "Creating private subnet",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_and_configure_subnets(
            config, network_client, resource_group_name, virtual_network_name, is_private=True)

    if is_use_peering_vpc(config):
        with cli_logger.group(
                "Creating virtual network peerings",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _create_vnet_peering_connections(config, resource_client, network_client, resource_group_name, virtual_network_name)

    return current_step


def bootstrap_azure_from_workspace(config):
    if not check_azure_workspace_integrity(config):
        workspace_name = config["workspace_name"]
        cli_logger.abort("Azure workspace {} doesn't exist or is in wrong state!", workspace_name)

    config = _configure_key_pair(config)
    config = _configure_workspace_resource(config)
    config = _configure_prefer_spot_node(config)
    return config


def bootstrap_azure_workspace(config):
    # create a copy of the input config to modify
    config = copy.deepcopy(config)
    _configure_allowed_ssh_sources(config)
    return config


def _configure_allowed_ssh_sources(config):
    provider_config = config["provider"]
    if "allowed_ssh_sources" not in provider_config:
        return

    allowed_ssh_sources = provider_config["allowed_ssh_sources"]
    if len(allowed_ssh_sources) == 0:
        return

    if "securityRules" not in provider_config:
        provider_config["securityRules"] = []
    security_rules = provider_config["securityRules"]

    security_rule = {
        "priority": 1000,
        "protocol": "Tcp",
        "access": "Allow",
        "direction": "Inbound",
        "source_address_prefixes": [allowed_ssh_source for allowed_ssh_source in allowed_ssh_sources],
        "source_port_range": "*",
        "destination_address_prefix": "*",
        "destination_port_range": 22
    }
    security_rules.append(security_rule)


def _configure_workspace_resource(config):
    config = _configure_resource_group_from_workspace(config)
    config = _configure_virtual_network_from_workspace(config)
    config = _configure_subnet_from_workspace(config)
    config = _configure_network_security_group_from_workspace(config)
    config = _configure_user_assigned_identity_from_workspace(config)
    config = _configure_cloud_storage_from_workspace(config)
    return config


def _configure_cloud_storage_from_workspace(config):
    use_managed_cloud_storage = is_use_managed_cloud_storage(config)
    use_working_vpc = is_use_working_vpc(config)
    resource_client = construct_resource_client(config)
    resource_group_name = get_resource_group_name(config, resource_client, use_working_vpc)
    if use_managed_cloud_storage:
        azure_cloud_storage = get_workspace_azure_storage(config, config["workspace_name"])
        if azure_cloud_storage is None:
            cli_logger.abort("No managed azure storage container was found. If you want to use managed azure storage, "
                             "you should set managed_cloud_storage equal to True when you creating workspace.")

        cloud_storage = get_azure_cloud_storage_config_for_update(config["provider"])
        cloud_storage["azure.storage.type"] = azure_cloud_storage["azure.storage.type"]
        cloud_storage["azure.storage.account"] = azure_cloud_storage["azure.storage.account"]
        cloud_storage["azure.container"] = azure_cloud_storage["azure.container"]

    user_assigned_identity = get_head_user_assigned_identity(config, resource_group_name)
    worker_user_assigned_identity = get_worker_user_assigned_identity(config, resource_group_name)
    for key, node_type in config["available_node_types"].items():
        node_config = node_type["node_config"]
        if key == config["head_node_type"]:
            node_config["azure.user.assigned.identity.client.id"] = user_assigned_identity.client_id
            node_config["azure.user.assigned.identity.tenant.id"] = user_assigned_identity.tenant_id
        else:
            node_config["azure.user.assigned.identity.client.id"] = worker_user_assigned_identity.client_id
            node_config["azure.user.assigned.identity.tenant.id"] = worker_user_assigned_identity.tenant_id

    return config


def get_workspace_azure_storage(config, workspace_name):
    use_working_vpc = is_use_working_vpc(config)
    resource_client = construct_resource_client(config)
    resource_group_name = get_resource_group_name(config, resource_client, use_working_vpc)
    storage_account = get_storage_account(config)
    container = get_container_for_storage_account(config, resource_group_name)
    if container is None:
        return None

    azure_cloud_storage = {"azure.storage.type": "datalake",
                           "azure.storage.account": storage_account.name,
                           "azure.container": container.name}
    return azure_cloud_storage


def _get_head_user_assigned_identity_name(config):
    workspace_name = config["workspace_name"]
    user_assigned_identity_name = get_workspace_head_user_assigned_identity_name(workspace_name)
    return user_assigned_identity_name


def _get_worker_user_assigned_identity_name(config):
    workspace_name = config["workspace_name"]
    user_assigned_identity_name = get_workspace_worker_user_assigned_identity_name(workspace_name)
    return user_assigned_identity_name


def _configure_user_assigned_identity_from_workspace(config):
    user_assigned_identity_name = _get_head_user_assigned_identity_name(config)
    worker_user_assigned_identity_name = _get_worker_user_assigned_identity_name(config)

    config["provider"]["userAssignedIdentity"] = user_assigned_identity_name
    for key, node_type in config["available_node_types"].items():
        node_config = node_type["node_config"]
        if key == config["head_node_type"]:
            node_config["azure_arm_parameters"]["userAssignedIdentity"] = user_assigned_identity_name
        else:
            node_config["azure_arm_parameters"]["userAssignedIdentity"] = worker_user_assigned_identity_name

    return config


def _configure_subnet_from_workspace(config):
    workspace_name = config["workspace_name"]
    use_internal_ips = is_use_internal_ip(config)

    public_subnet = get_workspace_subnet_name(workspace_name, isPrivate=False)
    private_subnet = get_workspace_subnet_name(workspace_name, isPrivate=True)

    for key, node_type in config["available_node_types"].items():
        node_config = node_type["node_config"]
        if key == config["head_node_type"]:
            if use_internal_ips:
                node_config["azure_arm_parameters"]["subnetName"] = private_subnet
                node_config["azure_arm_parameters"]["provisionPublicIp"] = False
            else:
                node_config["azure_arm_parameters"]["subnetName"] = public_subnet
                node_config["azure_arm_parameters"]["provisionPublicIp"] = True
        else:
            node_config["azure_arm_parameters"]["subnetName"] = private_subnet
            node_config["azure_arm_parameters"]["provisionPublicIp"] = False

    return config


def _configure_network_security_group_from_workspace(config):
    workspace_name = config["workspace_name"]
    network_security_group_name = get_workspace_network_security_group_name(workspace_name)

    for node_type_key in config["available_node_types"].keys():
        node_config = config["available_node_types"][node_type_key][
            "node_config"]
        node_config["azure_arm_parameters"]["networkSecurityGroupName"] = network_security_group_name

    return config


def _configure_virtual_network_from_workspace(config):
    use_working_vpc = is_use_working_vpc(config)
    resource_client = construct_resource_client(config)
    network_client = construct_network_client(config)

    virtual_network_name = get_virtual_network_name(config, resource_client, network_client, use_working_vpc)

    for node_type_key in config["available_node_types"].keys():
        node_config = config["available_node_types"][node_type_key][
            "node_config"]
        node_config["azure_arm_parameters"]["virtualNetworkName"] = virtual_network_name

    return config


def _configure_resource_group_from_workspace(config):
    use_working_vpc = is_use_working_vpc(config)
    resource_client = construct_resource_client(config)
    resource_group_name = get_resource_group_name(config, resource_client, use_working_vpc)
    config["provider"]["resource_group"] = resource_group_name
    return config


def _configure_spot_for_node_type(node_type_config,
                                  prefer_spot_node):
    # azure_arm_parameters
    #   priority: Spot
    node_config = node_type_config["node_config"]
    azure_arm_parameters = node_config["azure_arm_parameters"]
    if prefer_spot_node:
        # Add spot instruction
        azure_arm_parameters["priority"] = "Spot"
    else:
        # Remove spot instruction
        azure_arm_parameters.pop("priority", None)


def _configure_prefer_spot_node(config):
    prefer_spot_node = config["provider"].get("prefer_spot_node")

    # if no such key, we consider user don't want to override
    if prefer_spot_node is None:
        return config

    # User override, set or remove spot settings for worker node types
    node_types = config["available_node_types"]
    for node_type_name in node_types:
        if node_type_name == config["head_node_type"]:
            continue

        # worker node type
        node_type_data = node_types[node_type_name]
        _configure_spot_for_node_type(
            node_type_data, prefer_spot_node)

    return config


def bootstrap_azure(config):
    workspace_name = config.get("workspace_name", "")
    if workspace_name == "":
        raise RuntimeError("Workspace name is not specified in cluster configuration.")

    config = bootstrap_azure_from_workspace(config)
    return config


def bootstrap_azure_for_api(config):
    workspace_name = config.get("workspace_name", "")
    if workspace_name == "":
        raise ValueError(f"Workspace name is not specified.")

    return _configure_resource_group_from_workspace(config)


def _configure_key_pair(config):
    ssh_user = config["auth"]["ssh_user"]
    public_key = None
    # search if the keys exist
    for key_type in ["ssh_private_key", "ssh_public_key"]:
        try:
            key_path = Path(config["auth"][key_type]).expanduser()
        except KeyError:
            raise Exception("Config must define {}".format(key_type))
        except TypeError:
            raise Exception("Invalid config value for {}".format(key_type))

        assert key_path.is_file(), (
            "Could not find ssh key: {}".format(key_path))

        if key_type == "ssh_public_key":
            with open(key_path, "r") as f:
                public_key = f.read()

    for node_type in config["available_node_types"].values():
        azure_arm_parameters = node_type["node_config"].setdefault(
            "azure_arm_parameters", {})
        azure_arm_parameters["adminUsername"] = ssh_user
        azure_arm_parameters["publicKey"] = public_key

    return config


def _extract_metadata_for_node(vm, resource_group, compute_client, network_client):
    # get tags
    metadata = {"name": vm.name, "tags": vm.tags, "status": "", "vm_size": ""}

    # get status
    instance = compute_client.virtual_machines.instance_view(
        resource_group_name=resource_group, vm_name=vm.name).as_dict()
    for status in instance["statuses"]:
        status_list = status["code"].split("/")
        code = status_list[0]
        state = status_list[1]
        # skip provisioning status
        if code == "PowerState":
            metadata["status"] = state
            break

    # get ip data
    nic_id = vm.network_profile.network_interfaces[0].id
    metadata["nic_name"] = nic_id.split("/")[-1]
    nic = network_client.network_interfaces.get(
        resource_group_name=resource_group,
        network_interface_name=metadata["nic_name"])
    ip_config = nic.ip_configurations[0]

    public_ip_address = ip_config.public_ip_address
    if public_ip_address is not None:
        public_ip_id = public_ip_address.id
        metadata["public_ip_name"] = public_ip_id.split("/")[-1]
        public_ip = network_client.public_ip_addresses.get(
            resource_group_name=resource_group,
            public_ip_address_name=metadata["public_ip_name"])
        metadata["external_ip"] = public_ip.ip_address
    else:
        metadata["external_ip"] = None

    metadata["internal_ip"] = ip_config.private_ip_address

    # get vmSize
    metadata["vm_size"] = vm.hardware_profile.vm_size

    return metadata


def get_workspace_head_nodes(provider_config, workspace_name):
    compute_client = _construct_compute_client(provider_config)
    resource_client = _construct_resource_client(provider_config)
    use_working_vpc = _is_use_working_vpc(provider_config)
    resource_group_name = _get_resource_group_name(
        workspace_name, resource_client, use_working_vpc)
    return _get_workspace_head_nodes(
        workspace_name=workspace_name,
        resource_group_name=resource_group_name,
        compute_client=compute_client
    )


def _get_workspace_head_nodes(workspace_name,
                              resource_group_name,
                              compute_client):
    if resource_group_name is None:
        raise RuntimeError(
            "The workspace {} doesn't exist or is in the wrong state.".format(
                workspace_name
            ))
    all_heads = [node for node in list(
        compute_client.virtual_machines.list(resource_group_name=resource_group_name))
            if node.tags is not None and node.tags.get(CLOUDTIK_TAG_NODE_KIND, "") == NODE_KIND_HEAD]

    workspace_heads = []
    for head in all_heads:
        instance = compute_client.virtual_machines.instance_view(
            resource_group_name=resource_group_name, vm_name=head.name).as_dict()
        in_running = True
        for status in instance["statuses"]:
            status_list = status["code"].split("/")
            code = status_list[0]
            state = status_list[1]
            if code == "PowerState" and state != "running":
                in_running = False
        if in_running:
            workspace_heads.append(head)

    return workspace_heads


def verify_azure_blob_storage(provider_config: Dict[str, Any]):
    azure_cloud_storage = get_azure_cloud_storage_config(provider_config)
    azure_storage_account = azure_cloud_storage["azure.storage.account"]
    azure_account_key = azure_cloud_storage["azure.account.key"]
    azure_container = azure_cloud_storage["azure.container"]

    # Create the connection string
    connection_string = "DefaultEndpointsProtocol=https;"
    connection_string += f"AccountName={azure_storage_account};"
    connection_string += f"AccountKey={azure_account_key};"
    connection_string += "EndpointSuffix=core.windows.net"

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Instantiate a ContainerClient
    container_client = blob_service_client.get_container_client(azure_container)

    exists = container_client.exists()
    if not exists:
        raise RuntimeError(f"Container {azure_container} doesn't exist in Azure Blob Storage.")


def verify_azure_datalake_storage(provider_config: Dict[str, Any]):
    azure_cloud_storage = get_azure_cloud_storage_config(provider_config)
    azure_storage_account = azure_cloud_storage["azure.storage.account"]
    azure_container = azure_cloud_storage["azure.container"]
    azure_account_key = azure_cloud_storage.get("azure.account.key")
    if azure_account_key is None:
        # Check whether its existence by storage management client
        resource_group_name = provider_config["resource_group"]
        container = _get_container(
            provider_config,
            resource_group_name=resource_group_name,
            storage_account_name=azure_storage_account,
            container_name=azure_container)
    else:
        service_client = DataLakeServiceClient(account_url="{}://{}.dfs.core.windows.net".format(
            "https", azure_storage_account), credential=azure_account_key)

        file_system_client = service_client.get_file_system_client(file_system=azure_container)

        exists = file_system_client.exists()
        if not exists:
            raise RuntimeError(f"Container {azure_container} doesn't exist in Azure Data Lake Storage Gen 2.")


def verify_azure_cloud_storage(provider_config: Dict[str, Any]):
    # TO IMPROVE: if we use managed cloud storage or storage with role access
    # we verify only the existence of container
    if _is_use_managed_cloud_storage(provider_config):
        return

    azure_cloud_storage = get_azure_cloud_storage_config(provider_config)
    if azure_cloud_storage is None:
        return

    try:
        storage_type = azure_cloud_storage["azure.storage.type"]
        if storage_type == "blob":
            verify_azure_blob_storage(provider_config)
        else:
            verify_azure_datalake_storage(provider_config)
    except Exception as e:
        raise StorageTestingError("Error happens when verifying Azure cloud storage configurations. "
                                  "If you want to go without passing the verification, "
                                  "set 'verify_cloud_storage' to False under provider config. "
                                  "Error: {}.".format(str(e))) from None


def get_cluster_name_from_head(head_node) -> Optional[str]:
    for key, value in head_node.tags.items():
        if key == CLOUDTIK_TAG_CLUSTER_NAME:
            return value
    return None


def list_azure_clusters(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    compute_client = construct_compute_client(config)
    resource_client = construct_resource_client(config)
    network_client = construct_network_client(config)
    use_working_vpc = is_use_working_vpc(config)
    resource_group_name = get_resource_group_name(config, resource_client, use_working_vpc)

    head_nodes = _get_workspace_head_nodes(
        workspace_name=config.get("workspace_name"),
        resource_group_name=resource_group_name,
        compute_client=compute_client
    )

    clusters = {}
    for head_node in head_nodes:
        cluster_name = get_cluster_name_from_head(head_node)
        if cluster_name:
            head_node_meta = _extract_metadata_for_node(
                head_node,
                resource_group=resource_group_name,
                compute_client=compute_client,
                network_client=network_client)
            clusters[cluster_name] = _get_node_info(head_node_meta)
    return clusters


def with_azure_environment_variables(provider_config, node_type_config: Dict[str, Any], node_id: str):
    config_dict = {}
    export_azure_cloud_storage_config(provider_config, config_dict)

    if node_type_config is not None:
        node_config = node_type_config.get("node_config")
        user_assigned_identity_client_id = node_config.get(
            "azure.user.assigned.identity.client.id")
        if user_assigned_identity_client_id:
            config_dict["AZURE_MANAGED_IDENTITY_CLIENT_ID"] = user_assigned_identity_client_id

        user_assigned_identity_tenant_id = node_config.get(
            "azure.user.assigned.identity.tenant.id")
        if user_assigned_identity_tenant_id:
            config_dict["AZURE_MANAGED_IDENTITY_TENANT_ID"] = user_assigned_identity_tenant_id

    return config_dict
