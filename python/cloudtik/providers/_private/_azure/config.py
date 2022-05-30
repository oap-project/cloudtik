import copy
import json
import logging
import time
import uuid
import subprocess
from pathlib import Path
import random

from typing import Any, Callable, Dict, Optional

from cloudtik.core.tags import CLOUDTIK_TAG_NODE_KIND, NODE_KIND_HEAD, CLOUDTIK_TAG_CLUSTER_NAME
from cloudtik.core._private.cli_logger import cli_logger, cf
from cloudtik.core._private.utils import check_cidr_conflict, is_use_internal_ip, _is_use_internal_ip, \
    is_managed_cloud_storage, is_use_managed_cloud_storage, _is_use_managed_cloud_storage
from cloudtik.providers._private._azure.azure_identity_credential_adapter import AzureIdentityCredentialAdapter

from azure.common.credentials import get_cli_profile
from azure.identity import AzureCliCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.storage import StorageManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.msi import ManagedServiceIdentityClient
from azure.mgmt.authorization import AuthorizationManagementClient
from azure.mgmt.resource.resources.models import DeploymentMode

from azure.storage.blob import BlobServiceClient
from azure.storage.filedatalake import DataLakeServiceClient

from cloudtik.providers._private._azure.utils import _get_node_info
from cloudtik.providers._private.utils import StorageTestingError

AZURE_RESOURCE_NAME_PREFIX = "cloudtik"
AZURE_MSI_NAME = AZURE_RESOURCE_NAME_PREFIX + "-msi-user-identity"
AZURE_NSG_NAME = AZURE_RESOURCE_NAME_PREFIX + "-nsg"
AZURE_SUBNET_NAME = AZURE_RESOURCE_NAME_PREFIX + "-subnet"
AZURE_VNET_NAME = AZURE_RESOURCE_NAME_PREFIX + "-vnet"

NUM_AZURE_WORKSPACE_CREATION_STEPS = 9
NUM_AZURE_WORKSPACE_DELETION_STEPS = 7

logger = logging.getLogger(__name__)


def get_azure_sdk_function(client: Any, function_name: str) -> Callable:
    """Retrieve a callable function from Azure SDK client object.

       Newer versions of the various client SDKs renamed function names to
       have a begin_ prefix. This function supports both the old and new
       versions of the SDK by first trying the old name and falling back to
       the prefixed new name.
    """
    func = getattr(client, function_name,
                   getattr(client, f"begin_{function_name}"))
    if func is None:
        raise AttributeError(
            "'{obj}' object has no {func} or begin_{func} attribute".format(
                obj={client.__name__}, func=function_name))
    return func


def check_azure_workspace_resource(config):
    use_internal_ips = is_use_internal_ip(config)
    workspace_name = config["workspace_name"]
    managed_cloud_storage = is_managed_cloud_storage(config)
    network_client = construct_network_client(config)
    resource_client = construct_resource_client(config)

    """
         Do the work - order of operation
         1). Check resource group
         2.) Check vpc 
         3.) Check network security group
         4.) Check public IP address
         5.) Check NAT gateway
         6.) Check private subnet 
         7.) Check public subnet
         8.) Check role assignments
         9.) Check user assigned identities
         10.) Check cloud storage if need
    """
    resource_group_name = get_resource_group_name(config, resource_client, use_internal_ips)
    if resource_group_name is None:
        return False
    virtual_network_name = get_virtual_network_name(config, resource_client, network_client, use_internal_ips)

    if virtual_network_name is None:
        return False

    if get_network_security_group(config, network_client, resource_group_name) is None:
        return False

    if get_public_ip_address(config, network_client, resource_group_name) is None:
        return False

    if get_nat_gateway(config, network_client, resource_group_name) is None:
        return False

    private_subnet_name = "cloudtik-{}-private-subnet".format(workspace_name)
    if get_subnet(network_client, resource_group_name, virtual_network_name, private_subnet_name) is None:
        return False

    public_subnet_name = "cloudtik-{}-public-subnet".format(workspace_name)
    if get_subnet(network_client, resource_group_name, virtual_network_name, public_subnet_name) is None:
        return False

    if get_head_user_assigned_identity(config, resource_group_name) is None:
        return False

    if get_worker_user_assigned_identity(config, resource_group_name) is None:
        return False

    if get_role_assignment_for_contributor(config, resource_group_name) is None:
        return False

    if get_role_assignment_for_storage_blob_data_owner(config, resource_group_name) is None:
        return False

    if managed_cloud_storage:
        if get_container_for_storage_account(config, resource_group_name) is None:
            return False

    return True


def get_resource_group_name(config, resource_client, use_internal_ips):
    return _get_resource_group_name(
        config.get("workspace_name"), resource_client, use_internal_ips)


def get_virtual_network_name(config, resource_client, network_client, use_internal_ips):
    if use_internal_ips:
        virtual_network_name = get_working_node_virtual_network_name(resource_client, network_client)
    else:
        virtual_network_name = get_workspace_virtual_network_name(config, network_client)

    return virtual_network_name


def update_azure_workspace_firewalls(config):
    resource_client = construct_resource_client(config)
    network_client = construct_network_client(config)
    workspace_name = config["workspace_name"]
    use_internal_ips = is_use_internal_ip(config)
    resource_group_name = get_resource_group_name(config, resource_client, use_internal_ips)

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
            "Failed to update the firewalls of workspace {}. {}".format(workspace_name, str(e)))
        raise e

    cli_logger.print(
        "Successfully updated the firewalls of workspace: {}.",
        cf.bold(workspace_name))
    return None


def delete_workspace_azure(config, delete_managed_storage: bool = False):
    resource_client = construct_resource_client(config)
    workspace_name = config["workspace_name"]
    use_internal_ips = is_use_internal_ip(config)
    managed_cloud_storage = is_managed_cloud_storage(config)
    resource_group_name = get_resource_group_name(config, resource_client, use_internal_ips)

    if resource_group_name is None:
        cli_logger.print("Workspace: {} doesn't exist!".format(config["workspace_name"]))
        return

    current_step = 1
    total_steps = NUM_AZURE_WORKSPACE_DELETION_STEPS
    if not use_internal_ips:
        total_steps += 2
    if managed_cloud_storage and delete_managed_storage:
        total_steps += 1

    try:
        # delete network resources
        with cli_logger.group("Deleting workspace: {}", workspace_name):
            current_step = _delete_network_resources(config, resource_client, resource_group_name, current_step, total_steps)

            # delete role_assignments
            with cli_logger.group(
                    "Deleting role assignments for managed Identity ",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _delete_role_assignments(config, resource_group_name)

            # delete user_assigned_identities
            with cli_logger.group(
                    "Deleting user assigned identities",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _delete_user_assigned_identities(config, resource_group_name)

            if managed_cloud_storage and delete_managed_storage:
                with cli_logger.group(
                        "Deleting Azure storage account",
                        _numbered=("[]", current_step, total_steps)):
                    current_step += 1
                    _delete_workspace_cloud_storage(config, resource_group_name)
                    
            # delete resource group
            if not use_internal_ips:
                with cli_logger.group(
                        "Deleting resource group",
                        _numbered=("[]", current_step, total_steps)):
                    current_step += 1
                    _delete_resource_group(config, resource_client)
    except Exception as e:
        cli_logger.error(
            "Failed to delete workspace {}. {}".format(workspace_name, str(e)))
        raise e

    cli_logger.print(
        "Successfully deleted workspace: {}.",
        cf.bold(workspace_name))
    return None


def _delete_network_resources(config, resource_client, resource_group_name, current_step, total_steps):
    use_internal_ips = is_use_internal_ip(config)
    network_client = construct_network_client(config)
    virtual_network_name = get_virtual_network_name(config, resource_client, network_client, use_internal_ips)

    """
         Do the work - order of operation
         1.) Delete public subnet
         2.) Delete private subnet 
         3.) Delete NAT gateway
         4.) Delete public IP address
         5.) Delete network security group
         6.) Delete vpc
    """

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
    if not use_internal_ips:
        with cli_logger.group(
                "Deleting VPC",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _delete_vnet(config, resource_client, network_client)

    return current_step


def get_container_for_storage_account(config, resource_group_name):
    workspace_name = config["workspace_name"]
    container_name = "cloudtik-{}-storage-container".format(workspace_name)
    storage_client = construct_storage_client(config)
    storage_account = get_storage_account(config)
    if storage_account is None:
        return None
    containers = list(storage_client.blob_containers.list(
        resource_group_name=resource_group_name, account_name=storage_account.name))
    workspace_containers = [container for container in containers
                                  if container.name == container_name]
    return None if len(workspace_containers) == 0 else workspace_containers[0]


def get_storage_account(config):
    workspace_name = config["workspace_name"]
    storage_client = construct_storage_client(config)
    storage_account_name = "cloudtik-{}-storage-account".format(workspace_name)
    storage_accounts = list(storage_client.storage_accounts.list())

    workspace_storage_accounts = [storage_account for storage_account in storage_accounts
                                 for key, value in storage_account.tags.items()
                                 if key == "Name" and value == storage_account_name]
    return None if len(workspace_storage_accounts) == 0 else workspace_storage_accounts[0]


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
            "Failed to delete the storage account: {}. {}".format(storage_account.name, str(e)))
        raise e


def get_role_assignment_for_storage_blob_data_owner(config, resource_group_name):
    workspace_name = config["workspace_name"]
    authorization_client = construct_authorization_client(config)
    subscription_id = config["provider"].get("subscription_id")
    scope = "subscriptions/{subscriptionId}/resourcegroups/{resourceGroupName}".format(
        subscriptionId=subscription_id,
        resourceGroupName=resource_group_name
    )

    role_assignment_name = str(uuid.uuid3(uuid.UUID(subscription_id), workspace_name + "storage_blob_data_owner"))
    cli_logger.verbose("Getting the existing role assignment for Storage Blob Data Owner: {}.",
                       role_assignment_name)

    try:
        role_assignment = authorization_client.role_assignments.get(
            scope=scope,
            role_assignment_name=role_assignment_name,
        )
        cli_logger.verbose("Successfully get the role assignment for Storage Blob Data Owner: {}.".
                           format(role_assignment_name))
        return role_assignment_name
    except Exception as e:
        cli_logger.error(
            "Failed to get the role assignment. {}", str(e))
        return None


def get_role_assignment_for_contributor(config, resource_group_name):
    workspace_name = config["workspace_name"]
    authorization_client = construct_authorization_client(config)
    subscription_id = config["provider"].get("subscription_id")
    scope = "subscriptions/{subscriptionId}/resourcegroups/{resourceGroupName}".format(
        subscriptionId=subscription_id,
        resourceGroupName=resource_group_name
    )
    role_assignment_name = str(uuid.uuid3(uuid.UUID(subscription_id), workspace_name + "contributor"))
    cli_logger.verbose("Getting the existing role assignment for Contributor: {}.", role_assignment_name)

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


def _delete_role_assignment_for_storage_blob_data_owner(config, resource_group_name):
    role_assignment_name = get_role_assignment_for_storage_blob_data_owner(config, resource_group_name)
    authorization_client = construct_authorization_client(config)
    subscription_id = config["provider"].get("subscription_id")
    scope = "subscriptions/{subscriptionId}/resourcegroups/{resourceGroupName}".format(
        subscriptionId=subscription_id,
        resourceGroupName=resource_group_name
    )
    if role_assignment_name is None:
        cli_logger.print("The role assignment doesn't exist.")
        return

    """ Delete the role_assignment """
    cli_logger.print("Deleting the role assignment for Storage Blob Data Owner: {}...".format(
        role_assignment_name))
    try:
        authorization_client.role_assignments.delete(
            scope=scope,
            role_assignment_name=role_assignment_name
        )
        cli_logger.print("Successfully deleted the role assignment for Storage Blob Data Owner: {}.".format(
            role_assignment_name))
    except Exception as e:
        cli_logger.error(
            "Failed to delete the role assignment for Storage Blob Data Owner: {}. {}".format(
                role_assignment_name, str(e)))
        raise e


def _delete_role_assignment_for_contributor(config, resource_group_name):
    role_assignment_name = get_role_assignment_for_contributor(config, resource_group_name)
    authorization_client = construct_authorization_client(config)
    subscription_id = config["provider"].get("subscription_id")
    scope = "subscriptions/{subscriptionId}/resourcegroups/{resourceGroupName}".format(
        subscriptionId=subscription_id,
        resourceGroupName=resource_group_name
    )
    if role_assignment_name is None:
        cli_logger.print("The role assignment doesn't exist.")
        return

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
            "Failed to delete the role assignment for Contributor: {}. {}".format(role_assignment_name, str(e)))
        raise e


def _delete_role_assignments(config, resource_group_name):
    current_step = 1
    total_steps = 2
    with cli_logger.group("Deleting role assignments: "):
        with cli_logger.group(
                "Deleting role assignment for Contributor",
                _numbered=("()", current_step, total_steps)):
            current_step += 1
            _delete_role_assignment_for_contributor(config, resource_group_name)

        with cli_logger.group(
                "Deleting role assignment for Storage Blob Data Owner",
                _numbered=("()", current_step, total_steps)):
            current_step += 1
            _delete_role_assignment_for_storage_blob_data_owner(config, resource_group_name)


def get_head_user_assigned_identity(config, resource_group_name):
    user_assigned_identity_name = "cloudtik-{}-user-assigned-identity".format(config["workspace_name"])
    return get_user_assigned_identity(config, resource_group_name, user_assigned_identity_name)


def get_worker_user_assigned_identity(config, resource_group_name):
    user_assigned_identity_name = "cloudtik-{}-worker-user-assigned-identity".format(config["workspace_name"])
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
    except Exception as e:
        cli_logger.verbose_error(
            "Failed to get the user assigned identity: {}. {}".format(user_assigned_identity_name, e))
        return None


def _delete_user_assigned_identities(config, resource_group_name):
    workspace_name = config["workspace_name"]
    user_assigned_identity_name = "cloudtik-{}-user-assigned-identity".format(workspace_name)
    _delete_user_assigned_identity(config, resource_group_name, user_assigned_identity_name)

    worker_user_assigned_identity_name = "cloudtik-{}-worker-user-assigned-identity".format(workspace_name)
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
            "Failed to delete the user assigned identity: {}. {}".format(user_assigned_identity.name, str(e)))
        raise e


def get_network_security_group(config, network_client, resource_group_name):
    network_security_group_name = "cloudtik-{}-network-security-group".format(config["workspace_name"])

    cli_logger.verbose("Getting the existing network security group: {}.".format(network_security_group_name))
    try:
        network_client.network_security_groups.get(
            resource_group_name,
            network_security_group_name
        )
        cli_logger.verbose("Successfully get the network security group: {}.".format(network_security_group_name))
        return network_security_group_name
    except Exception as e:
        cli_logger.verbose_error("Failed to get the network security group: {}. {}".format(network_security_group_name, e))
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
        cli_logger.error("Failed to delete the network security group: {}. {}".format(network_security_group_name, str(e)))
        raise e


def get_public_ip_address(config, network_client, resource_group_name):
    public_ip_address_name = "cloudtik-{}-public-ip-address".format(config["workspace_name"])

    cli_logger.verbose("Getting the existing public IP address: {}.".format(public_ip_address_name))
    try:
        network_client.public_ip_addresses.get(
            resource_group_name,
            public_ip_address_name
        )
        cli_logger.verbose("Successfully get the public IP address: {}.".format(public_ip_address_name))
        return public_ip_address_name
    except Exception as e:
        cli_logger.verbose_error("Failed to get the public IP address: {}. {}".format(public_ip_address_name, e))
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
        cli_logger.error("Failed to delete the public IP address: {}. {}".format(public_ip_address_name, str(e)))
        raise e


def get_nat_gateway(config, network_client, resource_group_name):
    nat_gateway_name = "cloudtik-{}-nat".format(config["workspace_name"])

    cli_logger.verbose("Getting the existing NAT gateway: {}.".format(nat_gateway_name))
    try:
        network_client.nat_gateways.get(
            resource_group_name,
            nat_gateway_name
        )
        cli_logger.verbose("Successfully get the NAT gateway: {}.".format(nat_gateway_name))
        return nat_gateway_name
    except Exception as e:
        cli_logger.verbose_error("Failed to get the NAT gateway: {}. {}".format(nat_gateway_name, e))
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
        cli_logger.error("Failed to delete the Nat Gateway: {}. {}".format(nat_gateway_name, str(e)))
        raise e


def _delete_vnet(config, resource_client, network_client):
    use_internal_ips = is_use_internal_ip(config)
    resource_group_name = get_resource_group_name(config, resource_client, use_internal_ips)
    virtual_network_name = get_virtual_network_name(config, resource_client, network_client, use_internal_ips)
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
        cli_logger.error("Failed to delete the virtual network: {}. {}".format(virtual_network_name, str(e)))
        raise e


def _delete_resource_group(config, resource_client):
    resource_group_name = get_workspace_resource_group_name(config["workspace_name"], resource_client)

    if resource_group_name is None:
        cli_logger.print("The resource group: {} doesn't exist.".
                         format(resource_group_name))
        return

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
        cli_logger.error("Failed to delete the resource group: {}. {}".format(resource_group_name, str(e)))
        raise e


def create_azure_workspace(config):
    config = copy.deepcopy(config)
    config = _create_workspace(config)
    return config


def _create_workspace(config):
    workspace_name = config["workspace_name"]
    managed_cloud_storage = is_managed_cloud_storage(config)

    current_step = 1
    total_steps = NUM_AZURE_WORKSPACE_CREATION_STEPS
    if managed_cloud_storage:
        total_steps += 2

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
                    "Creating role assignments for managed Identity",
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
        cli_logger.error("Failed to create workspace. {}", str(e))
        raise e

    cli_logger.print(
        "Successfully created workspace: {}.",
        cf.bold(workspace_name))

    return config


def _create_resource_group(config, resource_client):
    workspace_name = config["workspace_name"]
    use_internal_ips = is_use_internal_ip(config)

    if use_internal_ips:
        # No need to create new resource group
        resource_group_name = get_working_node_resource_group_name(resource_client)
        if resource_group_name is None:
            cli_logger.abort("Only when the working node is "
                             "an Azure instance can use use_internal_ips=True.")
    else:

        # Need to create a new resource_group
        resource_group_name = get_workspace_resource_group_name(workspace_name, resource_client)
        if resource_group_name is None:
            resource_group = create_resource_group(config, resource_client)
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
        cli_logger.verbose_error(
            "Failed to get instance metadata: {}", str(e))
        return None


def get_working_node_resource_group_name(resource_client):
    resource_group = get_working_node_resource_group(resource_client)
    return None if resource_group is None else resource_group.name


def get_working_node_resource_group(resource_client):
    metadata = get_azure_instance_metadata()
    if metadata is None:
        cli_logger.error("Failed to get the metadata of the working node. "
                         "Please check whether the working node is a Azure instance or not!")
    resource_group_name = metadata.get("compute", {}).get("name", "")
    try:
        resource_group = resource_client.resource_groups.get(
            resource_group_name
        )
        cli_logger.verbose(
            "Successfully get the resource group: {} for working node.".format(resource_group_name))
        return resource_group
    except Exception as e:
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
            print(subnet.address_prefix)
            if subnet.address_prefix == subnet_address_prefix:
                return virtual_network.name

    return None


def get_working_node_virtual_network_name(resource_client, network_client):
    metadata = get_azure_instance_metadata()
    if metadata is None:
        cli_logger.error("Failed to get the metadata of the working node. "
                         "Please check whether the working node is a Azure instance or not!")
    resource_group_name = metadata.get("compute", {}).get("name", "")
    interfaces = metadata.get("network", {}).get("interface", "")
    subnet = interfaces[0]["ipv4"]["subnet"][0]
    virtual_network_name = get_virtual_network_name_by_subnet(resource_client, network_client, resource_group_name, subnet)
    if virtual_network_name is not None:
        cli_logger.print("Successfully get the VirtualNetworkName for working node.")

    return virtual_network_name


def get_workspace_virtual_network_name(config, network_client):
    resource_group_name = 'cloudtik-{}-resource-group'.format(config["workspace_name"])
    virtual_network_name = 'cloudtik-{}-vnet'.format(config["workspace_name"])
    cli_logger.verbose("Getting the VirtualNetworkName for workspace: {}...".
                       format(virtual_network_name))

    try:
        virtual_network = network_client.virtual_networks.get(
            resource_group_name=resource_group_name,
            virtual_network_name=virtual_network_name
        )
        cli_logger.verbose("Successfully get the VirtualNetworkName: {} for workspace.".
                                 format(virtual_network_name))
        return virtual_network.name
    except Exception as e:
        cli_logger.verbose_error(
            "The virtual network for workspace is not found: {}", str(e))
        return None


def get_workspace_resource_group_name(workspace_name, resource_client):
    resource_group = _get_workspace_resource_group(workspace_name, resource_client)
    return None if resource_group is None else resource_group.name


def _get_resource_group_name(
        workspace_name, resource_client, use_internal_ips):
    resource_group = _get_resource_group(workspace_name, resource_client, use_internal_ips)
    return None if resource_group is None else resource_group.name


def _get_resource_group(
        workspace_name, resource_client, use_internal_ips):
    if use_internal_ips:
        resource_group = get_working_node_resource_group(resource_client)
    else:
        resource_group =  _get_workspace_resource_group(workspace_name, resource_client)

    return resource_group


def _get_workspace_resource_group(workspace_name, resource_client):
    resource_group_name = 'cloudtik-{}-resource-group'.format(workspace_name)
    cli_logger.verbose("Getting the resource group name for workspace: {}...".
                       format(resource_group_name))

    try:
        resource_group = resource_client.resource_groups.get(
            resource_group_name
        )
        cli_logger.verbose(
            "Successfully get the resource group name: {} for workspace.".format(resource_group_name))
        return resource_group
    except Exception as e:
        cli_logger.verbose_error(
            "The resource group for workspace is not found: {}", str(e))
        return None


def create_resource_group(config, resource_client):
    resource_group_name = 'cloudtik-{}-resource-group'.format(config["workspace_name"])

    assert "location" in config["provider"], (
        "Provider config must include location field")
    params = {"location": config["provider"]["location"]}
    cli_logger.print("Creating workspace resource group: {} on Azure...", resource_group_name)
    # create resource group
    try:
        resource_group = resource_client.resource_groups.create_or_update(
            resource_group_name=resource_group_name, parameters=params)
        cli_logger.print("Successfully created workspace resource group: cloudtik-{}-resource_group.".
                         format(config["workspace_name"]))
        return resource_group
    except Exception as e:
        cli_logger.error(
            "Failed to create workspace resource group. {}", str(e))
        raise e


def _create_role_assignment_for_storage_blob_data_owner(config, resource_group_name):
    workspace_name = config["workspace_name"]
    authorization_client = construct_authorization_client(config)
    subscription_id = config["provider"].get("subscription_id")
    scope = "subscriptions/{subscriptionId}/resourcegroups/{resourceGroupName}".format(
        subscriptionId=subscription_id,
        resourceGroupName=resource_group_name
    )
    role_assignment_name = str(uuid.uuid3(uuid.UUID(subscription_id), workspace_name + 'storage_blob_data_owner'))
    user_assigned_identity = get_head_user_assigned_identity(config, resource_group_name)
    cli_logger.print("Creating workspace role assignment for Storage Blob Data Owner: {} on Azure...",
                     role_assignment_name)

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


def _create_role_assignment_for_contributor(config, resource_group_name):
    workspace_name = config["workspace_name"]
    authorization_client = construct_authorization_client(config)
    subscription_id = config["provider"].get("subscription_id")
    scope = "subscriptions/{subscriptionId}/resourcegroups/{resourceGroupName}".format(
        subscriptionId=subscription_id,
        resourceGroupName=resource_group_name
    )
    role_assignment_name = str(uuid.uuid3(uuid.UUID(subscription_id), workspace_name + "contributor"))
    user_assigned_identity = get_head_user_assigned_identity(config, resource_group_name)
    cli_logger.print("Creating workspace role assignment: {} on Azure...", role_assignment_name)

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
    total_steps = 2
    with cli_logger.group("Creating role assignments: "):
        # create role_assignment for Contributor
        with cli_logger.group(
                "Creating role assignment for Contributor",
                _numbered=("()", current_step, total_steps)):
            current_step += 1
            _create_role_assignment_for_contributor(config, resource_group_name)

        # create role_assignment for Storage Blob Data Owner
        with cli_logger.group(
                "Creating role assignment for Storage Blob Data Owner",
                _numbered=("()", current_step, total_steps)):
            current_step += 1
            _create_role_assignment_for_storage_blob_data_owner(config, resource_group_name)


def _create_container_for_storage_account(config, resource_group_name):
    workspace_name = config["workspace_name"]
    container_name = "cloudtik-{}-storage-container".format(workspace_name)
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
    location = config["provider"]["location"]
    subscription_id = config["provider"].get("subscription_id")
    use_internal_ips = is_use_internal_ip(config)
    resource_client = construct_resource_client(config)
    resource_group = _get_resource_group(workspace_name, resource_client, use_internal_ips)

    storage_suffix = str(uuid.uuid3(uuid.UUID(subscription_id), resource_group.id))[-12:]
    account_name = 'storage{}'.format(storage_suffix)
    storage_client = construct_storage_client(config)

    cli_logger.print("Creating workspace storage account: {} on Azure...", account_name)
    # Create storage account
    try:
        poller = storage_client.storage_accounts.begin_create(
            resource_group_name=resource_group_name,
            account_name=account_name,
            parameters={
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
                    "Name": 'cloudtik-{}-storage-account'.format(workspace_name)
                }
            }
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
    workspace_name = config["workspace_name"]
    user_assigned_identity_name = 'cloudtik-{}-user-assigned-identity'.format(workspace_name)
    _create_user_assigned_identity(config, resource_group_name, user_assigned_identity_name)

    worker_user_assigned_identity_name = 'cloudtik-{}-worker-user-assigned-identity'.format(workspace_name)
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


def _create_vnet(config, resource_client, network_client):
    workspace_name = config["workspace_name"]
    use_internal_ips = is_use_internal_ip(config)

    if use_internal_ips:
        # No need to create new virtual network
        virtual_network_name = get_working_node_virtual_network_name(resource_client, network_client)
        if virtual_network_name is None:
            cli_logger.abort("Only when the working node is "
                             "an Azure instance can use use_internal_ips=True.")
    else:

        # Need to create a new virtual network
        if get_workspace_virtual_network_name(config, network_client) is None:
            virtual_network = create_virtual_network(config, resource_client, network_client)
            virtual_network_name = virtual_network.name
        else:
            cli_logger.abort("There is a existing virtual network with the same name: {}, "
                             "if you want to create a new workspace with the same name, "
                             "you need to execute workspace delete first!".format(workspace_name))
    return virtual_network_name


def create_virtual_network(config, resource_client, network_client):
    virtual_network_name = 'cloudtik-{}-vnet'.format(config["workspace_name"])
    use_internal_ips = is_use_internal_ip(config)
    resource_group_name = get_resource_group_name(config, resource_client, use_internal_ips)
    assert "location" in config["provider"], (
        "Provider config must include location field")

    # choose a random subnet, skipping most common value of 0
    random.seed(virtual_network_name)

    params = {
        "address_space": {
            "address_prefixes": [
                "10.{}.0.0/16".format(random.randint(1, 254))
            ]
        },
        "location": config["provider"]["location"]
    }
    cli_logger.print("Creating workspace virtual network: {} on Azure...", virtual_network_name)
    # create virtual network
    try:
        virtual_network = network_client.virtual_networks.begin_create_or_update(
            resource_group_name=resource_group_name,
            virtual_network_name=virtual_network_name, parameters=params).result()
        cli_logger.print("Successfully created workspace virtual network: cloudtik-{}-vnet.".
                         format(config["workspace_name"]))
        return virtual_network
    except Exception as e:
        cli_logger.error(
            "Failed to create workspace virtual network. {}", str(e))
        raise e


def get_subnet(network_client, resource_group_name, virtual_network_name, subnet_name):
    cli_logger.verbose("Getting the existing subnet: {}.".format(subnet_name))
    try:
        subnet = network_client.subnets.get(
            resource_group_name=resource_group_name,
            virtual_network_name=virtual_network_name,
            subnet_name=subnet_name
        )
        cli_logger.verbose("Successfully get the subnet: {}.".format(subnet_name))
        return subnet
    except Exception as e:
        cli_logger.verbose_error("Failed to get the subnet: {}. {}".format(subnet_name, e))
        return None


def _delete_subnet(config, network_client, resource_group_name, virtual_network_name, is_private=True):
    if is_private:
        subnet_attribute = "private"
    else:
        subnet_attribute = "public"

    workspace_name = config["workspace_name"]
    subnet_name = "cloudtik-{}-{}-subnet".format(workspace_name, subnet_attribute)

    if get_subnet(network_client, resource_group_name, virtual_network_name, subnet_name) is None:
        cli_logger.print("The {} subnet \"{}\" is not found for workspace."
                         .format(subnet_attribute, subnet_name))
        return

    """ Delete custom subnet """
    cli_logger.print("Deleting {} subnet: {}...".format(subnet_attribute, subnet_name))
    try:
        network_client.subnets.begin_delete(
            resource_group_name=resource_group_name,
            virtual_network_name=virtual_network_name,
            subnet_name=subnet_name
        ).result()
        cli_logger.print("Successfully deleted {} subnet: {}."
                         .format(subnet_attribute, subnet_name))
    except Exception as e:
        cli_logger.error("Failed to delete the {} subnet: {}! {}"
                         .format(subnet_attribute, subnet_name, str(e)))
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


def _create_and_configure_subnets(config, network_client, resource_group_name, virtual_network_name, is_private=True):
    cidr_block = _configure_azure_subnet_cidr(network_client, resource_group_name, virtual_network_name)
    subscription_id = config["provider"].get("subscription_id")
    workspace_name = config["workspace_name"]
    nat_gateway_name = "cloudtik-{}-nat".format(workspace_name)
    network_security_group_name = "cloudtik-{}-network-security-group".format(workspace_name)
    if is_private:
        subnet_attribute = "private"
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
        subnet_attribute = "public"
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
            subnet_name="cloudtik-{}-{}-subnet".format(config["workspace_name"], subnet_attribute),
            subnet_parameters=subnet_parameters
        ).result()
        cli_logger.print("Successfully created {} subnet: cloudtik-{}-{}-subnet.".
                         format(subnet_attribute, config["workspace_name"], subnet_attribute))
    except Exception as e:
        cli_logger.error("Failed to create subnet. {}", str(e))
        raise e
    return


def _create_nat(config, network_client, resource_group_name, public_ip_address_name):
    subscription_id = config["provider"].get("subscription_id")
    workspace_name = config["workspace_name"]
    nat_gateway_name = "cloudtik-{}-nat".format(workspace_name)

    cli_logger.print("Creating NAT gateway: {}... ".format(nat_gateway_name))
    try:
        nat_gateway = network_client.nat_gateways.begin_create_or_update(
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
    public_ip_address_name = "cloudtik-{}-public-ip-address".format(workspace_name)
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
    network_security_group_name = "cloudtik-{}-network-security-group".format(workspace_name)

    for i in range(0, len(security_rules)):
        security_rules[i]["name"] = "cloudtik-{}-security-rule-{}".format(workspace_name, i)

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

    return current_step


def bootstrap_azure_from_workspace(config):
    if not check_azure_workspace_resource(config):
        workspace_name = config["workspace_name"]
        cli_logger.abort("Azure workspace {} doesn't exist or is in wrong state!", workspace_name)

    config = _configure_key_pair(config)
    config = _configure_workspace_resource(config)
    return config


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
    use_internal_ips = is_use_internal_ip(config)
    resource_client = construct_resource_client(config)
    resource_group_name = get_resource_group_name(config, resource_client, use_internal_ips)
    if use_managed_cloud_storage:
        storage_account = get_storage_account(config)
        container = get_container_for_storage_account(config, resource_group_name)
        if container is None:
            cli_logger.abort("No managed azure storage container was found. If you want to use managed azure storage, "
                             "you should set managed_cloud_storage equal to True when you creating workspace.")
        if "azure_cloud_storage" not in config["provider"]:
            config["provider"]["azure_cloud_storage"] = {}
        config["provider"]["azure_cloud_storage"]["azure.storage.type"] = "datalake"
        config["provider"]["azure_cloud_storage"]["azure.storage.account"] = storage_account.name
        config["provider"]["azure_cloud_storage"]["azure.container"] = container.name

    if "azure_cloud_storage" in config["provider"]:
        user_assigned_identity = get_head_user_assigned_identity(config, resource_group_name)
        config["provider"]["azure_cloud_storage"][
            "azure.user.assigned.identity.client.id"] = user_assigned_identity.client_id
        config["provider"]["azure_cloud_storage"][
            "azure.user.assigned.identity.tenant.id"] = user_assigned_identity.tenant_id

    return config


def _configure_user_assigned_identity_from_workspace(config):
    workspace_name = config["workspace_name"]
    user_assigned_identity_name = "cloudtik-{}-user-assigned-identity".format(workspace_name)

    config["provider"]["userAssignedIdentity"] = user_assigned_identity_name
    for node_type_key in config["available_node_types"].keys():
        node_config = config["available_node_types"][node_type_key][
            "node_config"]
        node_config["azure_arm_parameters"]["userAssignedIdentity"] = user_assigned_identity_name

    return config


def _configure_subnet_from_workspace(config):
    workspace_name = config["workspace_name"]
    use_internal_ips = is_use_internal_ip(config)

    public_subnet = "cloudtik-{}-public-subnet".format(workspace_name)
    private_subnet = "cloudtik-{}-private-subnet".format(workspace_name)

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
    network_security_group_name = "cloudtik-{}-network-security-group".format(workspace_name)

    for node_type_key in config["available_node_types"].keys():
        node_config = config["available_node_types"][node_type_key][
            "node_config"]
        node_config["azure_arm_parameters"]["networkSecurityGroupName"] = network_security_group_name

    return config


def _configure_virtual_network_from_workspace(config):
    use_internal_ips = is_use_internal_ip(config)
    resource_client = construct_resource_client(config)
    network_client = construct_network_client(config)

    virtual_network_name = get_virtual_network_name(config, resource_client, network_client, use_internal_ips)

    for node_type_key in config["available_node_types"].keys():
        node_config = config["available_node_types"][node_type_key][
            "node_config"]
        node_config["azure_arm_parameters"]["virtualNetworkName"] = virtual_network_name

    return config


def _configure_resource_group_from_workspace(config):
    use_internal_ips = is_use_internal_ip(config)
    resource_client = construct_resource_client(config)
    resource_group_name = get_resource_group_name(config, resource_client, use_internal_ips)
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
        return

    # User override, set or remove spot settings for worker node types
    node_types = config["available_node_types"]
    for node_type_name in node_types:
        if node_type_name == config["head_node_type"]:
            continue

        # worker node type
        node_type_data = node_types[node_type_name]
        _configure_spot_for_node_type(
            node_type_data, prefer_spot_node)


def bootstrap_azure(config):
    workspace_name = config.get("workspace_name", "")
    if workspace_name == "":
        config = bootstrap_azure_default(config)
    else:
        config = bootstrap_azure_from_workspace(config)

    _configure_prefer_spot_node(config)
    return config


def bootstrap_azure_default(config):
    config = _configure_key_pair(config)
    config = _configure_resource_group(config)
    config = _configure_provision_public_ip(config)
    return config


def bootstrap_azure_for_read(config):
    workspace_name = config.get("workspace_name", "")
    if workspace_name == "":
        raise ValueError(f"Workspace name is not specified.")

    return _configure_resource_group_from_workspace(config)


def _configure_provision_public_ip(config):
    use_internal_ips = is_use_internal_ip(config)

    for key, node_type in config["available_node_types"].items():
        node_config = node_type["node_config"]
        node_config["azure_arm_parameters"]["provisionPublicIp"] = False if use_internal_ips else True

    return config


def _configure_resource_group(config):
    # TODO: look at availability sets
    # https://docs.microsoft.com/en-us/azure/virtual-machines/windows/tutorial-availability-sets
    subscription_id = config["provider"].get("subscription_id")
    if subscription_id is None:
        subscription_id = get_cli_profile().get_subscription_id()
    credential = AzureCliCredential()
    resource_client = ResourceManagementClient(credential,
                                               subscription_id)
    config["provider"]["subscription_id"] = subscription_id
    logger.info("Using subscription id: %s", subscription_id)

    assert "resource_group" in config["provider"], (
        "Provider config must include resource_group field")
    resource_group = config["provider"]["resource_group"]

    assert "location" in config["provider"], (
        "Provider config must include location field")
    params = {"location": config["provider"]["location"]}

    if "tags" in config["provider"]:
        params["tags"] = config["provider"]["tags"]

    logger.info("Creating/Updating resource group: %s", resource_group)
    resource_client.resource_groups.create_or_update(
        resource_group_name=resource_group, parameters=params)

    # load the template file
    current_path = Path(__file__).parent
    template_path = current_path.joinpath("azure-config-template.json")
    with open(template_path, "r") as template_fp:
        template = json.load(template_fp)

    # choose a random subnet, skipping most common value of 0
    random.seed(resource_group)
    subnet_mask = "10.{}.0.0/16".format(random.randint(1, 254))

    parameters = {
        "properties": {
            "mode": DeploymentMode.incremental,
            "template": template,
            "parameters": {
                "subnet": {
                    "value": subnet_mask
                }
            }
        }
    }

    create_or_update = get_azure_sdk_function(
        client=resource_client.deployments, function_name="create_or_update")
    create_or_update(
        resource_group_name=resource_group,
        deployment_name="cloudtik-config",
        parameters=parameters).wait()

    return config


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
    use_internal_ips = _is_use_internal_ip(provider_config)
    resource_group_name = _get_resource_group_name(
        workspace_name, resource_client, use_internal_ips)
    return _get_workspace_head_nodes(
        resource_group_name=resource_group_name,
        compute_client=compute_client
    )


def _get_workspace_head_nodes(resource_group_name,
                              compute_client):

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
    azure_cloud_storage = provider_config["azure_cloud_storage"]
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
    azure_cloud_storage = provider_config["azure_cloud_storage"]
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

    azure_cloud_storage = provider_config.get("azure_cloud_storage")
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
    use_internal_ips = is_use_internal_ip(config)
    resource_group_name = get_resource_group_name(config, resource_client, use_internal_ips)

    head_nodes = _get_workspace_head_nodes(
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


def construct_resource_client(config):
    return _construct_resource_client(config["provider"])


def _construct_resource_client(provider_config):
    subscription_id = provider_config.get("subscription_id")
    if subscription_id is None:
        subscription_id = get_cli_profile().get_subscription_id()
    credential = AzureCliCredential()
    resource_client = ResourceManagementClient(credential, subscription_id)
    logger.debug("Using subscription id: %s", subscription_id)
    return resource_client


def construct_storage_client(config):
    return _construct_storage_client(config["provider"])


def _construct_storage_client(provider_config):
    subscription_id = provider_config.get("subscription_id")
    if subscription_id is None:
        subscription_id = get_cli_profile().get_subscription_id()
    credential = AzureCliCredential()
    storage_client = StorageManagementClient(credential, subscription_id)
    return storage_client


def construct_network_client(config):
    return _construct_network_client(config["provider"])


def _construct_network_client(provider_config):
    subscription_id = provider_config.get("subscription_id")
    if subscription_id is None:
        subscription_id = get_cli_profile().get_subscription_id()
    credential = AzureCliCredential()
    network_client = NetworkManagementClient(credential, subscription_id)

    return network_client


def construct_compute_client(config):
    return _construct_compute_client(config["provider"])


def _construct_compute_client(provider_config):
    subscription_id = provider_config.get("subscription_id")
    if subscription_id is None:
        subscription_id = get_cli_profile().get_subscription_id()
    credential = AzureCliCredential()
    compute_client = ComputeManagementClient(credential, subscription_id)

    return compute_client


def construct_manage_server_identity_client(config):
    return _construct_manage_server_identity_client(config["provider"])


def _construct_manage_server_identity_client(provider_config):
    subscription_id = provider_config.get("subscription_id")
    if subscription_id is None:
        subscription_id = get_cli_profile().get_subscription_id()
    credential = AzureCliCredential()
    # It showed that we no longer need to wrapper. Will fail with wrapper: no attribute get_token
    # wrapped_credential = AzureIdentityCredentialAdapter(credential)
    msi_client = ManagedServiceIdentityClient(credential, subscription_id)

    return msi_client


def construct_authorization_client(config):
    return _construct_authorization_client(config["provider"])


def _construct_authorization_client(provider_config):
    subscription_id = provider_config.get("subscription_id")
    if subscription_id is None:
        subscription_id = get_cli_profile().get_subscription_id()
    credential = AzureCliCredential()
    wrapped_credential = AzureIdentityCredentialAdapter(credential)
    authorization_client = AuthorizationManagementClient(
        credentials=wrapped_credential,
        subscription_id=subscription_id,
        api_version="2018-01-01-preview"
    )
    return authorization_client
