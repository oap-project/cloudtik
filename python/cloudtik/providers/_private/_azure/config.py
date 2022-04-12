import copy
import json
import logging
import time
import uuid
import subprocess
from pathlib import Path
import random

from typing import Any, Callable

from cloudtik.core._private.cli_logger import cli_logger, cf
from cloudtik.core._private.utils import check_cidr_conflict
from cloudtik.providers._private._azure.azure_identity_credential_adapter import AzureIdentityCredentialAdapter

from azure.common.credentials import get_cli_profile
from azure.identity import AzureCliCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.msi import ManagedServiceIdentityClient
from azure.mgmt.authorization import AuthorizationManagementClient
from azure.mgmt.resource.resources.models import DeploymentMode


RETRIES = 30
MSI_NAME = "cloudtik-msi-user-identity"
NSG_NAME = "cloudtik-nsg"
SUBNET_NAME = "cloudtik-subnet"
VNET_NAME = "cloudtik-vnet"
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
    use_internal_ips = config["provider"].get("use_internal_ips", False)
    workspace_name = config["workspace_name"]
    network_client = construct_network_client(config)
    resource_client = construct_resource_client(config)

    """
         Do the work - order of operation
         1). Check resource group
         2.) Check vpc 
         3.) Check network security group
         4.) Check public-ip-address
         5.) Check nat-gateway
         6.) Check private subnet 
         7.) Check public subnet
         8.) Check role_assignments
         9.) Check user_assigned_identities
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

    private_subnet_name = "cloudtik-private-{}-subnet".format(workspace_name)
    if get_subnet(network_client, resource_group_name, virtual_network_name, private_subnet_name) is None:
        return False

    public_subnet_name = "cloudtik-public-{}-subnet".format(workspace_name)
    if get_subnet(network_client, resource_group_name, virtual_network_name, public_subnet_name) is None:
        return False

    if get_role_assignments(config, resource_group_name) is None:
        return False

    if get_user_assigned_identities(config, resource_group_name) is None:
        return False

    return True


def get_resource_group_name(config, resource_client, use_internal_ips):
    if use_internal_ips:
        resource_group_name = get_working_node_resource_group_name()
    else:
        resource_group_name = get_workspace_resource_group_name(config, resource_client)

    return resource_group_name


def get_virtual_network_name(config, resource_client, network_client, use_internal_ips):
    if use_internal_ips:
        virtual_network_name = get_working_node_virtual_network_name(resource_client, network_client)
    else:
        virtual_network_name = get_workspace_virtual_network_name(config, network_client)

    return virtual_network_name


def delete_workspace_azure(config):
    resource_client = construct_resource_client(config)
    workspace_name = config["workspace_name"]
    use_internal_ips = config["provider"].get("use_internal_ips", False)
    resource_group_name = get_resource_group_name(config, resource_client, use_internal_ips)

    if resource_group_name is None:
        cli_logger.print("Workspace: {} doesn't exist!".format(config["workspace_name"]))
        return

    current_step = 1
    total_steps = NUM_AZURE_WORKSPACE_DELETION_STEPS
    if not use_internal_ips:
        total_steps += 2

    try:
        # delete network resources
        with cli_logger.group("Deleting workspace: {}", workspace_name):
            current_step = _delete_network_resources(config, resource_client, resource_group_name, current_step, total_steps)

        # delete role_assignments
        with cli_logger.group(
                "Deleting role assignments",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _delete_role_assignments(config, resource_group_name)

        # delete user_assigned_identities
        with cli_logger.group(
                "Deleting user_assigned_identities",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _delete_user_assigned_identities(config, resource_group_name)

        # delete resource group
        if not use_internal_ips:
            with cli_logger.group(
                    "Deleting Resource Group",
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
    use_internal_ips = config["provider"].get("use_internal_ips", False)
    network_client = construct_network_client(config)
    virtual_network_name = get_virtual_network_name(config, resource_client, network_client, use_internal_ips)

    """
         Do the work - order of operation
         1.) Delete public subnet
         2.) Delete private subnet 
         3.) Delete rnat-gateway
         4.) Delete public-ip-address
         5.) Delete network security group
         6.) Delete vpc
    """

    # delete public subnets
    with cli_logger.group(
            "Deleting public subnet",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_subnet(config, network_client, resource_group_name, virtual_network_name, isPrivate=False)

    # delete private subnets
    with cli_logger.group(
            "Deleting private subnet",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_subnet(config, network_client, resource_group_name, virtual_network_name, isPrivate=True)

    # delete nat-gateway
    with cli_logger.group(
            "Deleting nat-gateway",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_nat(config, network_client, resource_group_name)

    # delete public-ip-address
    with cli_logger.group(
            "Deleting public-ip-address",
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


def get_role_assignments(config, resource_group_name):
    workspace_name = config["workspace_name"]
    authorization_client = construct_authorization_client(config)
    subscriptionId = config["provider"].get("subscription_id")
    scope = "subscriptions/{subscriptionId}/resourcegroups/{resourceGroupName}".format(
        subscriptionId=subscriptionId,
        resourceGroupName=resource_group_name
    )
    role_assignment_name = str(uuid.uuid3(uuid.UUID(subscriptionId), workspace_name))
    cli_logger.print("Getting the existing role-assignment: {}.", role_assignment_name)

    try:
        role_assignment = authorization_client.role_assignments.get(
            scope=scope,
            role_assignment_name=role_assignment_name,
        )
        cli_logger.print("Successfully get the role-assignment: {}.".
                         format(role_assignment_name))
        return role_assignment_name
    except Exception as e:
        cli_logger.error(
            "Failed to get the role-assignment. {}", str(e))
        return None


def _delete_role_assignments(config, resource_group_name):
    role_assignments_name = get_role_assignments(config, resource_group_name)
    authorization_client = construct_authorization_client(config)
    subscriptionId = config["provider"].get("subscription_id")
    scope = "subscriptions/{subscriptionId}/resourcegroups/{resourceGroupName}".format(
        subscriptionId=subscriptionId,
        resourceGroupName=resource_group_name
    )
    if role_assignments_name is None:
        cli_logger.print("This role_assignments has not existed. No need to delete it.")
        return

    """ Delete the role_assignments """
    cli_logger.print("Deleting the role_assignments: {}...".format(role_assignments_name))
    try:
        authorization_client.role_assignments.delete(
            scope=scope,
            role_assignment_name=role_assignments_name
        )
        cli_logger.print("Successfully deleted the role_assignments: {}.".format(role_assignments_name))
    except Exception as e:
        cli_logger.error(
            "Failed to delete the role_assignments:{}. {}".format(role_assignments_name, str(e)))
        raise e


def get_user_assigned_identities(config, resource_group_name):
    user_assigned_identity_name = "cloudtik-{}-user-assigned-identity".format(config["workspace_name"])
    msi_client = construct_manage_server_identity_client(config)
    cli_logger.verbose("Getting the existing user-assigned-identity: {}.".format(user_assigned_identity_name))
    try:
        user_assigned_identity = msi_client.user_assigned_identities.get(
            resource_group_name,
            user_assigned_identity_name
        )
        cli_logger.verbose("Successfully get the user-assigned-identity: {}.".format(user_assigned_identity_name))
        return user_assigned_identity
    except Exception as e:
        cli_logger.verbose_error(
            "Failed to get the user-assigned-identity: {}. {}".format(user_assigned_identity_name, e))
        return None


def _delete_user_assigned_identities(config, resource_group_name):
    user_assigned_identity = get_user_assigned_identities(config, resource_group_name)
    msi_client = construct_manage_server_identity_client(config)
    if user_assigned_identity is None:
        cli_logger.print("This user_assigned_identity has not existed. No need to delete it.")
        return

    """ Delete the user_assigned_identity """
    cli_logger.print("Deleting the user_assigned_identity: {}...".format(user_assigned_identity.name))
    try:
        msi_client.user_assigned_identities.delete(
            resource_group_name=resource_group_name,
            resource_name=user_assigned_identity.name
        )
        cli_logger.print("Successfully deleted the user_assigned_identity: {}.".format(user_assigned_identity.name))
    except Exception as e:
        cli_logger.error(
            "Failed to delete the user_assigned_identity:{}. {}".format(user_assigned_identity.name, str(e)))
        raise e


def get_network_security_group(config, network_client, resource_group_name):
    network_security_group_name = "cloudtik-{}-network-security-group".format(config["workspace_name"])

    cli_logger.verbose("Getting the existing network-security-group: {}.".format(network_security_group_name))
    try:
        network_client.network_security_groups.get(
            resource_group_name,
            network_security_group_name
        )
        cli_logger.verbose("Successfully get the network-security-group: {}.".format(network_security_group_name))
        return network_security_group_name
    except Exception as e:
        cli_logger.verbose_error("Failed to get the network-security-group: {}. {}".format(network_security_group_name, e))
        return None


def _delete_network_security_group(config, network_client, resource_group_name):
    network_security_group_name = get_network_security_group(config, network_client, resource_group_name)
    if network_security_group_name is None:
        cli_logger.print("This network security group has not existed. No need to delete it.")
        return

    """ Delete the network security group """
    cli_logger.print("Deleting the network security group: {}...".format(network_security_group_name))
    try:
        network_client.network_security_groups.begin_delete(
            resource_group_name=resource_group_name,
            network_security_group_name=network_security_group_name
        ).result()
        cli_logger.print("Successfully deleted the network security group: {}.".format(network_security_group_name))
    except Exception as e:
        cli_logger.error("Failed to delete the network security group:{}. {}".format(network_security_group_name, str(e)))
        raise e


def get_public_ip_address(config, network_client, resource_group_name):
    public_ip_address_name = "cloudtik-{}-public-ip-address".format(config["workspace_name"])

    cli_logger.verbose("Getting the existing public-ip-address: {}.".format(public_ip_address_name))
    try:
        network_client.public_ip_addresses.get(
            resource_group_name,
            public_ip_address_name
        )
        cli_logger.verbose("Successfully get the public-ip-address: {}.".format(public_ip_address_name))
        return public_ip_address_name
    except Exception as e:
        cli_logger.verbose_error("Failed to get the public-ip-address: {}. {}".format(public_ip_address_name, e))
        return None


def _delete_public_ip_address(config, network_client, resource_group_name):
    public_ip_address_name = get_public_ip_address(config, network_client, resource_group_name)
    if public_ip_address_name is None:
        cli_logger.print("This Public Ip Address has not existed. No need to delete it.")
        return

    """ Delete the Public Ip Address """
    cli_logger.print("Deleting the Public Ip Address: {}...".format(public_ip_address_name))
    try:
        network_client.public_ip_addresses.begin_delete(
            resource_group_name=resource_group_name,
            public_ip_address_name=public_ip_address_name
        ).result()
        cli_logger.print("Successfully deleted the Public Ip Address: {}.".format(public_ip_address_name))
    except Exception as e:
        cli_logger.error("Failed to delete the Public Ip Address:{}. {}".format(public_ip_address_name, str(e)))
        raise e


def get_nat_gateway(config, network_client, resource_group_name):
    nat_gateway_name = "cloudtik-{}-nat".format(config["workspace_name"])

    cli_logger.verbose("Getting the existing nat-gateway: {}.".format(nat_gateway_name))
    try:
        network_client.nat_gateways.get(
            resource_group_name,
            nat_gateway_name
        )
        cli_logger.verbose("Successfully get the nat-gateway: {}.".format(nat_gateway_name))
        return nat_gateway_name
    except Exception as e:
        cli_logger.verbose_error("Failed to get the nat-gateway: {}. {}".format(nat_gateway_name, e))
        return None


def _delete_nat(config, network_client, resource_group_name):
    nat_gateway_name = get_nat_gateway(config, network_client, resource_group_name)
    if nat_gateway_name is None:
        cli_logger.print("This Nat Gateway has not existed. No need to delete it.")
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
        cli_logger.error("Failed to delete the Nat Gateway:{}. {}".format(nat_gateway_name, str(e)))
        raise e


def _delete_vnet(config, resource_client, network_client):
    use_internal_ips = config["provider"].get("use_internal_ips", False)
    resource_group_name = get_resource_group_name(config, resource_client, use_internal_ips)
    virtual_network_name = get_virtual_network_name(config, resource_client, network_client, use_internal_ips)
    if virtual_network_name is None:
        cli_logger.print("This Virtual Network: {} has not existed. No need to delete it.".
                         format(virtual_network_name))
        return

    """ Delete the Virtual Network """
    cli_logger.print("Deleting the Virtual Network: {}...".format(virtual_network_name))
    try:
        network_client.virtual_networks.begin_delete(
            resource_group_name=resource_group_name,
            virtual_network_name=virtual_network_name
        ).result()
        cli_logger.print("Successfully deleted the Virtual Network: {}.".format(virtual_network_name))
    except Exception as e:
        cli_logger.error("Failed to delete the Virtual Network:{}. {}".format(virtual_network_name, str(e)))
        raise e


def _delete_resource_group(config, resource_client):
    resource_group_name = get_workspace_resource_group_name(config, resource_client)

    if resource_group_name is None:
        cli_logger.print("This Resource Group: {} has not existed. No need to delete it.".
                         format(resource_group_name))
        return

    """ Delete the Resource Group """
    cli_logger.print("Deleting the Resource Group: {}...".format(resource_group_name))

    try:
        resource_client.resource_groups.begin_delete(
            resource_group_name
        ).result()
        cli_logger.print("Successfully deleted the Resource Group: {}.".format(resource_group_name))
    except Exception as e:
        cli_logger.error("Failed to delete the Resource Group:{}. {}".format(resource_group_name, str(e)))
        raise e


def create_azure_workspace(config):
    config = copy.deepcopy(config)
    # TODO: create vpc and security group

    config = _configure_workspace(config)

    return config


def _configure_workspace(config):
    workspace_name = config["workspace_name"]

    current_step = 1
    total_steps = NUM_AZURE_WORKSPACE_CREATION_STEPS

    resource_client = construct_resource_client(config)

    try:
        # create resource group
        with cli_logger.group(
                "Creating Resource Group",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            resource_group_name = _create_resource_group(config, resource_client)

        # create network resources
        with cli_logger.group(
                "Creating workspace: {}", workspace_name):
            current_step = _configure_network_resources(config, resource_group_name, current_step, total_steps)

        # create user_assigned_identities
        with cli_logger.group(
                "Creating user assigned identites",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _create_user_assigned_identities(config, resource_group_name)

        # create role_assignments
        with cli_logger.group(
                "Creating role assignments",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _create_role_assignments(config, resource_group_name)

    except Exception as e:
        cli_logger.error("Failed to create workspace. {}", str(e))
        raise e

    cli_logger.print(
        "Successfully created workspace: {}.",
        cf.bold(workspace_name))

    return config


def _create_resource_group(config, resource_client):
    workspace_name = config["workspace_name"]
    use_internal_ips = config["provider"].get("use_internal_ips", False)

    if use_internal_ips:
        # No need to create new resource group
        resource_group_name = get_working_node_resource_group_name()
        if resource_group_name is None:
            cli_logger.abort("Only when the working node is "
                             "an Azure instance can use use_internal_ips=True.")
    else:

        # Need to create a new resource_group
        if get_workspace_resource_group_name(config, resource_client) is None:
            resource_group = create_resource_group(config, resource_client)
            resource_group_name = resource_group.name
        else:
            cli_logger.abort("There is a existing Resource Group with the same name: {}, "
                             "if you want to create a new workspace with the same name, "
                             "you need to execute workspace delete first!".format(workspace_name))
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


def get_working_node_resource_group_name():
    metadata = get_azure_instance_metadata()
    if metadata is None:
        cli_logger.error("Failed to get the metadata of the working node. "
                         "Please check whether the working node is a Azure instance or not!")
    resource_group_name = metadata.get("compute", {}).get("name", "")
    cli_logger.print("Successfully get the ResourceGroupName for working node.")

    return resource_group_name


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
        cli_logger.verbose_error("Successfully get the VirtualNetworkName: {} for workspace.".
                                 format(virtual_network_name))
        return virtual_network.name
    except Exception as e:
        cli_logger.verbose_error(
            "The Virtual Network for workspace is not found: {}", str(e))
        return None


def get_workspace_resource_group_name(config, resource_client):
    resource_group_name = 'cloudtik-{}-resource-group'.format(config["workspace_name"])
    cli_logger.verbose("Getting the ResourceGroupName for workspace: {}...".
                       format(resource_group_name))

    try:
        resource_group = resource_client.resource_groups.get(
            resource_group_name
        )
        cli_logger.verbose(
            "Successfully get the ResourceGroupName: {} for workspace.".format(resource_group_name))
        return resource_group.name
    except Exception as e:
        cli_logger.verbose_error(
            "The Resource Group for workspace is not found: {}", str(e))
        return None


def create_resource_group(config, resource_client):
    resource_group_name = 'cloudtik-{}-resource-group'.format(config["workspace_name"])

    assert "location" in config["provider"], (
        "Provider config must include location field")
    params = {"location": config["provider"]["location"]}
    cli_logger.print("Creating workspace Resource Group: {} on Azure...", resource_group_name)
    # create resource group
    try:

        resource_group = resource_client.resource_groups.create_or_update(
            resource_group_name=resource_group_name, parameters=params)
        # time.sleep(20)
        cli_logger.print("Successfully created workspace Resource Group: cloudtik-{}-resource_group.".
                         format(config["workspace_name"]))
        return resource_group
    except Exception as e:
        cli_logger.error(
            "Failed to create workspace Resource Group. {}", str(e))
        raise e


def _create_role_assignments(config, resource_group_name):
    workspace_name = config["workspace_name"]
    authorization_client = construct_authorization_client(config)
    subscriptionId = config["provider"].get("subscription_id")
    scope = "subscriptions/{subscriptionId}/resourcegroups/{resourceGroupName}".format(
        subscriptionId=subscriptionId,
        resourceGroupName=resource_group_name
    )
    role_assignment_name = str(uuid.uuid3(uuid.UUID(subscriptionId), workspace_name))
    user_assigned_identity = get_user_assigned_identities(config,resource_group_name)
    cli_logger.print("Creating workspace role-assignment: {} on Azure...", role_assignment_name)

    # Create role assignment
    try:
        role_assignment = authorization_client.role_assignments.create(
            scope=scope,
            role_assignment_name=role_assignment_name,
            parameters={
                "role_definition_id": "/subscriptions/{}/providers/Microsoft.Authorization/roleDefinitions/b24988ac-6180-42a0-ab88-20f7382dd24c".format(
                    subscriptionId),
                "principal_id": user_assigned_identity.principal_id,
                "principalType": "ServicePrincipal"
            }
        )
        time.sleep(20)
        cli_logger.print("Successfully created workspace role-assignment: {}.".
                         format(role_assignment_name))
    except Exception as e:
        cli_logger.error(
            "Failed to create workspace role-assignment. {}", str(e))
        raise e


def _create_user_assigned_identities(config, resource_group_name):
    workspace_name = config["workspace_name"]
    location = config["provider"]["location"]
    user_assigned_identiy_name = 'cloudtik-{}-user-assigned-identity'.format(workspace_name)
    msi_client = construct_manage_server_identity_client(config)

    cli_logger.print("Creating workspace user-assigned-identity: {} on Azure...", user_assigned_identiy_name)
    # Create identity
    try:
        msi_client.user_assigned_identities.create_or_update(
            resource_group_name,
            user_assigned_identiy_name,
            location,
        )
        time.sleep(20)
        cli_logger.print("Successfully created workspace user-assigned-identity: {}.".
                         format(user_assigned_identiy_name))
    except Exception as e:
        cli_logger.error(
            "Failed to create workspace user-assigned-identity. {}", str(e))
        raise e


def _create_vnet(config, resource_client, network_client):
    workspace_name = config["workspace_name"]
    use_internal_ips = config["provider"].get("use_internal_ips", False)

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
            cli_logger.abort("There is a existing Virtual Network with the same name: {}, "
                             "if you want to create a new workspace with the same name, "
                             "you need to execute workspace delete first!".format(workspace_name))
    return virtual_network_name


def create_virtual_network(config, resource_client, network_client):
    virtual_network_name = 'cloudtik-{}-vnet'.format(config["workspace_name"])
    use_internal_ips = config["provider"].get("use_internal_ips", False)
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
    cli_logger.print("Creating workspace Virtual Network: {} on Azure...", virtual_network_name)
    # create virtual network
    try:
        virtual_network = network_client.virtual_networks.begin_create_or_update(
            resource_group_name=resource_group_name,
            virtual_network_name=virtual_network_name, parameters=params).result()
        cli_logger.print("Successfully created workspace Virtual Network: cloudtik-{}-vnet.".
                         format(config["workspace_name"]))
        return virtual_network
    except Exception as e:
        cli_logger.error(
            "Failed to create workspace Virtual Network. {}", str(e))
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


def _delete_subnet(config, network_client, resource_group_name, virtual_network_name, isPrivate=True):
    if isPrivate:
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


def _create_and_configure_subnets(config, network_client, resource_group_name, virtual_network_name, isPrivate=True):
    cidr_block = _configure_azure_subnet_cidr(network_client, resource_group_name, virtual_network_name)
    subscription_id = config["provider"].get("subscription_id")
    workspace_name = config["workspace_name"]
    nat_gateway_name = "cloudtik-{}-nat".format(workspace_name)
    network_security_group_name = "cloudtik-{}-network-security-group".format(workspace_name)
    if isPrivate:
        subnet_attribute = "private"
        subnet_parameters = {
            "address_prefix": cidr_block,
            "nat_gateway": {
                "id": "/subscriptions/{}/resourceGroups/{}/providers/Microsoft.Network/natGateways/{}"
                    .format(subscription_id, resource_group_name, nat_gateway_name)
            },
            "network_security_group":{
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

    cli_logger.print("Creating nat-gateway: {}... ".format(nat_gateway_name))
    try:
        nat_gateway = network_client.nat_gateways.begin_create_or_update(
            resource_group_name=resource_group_name,
            nat_gateway_name=nat_gateway_name,
            parameters = {
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
        cli_logger.print("Successfully created nat-gateway: {}.".
                         format(nat_gateway_name))
    except Exception as e:
        cli_logger.error("Failed to create nat-gateway. {}", str(e))
        raise e


def _create_public_ip_address(config, network_client, resource_group_name):
    workspace_name = config["workspace_name"]
    public_ip_address_name = "cloudtik-{}-public-ip-address".format(workspace_name)
    location = config["provider"]["location"]

    cli_logger.print("Creating public-ip-address: {}... ".format(public_ip_address_name))
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
        )
        cli_logger.print("Successfully created public-ip-address: {}.".
                         format(public_ip_address_name))
    except Exception as e:
        cli_logger.error("Failed to create public-ip-address. {}", str(e))
        raise e

    return public_ip_address_name


def _create_network_security_group(config, network_client, resource_group_name):
    workspace_name = config["workspace_name"]
    location = config["provider"]["location"]
    securityRules = config["provider"].get("securityRules", [])
    network_security_group_name = "cloudtik-{}-network-security-group".format(workspace_name)

    for i in range(0, len(securityRules)):
        securityRules[i]["name"] = "cloudtik-{}-security-rule-{}".format(workspace_name, i)

    cli_logger.print("Creating network-security-group: {}... ".format(network_security_group_name))
    try:
        network_client.network_security_groups.begin_create_or_update(
            resource_group_name=resource_group_name,
            network_security_group_name=network_security_group_name,
            parameters={
                "location": location,
                "securityRules": securityRules
            }
        ).result()
        cli_logger.print("Successfully created network-security-group: {}.".
                         format(network_security_group_name))
    except Exception as e:
        cli_logger.error("Failed to create network-security-group. {}", str(e))
        raise e

    return network_security_group_name


def _configure_network_resources(config, resource_group_name, current_step, total_steps):
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
        _create_network_security_group(config, network_client, resource_group_name)

    # create public subnet
    with cli_logger.group(
            "Creating public subnet",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_and_configure_subnets(
            config, network_client, resource_group_name, virtual_network_name, isPrivate=False)

    # create public-ip-address
    with cli_logger.group(
            "Creating public-ip-address for nat-gateway",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        public_ip_address_name = _create_public_ip_address(config, network_client, resource_group_name,)

    # create nat-gateway
    with cli_logger.group(
            "Creating nat-gateway",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_nat(config, network_client, resource_group_name, public_ip_address_name)

    # create private subnet
    with cli_logger.group(
            "Creating private subnet",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_and_configure_subnets(
            config, network_client, resource_group_name, virtual_network_name, isPrivate=True)

    return current_step


def bootstrap_azure_from_workspace(config):
    config = _configure_key_pair(config)
    config = _configure_workspace_resource(config)
    return config


def _configure_workspace_resource(config):
    config = _configure_resource_group_from_workspace(config)
    config = _configure_virtual_network_from_workspace(config)
    config = _configure_subnet_from_workspace(config)
    config = _configure_network_security_group_from_workspace(config)
    config = _configure_user_assigned_identities_from_workspace(config)
    return config


def _configure_user_assigned_identities_from_workspace(config):
    workspace_name = config["workspace_name"]
    user_assigned_identity_name = "cloudtik-{}-user-assigned-identity".format(workspace_name)

    for node_type_key in config["available_node_types"].keys():
        node_config = config["available_node_types"][node_type_key][
            "node_config"]
        node_config["userAssignedIdentities"] = user_assigned_identity_name

    return config


def _configure_subnet_from_workspace(config):
    workspace_name = config["workspace_name"]
    use_internal_ips = config["provider"].get("use_internal_ips", False)

    public_subnet = "cloudtik-{}-public-subnet".format(workspace_name)
    private_subnet = "cloudtik-{}-private-subnet".format(workspace_name)

    for key, node_type in config["available_node_types"].items():
        node_config = node_type["node_config"]
        if key == config["head_node_type"]:
            if use_internal_ips:
                node_config["subnet"] = private_subnet
            else:
                node_config["subnet"] = public_subnet
        else:
            node_config["subnet"] = private_subnet

    return config


def _configure_network_security_group_from_workspace(config):
    workspace_name = config["workspace_name"]
    network_security_group_name = "cloudtik-{}-network-security-group".format(workspace_name)

    for node_type_key in config["available_node_types"].keys():
        node_config = config["available_node_types"][node_type_key][
            "node_config"]
        node_config["networkSecurityGroup"] = network_security_group_name

    return config


def _configure_virtual_network_from_workspace(config):
    use_internal_ips = config["provider"].get("use_internal_ips", False)
    resource_client = construct_resource_client(config)
    network_client = construct_network_client(config)

    virtual_network_name = get_virtual_network_name(config, resource_client, network_client, use_internal_ips)

    for node_type_key in config["available_node_types"].keys():
        node_config = config["available_node_types"][node_type_key][
            "node_config"]
        node_config["virtualNetwork"] = virtual_network_name

    return config


def _configure_resource_group_from_workspace(config):
    use_internal_ips = config["provider"].get("use_internal_ips", False)
    resource_client = construct_resource_client(config)
    resource_group_name = get_resource_group_name(config, resource_client, use_internal_ips)
    config["provider"]["resource_group"] = resource_group_name
    return config


def bootstrap_azure(config):
    config = _configure_key_pair(config)
    config = _configure_resource_group(config)
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

    logger.info("Creating/Updating Resource Group: %s", resource_group)
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


def construct_resource_client(config):
    subscription_id = config["provider"].get("subscription_id")
    if subscription_id is None:
        subscription_id = get_cli_profile().get_subscription_id()
    credential = AzureCliCredential()
    resource_client = ResourceManagementClient(credential, subscription_id)
    logger.info("Using subscription id: %s", subscription_id)
    return resource_client


def construct_network_client(config):
    subscription_id = config["provider"].get("subscription_id")
    if subscription_id is None:
        subscription_id = get_cli_profile().get_subscription_id()
    credential = AzureCliCredential()
    network_client = NetworkManagementClient(credential, subscription_id)

    return network_client


def construct_compute_client(config):
    subscription_id = config["provider"].get("subscription_id")
    if subscription_id is None:
        subscription_id = get_cli_profile().get_subscription_id()
    credential = AzureCliCredential()
    compute_client = ComputeManagementClient(credential, subscription_id)

    return compute_client


def construct_manage_server_identity_client(config):
    subscription_id = config["provider"].get("subscription_id")
    if subscription_id is None:
        subscription_id = get_cli_profile().get_subscription_id()
    credential = AzureCliCredential()
    wrapped_credential = AzureIdentityCredentialAdapter(credential)
    msi_client = ManagedServiceIdentityClient(wrapped_credential, subscription_id)

    return msi_client


def construct_authorization_client(config):
    subscription_id = config["provider"].get("subscription_id")
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