import copy
import json
import logging
import subprocess
from pathlib import Path
import random
import time
from typing import Any, Callable

from cloudtik.core._private.cli_logger import cli_logger, cf

from azure.common.credentials import get_cli_profile
from azure.identity import AzureCliCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.resource.resources.models import DeploymentMode


RETRIES = 30
MSI_NAME = "cloudtik-msi-user-identity"
NSG_NAME = "cloudtik-nsg"
SUBNET_NAME = "cloudtik-subnet"
VNET_NAME = "cloudtik-vnet"
NUM_AZURE_WORKSPACE_CREATION_STEPS = 6
NUM_AZURE_WORKSPACE_DELETION_STEPS = 4

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
        total_steps += 1

    try:
        # delete network resources
        with cli_logger.group("Deleting workspace: {}", workspace_name):
            _delete_network_resources(config, resource_client, current_step, total_steps)

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


def _delete_network_resources(config, resource_client, current_step, total_steps):
    use_internal_ips = config["provider"].get("use_internal_ips", False)
    network_client = construct_network_client(config)

    """
         Do the work - order of operation
         1.) Delete public subnet
         2.) Delete router for private subnet 
         3.) Delete private subnets
         4.) Delete firewalls
         5.) Delete virtual network
    """

    # delete public subnets
    # with cli_logger.group(
    #         "Deleting public subnet",
    #         _numbered=("[]", current_step, total_steps)):
    #     current_step += 1
    #     _delete_subnet(config, compute, isPrivate=False)

    # delete router for private subnets
    # with cli_logger.group(
    #         "Deleting router",
    #         _numbered=("[]", current_step, total_steps)):
    #     current_step += 1
    #     _delete_router(config, compute)

    # delete private subnets
    # with cli_logger.group(
    #         "Deleting private subnet",
    #         _numbered=("[]", current_step, total_steps)):
    #     current_step += 1
    #     _delete_subnet(config, compute, isPrivate=True)

    # delete firewalls
    # with cli_logger.group(
    #         "Deleting firewall rules",
    #         _numbered=("[]", current_step, total_steps)):
    #     current_step += 1
    #     _delete_firewalls(config, compute)

    # delete virtual network
    if not use_internal_ips:
        with cli_logger.group(
                "Deleting VPC",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _delete_vnet(config, resource_client, network_client)


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

    return


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

    return


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
            _create_resource_group(config, resource_client)

        # create network resources
        with cli_logger.group("Creating workspace: {}", workspace_name):
            config = _configure_network_resources(config, current_step, total_steps)

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
        cli_logger.verbose("Successfully get the ResourceGroupName: {} for workspace.".
                                 format(resource_group_name))
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


# def _create_resource_group(config, resource_client):
#     workspace_name = config["workspace_name"]
#     use_internal_ips = config["provider"].get("use_internal_ips", False)
#
#     if use_internal_ips:
#         # No need to create new resource group
#         resource_group_name = get_working_node_resource_group_name()
#         if resource_group_name is None:
#             cli_logger.abort("Only when the working node is "
#                              "an Azure instance can use use_internal_ips=True.")
#     else:
#
#         # Need to create a new resource_group
#         if get_workspace_resource_group_name(config, resource_client) is None:
#             resource_group = create_resource_group(config, resource_client)
#             resource_group_name = resource_group.name
#         else:
#             cli_logger.abort("There is a existing Resource Group with the same name: {}, "
#                              "if you want to create a new workspace with the same name, "
#                              "you need to execute workspace delete first!".format(workspace_name))
#     return resource_group_name


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
        # time.sleep(20)
        cli_logger.print("Successfully created workspace Virtual Network: cloudtik-{}-vnet.".
                         format(config["workspace_name"]))
        return virtual_network
    except Exception as e:
        cli_logger.error(
            "Failed to create workspace Virtual Network. {}", str(e))
        raise e


def _configure_network_resources(config, current_step, total_steps):
    network_client = construct_network_client(config)
    resource_client = construct_resource_client(config)
    # create virtual network
    with cli_logger.group(
            "Creating Virtual Network",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        virtual_network = _create_vnet(config, resource_client, network_client)

    # # create subnets
    # with cli_logger.group(
    #         "Creating subnets",
    #         _numbered=("[]", current_step, total_steps)):
    #     current_step += 1
    #     _create_and_configure_subnets(config, compute, VpcId)
    #
    # # create router
    # with cli_logger.group(
    #         "Creating router",
    #         _numbered=("[]", current_step, total_steps)):
    #     current_step += 1
    #     _create_router(config, compute, VpcId)
    #
    # # create nat-gateway for router
    # with cli_logger.group(
    #         "Creating NAT for router",
    #         _numbered=("[]", current_step, total_steps)):
    #     current_step += 1
    #     _create_nat_for_router(config, compute)
    #
    # # create firewalls
    # with cli_logger.group(
    #         "Creating firewall rules",
    #         _numbered=("[]", current_step, total_steps)):
    #     current_step += 1
    #     _create_firewalls(config, compute, VpcId)

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
    logger.info("Using subscription id: %s", subscription_id)
    return network_client

