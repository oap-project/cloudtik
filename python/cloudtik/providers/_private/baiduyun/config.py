import copy
import json
import logging
import time
import uuid
import subprocess
from pathlib import Path
import random

from typing import Any, Dict, Optional

from baidubce.auth.bce_credentials import BceCredentials
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.services.subnet import subnet_client
from baidubce.services.vpc import vpc_client


from cloudtik.core.tags import CLOUDTIK_TAG_NODE_KIND, NODE_KIND_HEAD, CLOUDTIK_TAG_CLUSTER_NAME
from cloudtik.core._private.cli_logger import cli_logger, cf
from cloudtik.core._private.utils import check_cidr_conflict, is_use_internal_ip, _is_use_working_vpc, is_use_working_vpc, is_use_peering_vpc, \
    is_managed_cloud_storage, is_use_managed_cloud_storage, _is_use_managed_cloud_storage, update_nested_dict
from cloudtik.core.workspace_provider import Existence, CLOUDTIK_MANAGED_CLOUD_STORAGE, \
    CLOUDTIK_MANAGED_CLOUD_STORAGE_URI

BCE_RESOURCE_NAME_PREFIX = "cloudtik"

BCE_WORKSPACE_SUBNET_NAME = BCE_RESOURCE_NAME_PREFIX + "-{}-{}-subnet"
BCE_WORKSPACE_VNET_PEERING_NAME = BCE_RESOURCE_NAME_PREFIX + "-{}-virtual-network-peering"
BCE_WORKSPACE_STORAGE_ACCOUNT_NAME = BCE_RESOURCE_NAME_PREFIX + "-{}-storage-account"
BCE_WORKSPACE_STORAGE_CONTAINER_NAME = BCE_RESOURCE_NAME_PREFIX + "-{}"
BCE_WORKSPACE_NETWORK_SECURITY_GROUP_NAME = BCE_RESOURCE_NAME_PREFIX + "-{}-network-security-group"
BCE_WORKSPACE_PUBLIC_IP_ADDRESS_NAME = BCE_RESOURCE_NAME_PREFIX + "-{}-public-ip-address"
BCE_WORKSPACE_NAT_NAME = BCE_RESOURCE_NAME_PREFIX + "-{}-nat"
BCE_WORKSPACE_SECURITY_RULE_NAME = BCE_RESOURCE_NAME_PREFIX + "-{}-security-rule-{}"
BCE_WORKSPACE_WORKER_USI_NAME = BCE_RESOURCE_NAME_PREFIX + "-{}-worker-user-assigned-identity"
BCE_WORKSPACE_HEAD_USI_NAME = BCE_RESOURCE_NAME_PREFIX + "-{}-user-assigned-identity"

BCE_WORKSPACE_VERSION_TAG_NAME = "cloudtik-workspace-version"
BCE_WORKSPACE_VERSION_CURRENT = "1"

BCE_WORKSPACE_NUM_CREATION_STEPS = 9
BCE_WORKSPACE_NUM_DELETION_STEPS = 9
BCE_WORKSPACE_TARGET_RESOURCES = 12

BCE_WORKSPACE_VPC_NAME = BCE_RESOURCE_NAME_PREFIX + "-{}-vpc"

BCE_VPC_SUBNETS_COUNT = 2

logger = logging.getLogger(__name__)


def create_baiduyun_workspace(config):
    return config


def delete_baiduyun_workspace(config, delete_managed_storage: bool = False):
    pass


def check_baiduyun_workspace_integrity(config):
    # existence = check_azure_workspace_existence(config)
    # return True if existence == Existence.COMPLETED else False
    pass


def update_baiduyun_workspace_firewalls(config):
    pass


def get_workspace_head_nodes(provider_config, workspace_name):
    pass


def list_baiduyun_clusters(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    pass


def bootstrap_baiduyun_workspace(config):
    pass


def check_baiduyun_workspace_existence(config):
    pass


def get_baiduyun_workspace_info(config):
    pass


def _get_workspace_vpc_name(workspace_name):
    return BCE_WORKSPACE_VPC_NAME.format(workspace_name)


def _create_vpc(config, vpc_cli):
    workspace_name = config["workspace_name"]
    vpc_name = _get_workspace_vpc_name(workspace_name)
    cli_logger.print("Creating workspace VPC: {}...", vpc_name)
    # create vpc
    cidr_block = '10.0.0.0/16'
    if is_use_peering_vpc(config):
        # TODO
        return
        # current_vpc = get_current_vpc(config)
        # cidr_block = _configure_peering_vpc_cidr_block(current_vpc)

    try:
        response = vpc_cli.create_vpc(vpc_name, cidr_block)
        cli_logger.print("Successfully created workspace VPC: {}.", vpc_name)
        return response.vpc_id
    except Exception as e:
        cli_logger.error("Failed to create workspace VPC. {}", str(e))
        raise e


def _delete_vpc(config, vpc_cli):
    use_working_vpc = is_use_working_vpc(config)
    if use_working_vpc:
        cli_logger.print("Will not delete the current working VPC.")
        return

    vpc_id = get_workspace_vpc_id(config, vpc_cli)
    vpc_name = _get_workspace_vpc_name(config["workspace_name"])

    if vpc_id is None:
        cli_logger.print("The VPC: {} doesn't exist.".format(vpc_name))
        return

    """ Delete the VPC """
    cli_logger.print("Deleting the VPC: {}...".format(vpc_name))

    try:
        vpc_cli.delete_vpc(vpc_id)
        cli_logger.print("Successfully deleted the VPC: {}.".format(vpc_name))
    except Exception as e:
        cli_logger.error("Failed to delete the VPC: {}. {}", vpc_name, str(e))
        raise e


def get_workspace_vpc_id(config, vpc_cli):
    return _get_workspace_vpc_id(config["workspace_name"], vpc_cli)


def _get_workspace_vpc_id(workspace_name, vpc_cli):
    vpc_name = _get_workspace_vpc_name(workspace_name)
    cli_logger.verbose("Getting the VPC Id for workspace: {}...".format(vpc_name))
    vpc_ids = [vpc.vpc_id for vpc in vpc_cli.list_vpcs().vpcs if vpc.name == vpc_name]
    if len(vpc_ids) == 0:
        cli_logger.verbose("The VPC for workspace is not found: {}.".format(vpc_name))
        return None
    else:
        cli_logger.verbose_error("Successfully get the VPC Id of {} for workspace.".format(vpc_name))
        return vpc_ids[0]


def get_vpc(vpc_cli, vpc_id):
    return vpc_cli.get_vpc(vpc_id).vpc


def _create_and_configure_subnets(config, vpc_cli, subnet_cli):
    workspace_name = config["workspace_name"]
    vpc_id = _get_workspace_vpc_id(workspace_name, vpc_cli)
    vpc = get_vpc(vpc_cli, vpc_id)

    subnets = []
    cidr_list = _configure_subnets_cidr(vpc)
    cidr_len = len(cidr_list)

    availability_zones = set(_get_availability_zones(subnet_cli))
    used_availability_zones = set()
    default_availability_zone = list(availability_zones)[0]
    last_availability_zone = None

    for i in range(0, cidr_len):
        cidr_block = cidr_list[i]
        subnet_type = "public" if i == 0 else "private"
        with cli_logger.group(
                "Creating {} subnet", subnet_type,
                _numbered=("()", i + 1, cidr_len)):
            try:
                if i == 0:
                    subnet = _create_subnet(subnet_cli, default_availability_zone, workspace_name, vpc_id, cidr_block, isPrivate=False)
                else:
                    if last_availability_zone is None:
                        last_availability_zone = default_availability_zone

                    subnet = _create_subnet(subnet_cli, last_availability_zone, workspace_name, vpc_id, cidr_block)

                    last_availability_zone = _next_availability_zone(
                        availability_zones, used_availability_zones, last_availability_zone)

            except Exception as e:
                cli_logger.error("Failed to create {} subnet. {}", subnet_type, str(e))
                raise e
            subnets.append(subnet)

    assert len(subnets) == BCE_VPC_SUBNETS_COUNT, "We must create {} subnets for VPC: {}!".format(
        BCE_VPC_SUBNETS_COUNT, vpc_id)
    return subnets


def _delete_private_subnets(workspace_name, vpc_id, subnet_cli):
    _delete_subnets(workspace_name, vpc_id, subnet_cli, isPrivate=True)


def _delete_public_subnets(workspace_name, vpc_id, subnet_cli):
    _delete_subnets(workspace_name, vpc_id, subnet_cli, isPrivate=False)


def _delete_subnets(workspace_name, vpc_id, subnet_cli, isPrivate=True):
    subnetsType = "private" if isPrivate else "public"
    """ Delete custom subnets """
    subnets =  get_workspace_private_subnets(workspace_name, vpc_id, subnet_cli) \
        if isPrivate else get_workspace_public_subnets(workspace_name, vpc_id, subnet_cli)

    if len(subnets) == 0:
        cli_logger.print("No subnets for workspace were found under this VPC: {}...".format(vpc_id))
        return
    try:
        for subnet in subnets:
            cli_logger.print("Deleting {} subnet: {}...".format(subnetsType, subnet.subnet_id))
            subnet.delete()
            cli_logger.print("Successfully deleted {} subnet: {}.".format(subnetsType, subnet.subnet_id))
    except Exception as e:
        cli_logger.error("Failed to delete {} subnet. {}".format(subnetsType, str(e)))
        raise e


def get_workspace_private_subnets(workspace_name, vpc_id, subnet_cli):
    return _get_workspace_subnets(workspace_name, vpc_id, subnet_cli, "cloudtik-{}-private-subnet")


def get_workspace_public_subnets(workspace_name, vpc_id, subnet_cli):
    return _get_workspace_subnets(workspace_name, vpc_id, subnet_cli, "cloudtik-{}-public-subnet")


def _get_workspace_subnets(workspace_name, vpc_id, subnet_cli, name_pattern):
    subnets = [subnet for subnet in subnet_cli.list_subnets(vpc_id=vpc_id).subnets
               if subnet.name.startswith(name_pattern.format(workspace_name))]
    return subnets


def _next_availability_zone(availability_zones: set, used: set, last_availability_zone):
    used.add(last_availability_zone)
    unused = availability_zones.difference(used)
    if len(unused) > 0:
        return unused.pop()

    # Used all, restart
    used.clear()
    if len(availability_zones) > 0:
        return next(iter(availability_zones))

    return None


def _create_public_subnet(subnet_cli, zone_name, workspace_name, vpc_id, cidr_block):
    cli_logger.print("Creating public subnet for VPC: {} with CIDR: {}...".format(vpc_id, cidr_block))
    subnet_name = 'cloudtik-{}-public-subnet'.format(workspace_name)

    response = subnet_cli.create_subnet(name=subnet_name, zone_name=zone_name, cidr=cidr_block, vpc_id=vpc_id)
    cli_logger.print("Successfully created public subnet: {}.".format(subnet_name))

    return response.subnet


def _create_subnet(subnet_cli, zone_name, workspace_name, vpc_id, cidr_block, isPrivate=True):
    subnetType = "private" if isPrivate else "public"
    cli_logger.print("Creating {} subnet for VPC: {} with CIDR: {}...".format(subnetType, vpc_id, cidr_block))
    subnet_name = 'cloudtik-{}-{}-subnet'.format(workspace_name, subnetType)
    response = subnet_cli.create_subnet(name=subnet_name, zone_name=zone_name, cidr=cidr_block, vpc_id=vpc_id)
    cli_logger.print("Successfully created {} subnet: {}.".format(subnetType, subnet_name))

    return response.subnet


def _get_availability_zones(vpc_cli, subnet_cli):
    default_vpc = vpc_cli.list_vpcs(isDefault=True).vpcs[0]
    availability_zones = [subnet.zone_name for subnet in subnet_cli.list_subnets(vpc_id=default_vpc.vpc_id).subnets]
    return availability_zones


def _configure_subnets_cidr(vpc):
    cidr_list = []
    subnets = vpc.subnets
    vpc_cidr = vpc.cidr
    ip = vpc_cidr.split("/")[0].split(".")

    if len(subnets) == 0:
        for i in range(0, BCE_VPC_SUBNETS_COUNT):
            cidr_list.append(ip[0] + "." + ip[1] + "." + str(i) + ".0/24")
    else:
        cidr_blocks = [subnet.cidr for subnet in subnets]
        for i in range(0, 256):
            tmp_cidr_block = ip[0] + "." + ip[1] + "." + str(i) + ".0/24"

            if check_cidr_conflict(tmp_cidr_block, cidr_blocks):
                cidr_list.append(tmp_cidr_block)

            if len(cidr_list) == BCE_VPC_SUBNETS_COUNT:
                break

    return cidr_list


def _get_bce_credentials(provider_config):
    access_key = provider_config.get("access_key")
    access_key_secret = provider_config.get("access_key_secret")
    return BceCredentials(access_key, access_key_secret)


def _create_vpc_client(bce_credentials, endpoint):
    config = BceClientConfiguration(credentials=bce_credentials, endpoint=endpoint)
    vpc_cli = vpc_client.VpcClient(config)
    return vpc_cli


def _create_subnet_client(bce_credentials, endpoint):
    config = BceClientConfiguration(credentials=bce_credentials, endpoint=endpoint)
    subnet_cli = subnet_client.SubnetClient(config)
    return subnet_cli


def construct_vpc_client(provider_config):
    credentials = _get_bce_credentials(provider_config)
    endpoint = _get_vpc_endpoint(provider_config["region"])
    return _create_vpc_client(credentials, endpoint)


def construct_subnet_client(provider_config):
    credentials = _get_bce_credentials(provider_config)
    endpoint = _get_vpc_endpoint(provider_config["region"])
    return _create_subnet_client(credentials, endpoint)


def check_bce_region(region):
    bce_available_regions = ["bj", "bd", "su", "gz", "hkg", "sin", "fwh", "fsh"]
    if region not in bce_available_regions:
        cli_logger.abort(
            "Unknown region " + cf.bold("{}") + "\n"
             "Available regions are: {}", region, cli_logger.render_list(bce_available_regions))


def _get_bos_endpoint(region):
    check_bce_region(region)
    return f"https://{region}.bcebos.com"


def _get_vpc_endpoint(region):
    check_bce_region(region)
    return f"https://bcc.{region}.baidubce.com"
