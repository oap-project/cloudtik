from distutils.version import StrictVersion
from functools import lru_cache
from functools import partial
import copy
import itertools
import json
import os
import time
from typing import Any, Dict, List, Optional
import logging

import boto3
import botocore

from cloudtik.core.tags import CLOUDTIK_TAG_NODE_KIND, NODE_KIND_HEAD
from cloudtik.core._private.providers import _PROVIDER_PRETTY_NAMES
from cloudtik.core._private.cli_logger import cli_logger, cf
from cloudtik.core._private.event_system import (CreateClusterEvent,
                                                  global_event_system)
from cloudtik.core._private.services import get_node_ip_address
from cloudtik.core._private.utils import check_cidr_conflict
from cloudtik.providers._private.aws.utils import LazyDefaultDict, \
    handle_boto_error, resource_cache, get_boto_error_code
from cloudtik.providers._private.utils import StorageTestingError

logger = logging.getLogger(__name__)

AWS_RESOURCE_NAME_PREFIX = "cloudtik"
AWS_DEFAULT_INSTANCE_PROFILE = AWS_RESOURCE_NAME_PREFIX + "-v1"
AWS_DEFAULT_IAM_ROLE = AWS_RESOURCE_NAME_PREFIX + "-v1"

SECURITY_GROUP_TEMPLATE = AWS_RESOURCE_NAME_PREFIX + "-{}"

DEFAULT_AMI_NAME = "Ubuntu Server 20.04 LTS (HVM), SSD Volume Type"

# Obtained from https://aws.amazon.com/marketplace/pp/B07Y43P7X5 on 8/4/2020.
DEFAULT_AMI = {
    "us-east-1": "ami-04505e74c0741db8d",  # US East (N. Virginia)
    "us-east-2": "ami-0fb653ca2d3203ac1",  # US East (Ohio)
    "us-west-1": "ami-01f87c43e618bf8f0",  # US West (N. California)
    "us-west-2": "ami-0892d3c7ee96c0bf7",  # US West (Oregon)
    "af-souce-1": "ami-030b8d2037063bab3", # Africa (Cape Town)
    "ap-east-1": "ami-0b981d9ee99b28eba", # Asia Pacific (Hong Kong)
    "ap-south-1": "ami-0851b76e8b1bce90b", # # Asia Pacific (Mumbai)
    "ap-northeast-1": "ami-088da9557aae42f39", # Asia Pacific (Tokyo)
    "ap-northeast-2": "ami-0454bb2fefc7de534", # Asia Pacific (Seoul),
    "ap-northeast-3": "ami-096c4b6e0792d8c16", # Asia Pacific (Osaka),
    "ap-southeast-1": "ami-055d15d9cfddf7bd3", # Asia Pacific (Singapore)
    "ap-southeast-2": "ami-0b7dcd6e6fd797935", # Asia Pacific (Sydney),
    "ap-southeast-3": "ami-0a9c8e0ccf1d85f67", # Asia Pacific (Jakarta)
    "ca-central-1": "ami-0aee2d0182c9054ac",  # Canada (Central)
    "eu-central-1": "ami-0d527b8c289b4af7f",  # EU (Frankfurt)
    "eu-west-1": "ami-08ca3fed11864d6bb",  # EU (Ireland)
    "eu-west-2": "ami-0015a39e4b7c0966f",  # EU (London)
    "eu-west-3": "ami-0c6ebbd55ab05f070",  # EU (Paris)
    "eu-south-1": "ami-0f8ce9c417115413d",  # EU (Milan)
    "eu-north-1": "ami-092cce4a19b438926",  # EU (Stockholm)
    "me-south-1": "ami-0b4946d7420c44be4",  # Middle East (Bahrain)
    "sa-east-1": "ami-090006f29ecb2d79a",  # SA (Sao Paulo)
}

NUM_AWS_WORKSPACE_CREATION_STEPS = 7
NUM_AWS_WORKSPACE_DELETION_STEPS = 7

# todo: cli_logger should handle this assert properly
# this should probably also happens somewhere else
assert StrictVersion(boto3.__version__) >= StrictVersion("1.4.8"), \
    "Boto3 version >= 1.4.8 required, try `pip install -U boto3`"


def key_pair(i, region, key_name):
    """
    If key_name is not None, key_pair will be named after key_name.
    Returns the ith default (aws_key_pair_name, key_pair_path).
    """
    if i == 0:
        key_pair_name = ("{}_{}".format(AWS_RESOURCE_NAME_PREFIX, region)
                         if key_name is None else key_name)
        return (key_pair_name,
                os.path.expanduser("~/.ssh/{}.pem".format(key_pair_name)))

    key_pair_name = ("{}_{}_{}".format(AWS_RESOURCE_NAME_PREFIX, i, region)
                     if key_name is None else key_name + "_key-{}".format(i))
    return (key_pair_name,
            os.path.expanduser("~/.ssh/{}.pem".format(key_pair_name)))


# Suppress excessive connection dropped logs from boto
logging.getLogger("botocore").setLevel(logging.WARNING)

_log_info = {}


def reload_log_state(override_log_info):
    _log_info.update(override_log_info)


def get_log_state():
    return _log_info.copy()


def _set_config_info(**kwargs):
    """Record configuration artifacts useful for logging."""

    # todo: this is technically fragile iff we ever use multiple configs

    for k, v in kwargs.items():
        _log_info[k] = v


def _arn_to_name(arn):
    return arn.split(":")[-1].split("/")[-1]


def log_to_cli(config: Dict[str, Any]) -> None:
    provider_name = _PROVIDER_PRETTY_NAMES.get("aws", None)

    cli_logger.doassert(provider_name is not None,
                        "Could not find a pretty name for the AWS provider.")

    head_node_type = config["head_node_type"]
    head_node_config = config["available_node_types"][head_node_type][
        "node_config"]

    with cli_logger.group("{} config", provider_name):

        def print_info(resource_string: str,
                       key: str,
                       src_key: str,
                       allowed_tags: Optional[List[str]] = None,
                       list_value: bool = False) -> None:
            if allowed_tags is None:
                allowed_tags = ["default"]

            node_tags = {}

            # set of configurations corresponding to `key`
            unique_settings = set()

            for node_type_key, node_type in config[
                    "available_node_types"].items():
                node_tags[node_type_key] = {}
                tag = _log_info[src_key][node_type_key]
                if tag in allowed_tags:
                    node_tags[node_type_key][tag] = True
                setting = node_type["node_config"].get(key)

                if list_value:
                    unique_settings.add(tuple(setting))
                else:
                    unique_settings.add(setting)

            head_value_str = head_node_config[key]
            if list_value:
                head_value_str = cli_logger.render_list(head_value_str)

            if len(unique_settings) == 1:
                # all node types are configured the same, condense
                # log output
                cli_logger.labeled_value(
                    resource_string + " (all available node types)",
                    "{}",
                    head_value_str,
                    _tags=node_tags[config["head_node_type"]])
            else:
                # do head node type first
                cli_logger.labeled_value(
                    resource_string + f" ({head_node_type})",
                    "{}",
                    head_value_str,
                    _tags=node_tags[head_node_type])

                # go through remaining types
                for node_type_key, node_type in config[
                        "available_node_types"].items():
                    if node_type_key == head_node_type:
                        continue
                    workers_value_str = node_type["node_config"][key]
                    if list_value:
                        workers_value_str = cli_logger.render_list(
                            workers_value_str)
                    cli_logger.labeled_value(
                        resource_string + f" ({node_type_key})",
                        "{}",
                        workers_value_str,
                        _tags=node_tags[node_type_key])

        tags = {"default": _log_info["head_instance_profile_src"] == "default"}
        # head_node_config is the head_node_type's config,
        # config["head_node"] is a field that gets applied only to the actual
        # head node (and not workers of the head's node_type)
        assert ("IamInstanceProfile" in head_node_config
                or "IamInstanceProfile" in config["head_node"])
        if "IamInstanceProfile" in head_node_config:
            # If the user manually configured the role we're here.
            IamProfile = head_node_config["IamInstanceProfile"]
        elif "IamInstanceProfile" in config["head_node"]:
            # If we filled the default IAM role, we're here.
            IamProfile = config["head_node"]["IamInstanceProfile"]
        profile_arn = IamProfile.get("Arn")
        profile_name = _arn_to_name(profile_arn) \
            if profile_arn \
            else IamProfile["Name"]
        cli_logger.labeled_value("IAM Profile", "{}", profile_name, _tags=tags)

        if all("KeyName" in node_type["node_config"]
               for node_type in config["available_node_types"].values()):
            print_info("EC2 Key pair", "KeyName", "keypair_src")

        print_info("VPC Subnets", "SubnetIds", "subnet_src", list_value=True)
        print_info(
            "EC2 Security groups",
            "SecurityGroupIds",
            "security_group_src",
            list_value=True)
        print_info("EC2 AMI", "ImageId", "ami_src", allowed_tags=["dlami"])

    cli_logger.newline()


def get_workspace_vpc_id(workspace_name, ec2_client):
    vpcs = [vpc for vpc in ec2_client.describe_vpcs()["Vpcs"] if not vpc.get("Tags") is None]
    VpcIds = [vpc["VpcId"] for vpc in vpcs
             for tag in vpc["Tags"]
                if tag['Key'] == 'Name' and tag['Value'] == 'cloudtik-{}-vpc'.format(workspace_name)]

    if len(VpcIds) == 0:
        return None
    elif len(VpcIds) == 1:
        return VpcIds[0]
    else:
        cli_logger.abort("This workspace: {} should not contain More than one VPC!!".format(workspace_name))
        return None


def get_workspace_private_subnets(workspace_name, ec2, VpcId):
    vpc_resource = ec2.Vpc(VpcId)
    subnets = [subnet for subnet in vpc_resource.subnets.all() if subnet.tags]

    workspace_private_subnets = [subnet for subnet in subnets
                        for tag in subnet.tags
                            if tag['Key'] == 'Name' and tag['Value'].startswith(
                        "cloudtik-{}-private-subnet".format(workspace_name))]

    return workspace_private_subnets


def get_workspace_public_subnets(workspace_name, ec2, VpcId):
    vpc_resource = ec2.Vpc(VpcId)
    subnets = [subnet for subnet in vpc_resource.subnets.all() if subnet.tags]

    workspace_public_subnets = [subnet for subnet in subnets
                                for tag in subnet.tags
                                    if tag['Key'] == 'Name' and tag['Value'].startswith(
            "cloudtik-{}-public-subnet".format(workspace_name))]

    return workspace_public_subnets


def get_workspace_nat_gateways(workspace_name, ec2_client, VpcId):
    nat_gateways = [nat for nat in ec2_client.describe_nat_gateways()['NatGateways']
                    if nat["VpcId"] == VpcId and nat["State"] != 'deleted'
                    and not nat.get("Tags") is None]

    workspace_nat_gateways = [nat for nat in nat_gateways
                              for tag in nat['Tags']
                                if tag['Key'] == 'Name' and tag['Value'] == "cloudtik-{}-nat-gateway".format(
                                    workspace_name)]

    return workspace_nat_gateways


def get_vpc_nat_gateways(ec2_client, VpcId):
    nat_gateways = [nat for nat in ec2_client.describe_nat_gateways()['NatGateways']
                    if nat["VpcId"] == VpcId and nat["State"] != 'deleted']

    return nat_gateways


def get_workspace_private_route_tables(workspace_name, ec2, VpcId):
    vpc_resource = ec2.Vpc(VpcId)
    rtbs = [rtb for rtb in vpc_resource.route_tables.all() if rtb.tags]

    workspace_private_rtbs = [rtb for rtb in rtbs
                      for tag in rtb.tags
                        if tag['Key'] == 'Name' and tag['Value'] == "cloudtik-{}-private-route-table".format(
                         workspace_name)]

    return workspace_private_rtbs


def get_workspace_public_route_tables(workspace_name, ec2, VpcId):
    vpc_resource = ec2.Vpc(VpcId)
    rtbs = [rtb for rtb in vpc_resource.route_tables.all() if rtb.tags]

    workspace_public_rtbs = [rtb for rtb in rtbs
                      for tag in rtb.tags
                        if tag['Key'] == 'Name' and tag['Value'] == "cloudtik-{}-public-route-table".format(
                         workspace_name)]

    return workspace_public_rtbs


def get_workspace_security_group(config, VpcId, workspace_name):
    return _get_security_group(config, VpcId, SECURITY_GROUP_TEMPLATE.format(workspace_name))


def get_workspace_internat_gateways(workspace_name, ec2, VpcId):
    vpc_resource = ec2.Vpc(VpcId)
    igws = [igw for igw in vpc_resource.internet_gateways.all() if igw.tags]

    workspace_igws = [igw for igw in igws
                      for tag in igw.tags
                        if tag['Key'] == 'Name' and tag['Value'] == "cloudtik-{}-internet-gateway".format(
                            workspace_name)]

    return workspace_igws


def get_vpc_internat_gateways(ec2, VpcId):
    vpc_resource = ec2.Vpc(VpcId)
    igws = list(vpc_resource.internet_gateways.all())

    return igws


def check_aws_workspace_resource(config):
    ec2 = _resource("ec2", config)
    ec2_client = _client("ec2", config)
    workspace_name = config["workspace_name"]
    VpcId = get_workspace_vpc_id(workspace_name, ec2_client)
    """
         Do the work - order of operation
         1.) Check VPC 
         2.) Check private subnets
         3.) Check public subnets
         4.) Check nat-gatways
         5.) Check route-tables
         6.) Check Internat-gateways
         7.) Check security-group
    """
    if VpcId is None:
        return False
    if len(get_workspace_private_subnets(workspace_name, ec2, VpcId)) == 0:
        return False
    if len(get_workspace_public_subnets(workspace_name, ec2, VpcId)) == 0:
        return False
    if len(get_vpc_nat_gateways(ec2_client, VpcId)) == 0:
        return False
    if len(get_workspace_private_route_tables(workspace_name, ec2, VpcId)) == 0:
        return False
    if len(get_vpc_internat_gateways(ec2, VpcId)) == 0:
        return False
    if get_workspace_security_group(config, VpcId, workspace_name) is None:
        return False
    return True


def update_aws_workspace_firewalls(config):
    ec2_client = _client("ec2", config)
    workspace_name = config["workspace_name"]
    vpcid = get_workspace_vpc_id(workspace_name, ec2_client)
    if vpcid is None:
        cli_logger.print("The workspace: {} doesn't exist!".format(config["workspace_name"]))
        return

    current_step = 1
    total_steps = 1

    try:

        with cli_logger.group(
                "Updating workspace firewalls",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _update_security_group(config, vpcid)

    except Exception as e:
        cli_logger.error(
            "Failed to update the firewalls of workspace {}. {}".format(workspace_name, str(e)))
        raise e

    cli_logger.print(
        "Successfully updated the firewalls of workspace: {}.",
        cf.bold(workspace_name))
    return None


def delete_workspace_aws(config):
    ec2 = _resource("ec2", config)
    ec2_client = _client("ec2", config)
    workspace_name = config["workspace_name"]
    use_internal_ips = config["provider"].get("use_internal_ips", False)
    vpcid = get_workspace_vpc_id(workspace_name, ec2_client)
    if vpcid is None:
        cli_logger.print("The workspace: {} doesn't exist!".format(config["workspace_name"]))
        return

    current_step = 1
    total_steps = NUM_AWS_WORKSPACE_DELETION_STEPS
    if not use_internal_ips:
        total_steps += 1

    try:

        with cli_logger.group("Deleting workspace: {}", workspace_name):
            with cli_logger.group(
                    "Deleting instance profile",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _delete_workspace_instance_profile(config, workspace_name)

            _delete_network_resources(config, workspace_name,
                                      ec2, ec2_client, vpcid,
                                      current_step, total_steps)

    except Exception as e:
        cli_logger.error(
            "Failed to delete workspace {}. {}".format(workspace_name, str(e)))
        raise e

    cli_logger.print(
            "Successfully deleted workspace: {}.",
            cf.bold(workspace_name))
    return None


def _delete_workspace_instance_profile(config, workspace_name):
    instance_profile_name = _get_workspace_instance_profile_name(workspace_name)
    instance_role_name = "cloudtik-{}-role".format(workspace_name)
    _delete_instance_profile(config, instance_profile_name, instance_role_name)


def _delete_network_resources(config, workspace_name,
                              ec2, ec2_client, vpcid,
                              current_step, total_steps):
    use_internal_ips = config["provider"].get("use_internal_ips", False)

    """
         Do the work - order of operation
         1.) Delete private subnets 
         2.) Delete route-tables for private subnets 
         3.) Delete nat-gateway for private subnets
         4.) Delete public subnets
         5.) Delete internet gateway
         6.) Delete security group
         7.) Delete vpc
    """

    # delete private subnets
    with cli_logger.group(
            "Deleting private subnet",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_private_subnets(workspace_name, ec2, vpcid)

    # delete route tables for private subnets
    with cli_logger.group(
            "Deleting route table",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_route_table(workspace_name, ec2, vpcid)

    # delete nat-gateway
    with cli_logger.group(
            "Deleting NAT gateway",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_nat_gateway(workspace_name, ec2_client, vpcid)

    # delete public subnets
    with cli_logger.group(
            "Deleting public subnet",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_public_subnets(workspace_name, ec2, vpcid)

    # delete internet gateway
    with cli_logger.group(
            "Deleting Internet gateway",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_internet_gateway(workspace_name, ec2, vpcid)

    # delete security group
    with cli_logger.group(
            "Deleting security group",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_security_group(config, vpcid)

    # delete vpc
    if not use_internal_ips:
        with cli_logger.group(
                "Deleting VPC",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _delete_vpc(ec2, ec2_client, vpcid)


def create_aws_workspace(config):
    # create a copy of the input config to modify
    config = copy.deepcopy(config)

    # create workspace
    config = _configure_workspace(config)

    return config


def _configure_spot_for_node_type(node_type_config,
                                  prefer_spot_node):
    # To be improved if scheduling has other configurations
    # InstanceMarketOptions:
    #   MarketType: spot
    node_config = node_type_config["node_config"]
    if prefer_spot_node:
        # Add spot instruction
        node_config.pop("InstanceMarketOptions", None)
        node_config["InstanceMarketOptions"] = {"MarketType": "spot"}
    else:
        # Remove spot instruction
        node_config.pop("InstanceMarketOptions", None)


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


def bootstrap_aws(config):
    workspace_name = config.get("workspace_name", "")
    if workspace_name == "":
        config = bootstrap_aws_default(config)
    else:
        config = bootstrap_aws_from_workspace(config)

    _configure_prefer_spot_node(config)
    return config


def bootstrap_aws_default(config):
    # create a copy of the input config to modify
    config = copy.deepcopy(config)

    # Used internally to store head IAM role.
    config["head_node"] = {}

    # If a LaunchTemplate is provided, extract the necessary fields for the
    # config stages below.
    config = _configure_from_launch_template(config)

    # If NetworkInterfaces are provided, extract the necessary fields for the
    # config stages below.
    config = _configure_from_network_interfaces(config)

    # The head node needs to have an IAM role that allows it to create further
    # EC2 instances.
    config = _configure_iam_role(config)

    # Configure SSH access, using an existing key pair if possible.
    config = _configure_key_pair(config)
    global_event_system.execute_callback(
        CreateClusterEvent.ssh_keypair_downloaded,
        {"ssh_key_path": config["auth"]["ssh_private_key"]})

    # Pick a reasonable subnet if not specified by the user.
    config = _configure_subnet(config)

    # Cluster workers should be in a security group that permits traffic within
    # the group, and also SSH access from outside.
    config = _configure_security_group(config)

    # Provide a helpful message for missing AMI.
    _configure_ami(config)

    return config


def bootstrap_aws_from_workspace(config):
    if not check_aws_workspace_resource(config):
        workspace_name = config["workspace_name"]
        cli_logger.abort("AWS workspace {} doesn't exist or is in wrong state!", workspace_name)

    # create a copy of the input config to modify
    config = copy.deepcopy(config)

    # Used internally to store head IAM role.
    config["head_node"] = {}

    # The head node needs to have an IAM role that allows it to create further
    # EC2 instances.
    config = _configure_iam_role_from_workspace(config)

    # Configure SSH access, using an existing key pair if possible.
    config = _configure_key_pair(config)
    global_event_system.execute_callback(
        CreateClusterEvent.ssh_keypair_downloaded,
        {"ssh_key_path": config["auth"]["ssh_private_key"]})

    # Pick a reasonable subnet if not specified by the user.
    config = _configure_subnet_from_workspace(config)

    # Cluster workers should be in a security group that permits traffic within
    # the group, and also SSH access from outside.
    config = _configure_security_group_from_workspace(config)

    # Provide a helpful message for missing AMI.
    config = _configure_ami(config)

    return config


def get_workspace_head_nodes(cluster_config):
    ec2 = _resource("ec2", cluster_config)
    ec2_client = _client("ec2", cluster_config)
    workspace_name = cluster_config["workspace_name"]
    VpcId = get_workspace_vpc_id(workspace_name, ec2_client)

    filters = [
        {
           "Name": "vpc-id",
            "Values": [VpcId]
        },
        {
            "Name": "instance-state-name",
            "Values": ["running"],
        },
        {
            "Name": "tag:{}".format(CLOUDTIK_TAG_NODE_KIND),
            "Values": [NODE_KIND_HEAD],
        },
    ]

    nodes = list(ec2.instances.filter(Filters=filters))
    return nodes


def _configure_iam_role(config):
    head_node_type = config["head_node_type"]
    head_node_config = config["available_node_types"][head_node_type][
        "node_config"]
    if "IamInstanceProfile" in head_node_config:
        _set_config_info(head_instance_profile_src="config")
        return config
    _set_config_info(head_instance_profile_src="default")

    profile = _create_or_update_instance_profile(config,
                                                 AWS_DEFAULT_INSTANCE_PROFILE,
                                                 AWS_DEFAULT_IAM_ROLE)
    # Add IAM role to "head_node" field so that it is applied only to
    # the head node -- not to workers with the same node type as the head.
    config["head_node"]["IamInstanceProfile"] = {"Arn": profile.arn}

    return config


def _configure_iam_role_from_workspace(config):
    head_node_type = config["head_node_type"]
    head_node_config = config["available_node_types"][head_node_type][
        "node_config"]
    if "IamInstanceProfile" in head_node_config:
        _set_config_info(head_instance_profile_src="config")
        return config
    _set_config_info(head_instance_profile_src="workspace")

    instance_profile_name = _get_workspace_instance_profile_name(
        config["workspace_name"])
    profile = _get_instance_profile(instance_profile_name, config)

    if not profile:
        raise RuntimeError("Workspace instance profile: {} not found!".format(instance_profile_name))

    # Add IAM role to "head_node" field so that it is applied only to
    # the head node -- not to workers with the same node type as the head.
    config["head_node"]["IamInstanceProfile"] = {"Arn": profile.arn}

    return config


def _create_or_update_instance_profile(config, instance_profile_name, instance_role_name):
    profile = _get_instance_profile(instance_profile_name, config)

    if profile is None:
        cli_logger.verbose(
            "Creating new IAM instance profile {} for use as the default.",
            cf.bold(instance_profile_name))
        client = _client("iam", config)
        client.create_instance_profile(
            InstanceProfileName=instance_profile_name)
        profile = _get_instance_profile(instance_profile_name, config)
        time.sleep(15)  # wait for propagation

    cli_logger.doassert(profile is not None,
                        "Failed to create instance profile.")  # todo: err msg
    assert profile is not None, "Failed to create instance profile"

    if not profile.roles:
        role_name = instance_role_name
        role = _get_role(role_name, config)
        if role is None:
            cli_logger.verbose(
                "Creating new IAM role {} for "
                "use as the default instance role.", cf.bold(role_name))
            iam = _resource("iam", config)
            policy_doc = {
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {
                            "Service": "ec2.amazonaws.com"
                        },
                        "Action": "sts:AssumeRole",
                    },
                ]
            }
            attach_policy_arns =  [
                    "arn:aws:iam::aws:policy/AmazonEC2FullAccess",
                    "arn:aws:iam::aws:policy/AmazonS3FullAccess"
                ]

            iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(policy_doc))
            role = _get_role(role_name, config)
            cli_logger.doassert(role is not None,
                                "Failed to create role.")  # todo: err msg

            assert role is not None, "Failed to create role"

            for policy_arn in attach_policy_arns:
                role.attach_policy(PolicyArn=policy_arn)

        profile.add_role(RoleName=role.name)
        time.sleep(15)  # wait for propagation

    return profile


def _delete_instance_profile(config, instance_profile_name, instance_profile_role):
    cli_logger.print("Deleting instance profile: {}...".format(instance_profile_name))

    profile = _get_instance_profile(instance_profile_name, config)
    if profile is None:
        cli_logger.warning("No instance profile with the name found.")
        return

    # Remove all roles from instance profile
    if profile.roles:
        for role in profile.roles:
            profile.remove_role(RoleName=role.name)

    # first delete role and policy
    _delete_instance_profile_role(config, instance_profile_role)

    client = _client("iam", config)
    # Deletes the specified instance profile. The instance profile must not have an associated role.
    client.delete_instance_profile(
        InstanceProfileName=instance_profile_name)

    cli_logger.print("Successfully deleted instance profile.")


def _delete_instance_profile_role(config, instance_profile_role):
    role = _get_role(instance_profile_role, config)
    if role is None:
        cli_logger.print("IAM role {} doesn't exist", instance_profile_role)
        return

    # detach the policies
    for policy in role.attached_policies.all():
        role.detach_policy(PolicyArn=policy.arn)

    # delete the role
    role.delete()


def _configure_key_pair(config):
    node_types = config["available_node_types"]

    # map from node type key -> source of KeyName field
    key_pair_src_info = {}
    _set_config_info(keypair_src=key_pair_src_info)

    if "ssh_private_key" in config["auth"]:
        for node_type_key in node_types:
            # keypairs should be provided in the config
            key_pair_src_info[node_type_key] = "config"

        # If the key is not configured via the cloudinit
        # UserData, it should be configured via KeyName or
        # else we will risk starting a node that we cannot
        # SSH into:

        for node_type in node_types:
            node_config = node_types[node_type]["node_config"]
            if "UserData" not in node_config:
                cli_logger.doassert("KeyName" in node_config,
                                    _key_assert_msg(node_type))
                assert "KeyName" in node_config

        return config

    for node_type_key in node_types:
        key_pair_src_info[node_type_key] = "default"

    ec2 = _resource("ec2", config)

    # Writing the new ssh key to the filesystem fails if the ~/.ssh
    # directory doesn't already exist.
    os.makedirs(os.path.expanduser("~/.ssh"), exist_ok=True)

    # Try a few times to get or create a good key pair.
    MAX_NUM_KEYS = 30
    for i in range(MAX_NUM_KEYS):

        key_name = config["provider"].get("key_pair", {}).get("key_name")

        key_name, key_path = key_pair(i, config["provider"]["region"],
                                      key_name)
        key = _get_key(key_name, config)

        # Found a good key.
        if key and os.path.exists(key_path):
            break

        # We can safely create a new key.
        if not key and not os.path.exists(key_path):
            cli_logger.verbose(
                "Creating new key pair {} for use as the default.",
                cf.bold(key_name))
            key = ec2.create_key_pair(KeyName=key_name)

            # We need to make sure to _create_ the file with the right
            # permissions. In order to do that we need to change the default
            # os.open behavior to include the mode we want.
            with open(key_path, "w", opener=partial(os.open, mode=0o600)) as f:
                f.write(key.key_material)
            break

    if not key:
        cli_logger.abort(
            "No matching local key file for any of the key pairs in this "
            "account with ids from 0..{}. "
            "Consider deleting some unused keys pairs from your account.",
            key_name)

    cli_logger.doassert(
        os.path.exists(key_path), "Private key file " + cf.bold("{}") +
        " not found for " + cf.bold("{}"), key_path, key_name)  # todo: err msg
    assert os.path.exists(key_path), \
        "Private key file {} not found for {}".format(key_path, key_name)

    config["auth"]["ssh_private_key"] = key_path
    for node_type in node_types.values():
        node_config = node_type["node_config"]
        node_config["KeyName"] = key_name

    return config


def _key_assert_msg(node_type: str) -> str:
    return ("`KeyName` missing from the `node_config` of"
            f" node type `{node_type}`.")


def _delete_internet_gateway(workspace_name, ec2, VpcId):
    """ Detach and delete the internet-gateway """
    igws = get_workspace_internat_gateways(workspace_name, ec2, VpcId)

    if len(igws) == 0:
        cli_logger.print("No Internet Gateways for workspace were found under this VPC: {} ...".format(VpcId))
        return
    for igw in igws:
        try:
            cli_logger.print("Detaching and deleting Internet Gateway: {} ...".format(igw.id))
            igw.detach_from_vpc(VpcId=VpcId)
            igw.delete()
        except boto3.exceptions.Boto3Error as e:
            cli_logger.error("Failed to detach or delete Internet Gateway. {}", str(e))
            raise e
    return


def _delete_private_subnets(workspace_name, ec2, VpcId):
    """ Delete custom private subnets """
    subnets = get_workspace_private_subnets(workspace_name, ec2, VpcId)

    if len(subnets) == 0:
        cli_logger.print("No subnets for workspace were found under this VPC: {} ...".format(VpcId))
        return
    try:
        for subnet in subnets:
            cli_logger.print("Deleting private subnet: {} ...".format(subnet.id))
            subnet.delete()
    except boto3.exceptions.Boto3Error as e:
        cli_logger.error("Failed to delete private subnet. {}", str(e))
        raise e
    return


def _delete_public_subnets(workspace_name, ec2, VpcId):
    """ Delete custom public subnets """
    subnets = get_workspace_public_subnets(workspace_name, ec2, VpcId)

    if len(subnets) == 0:
        cli_logger.print("No subnets for workspace were found under this VPC: {} ...".format(VpcId))
        return
    try:
        for subnet in subnets:
            cli_logger.print("Deleting public subnet: {} ...".format(subnet.id))
            subnet.delete()
    except boto3.exceptions.Boto3Error as e:
        cli_logger.error("Failed to delete public subnet. {}", str(e))
        raise e
    return


def _delete_route_table(workspace_name, ec2, VpcId):
    """ Delete the route-tables for private subnets """
    rtbs = get_workspace_private_route_tables(workspace_name, ec2, VpcId)
    if len(rtbs) == 0:
        cli_logger.print("No route tables for workspace were found under this VPC: {} ...".format(VpcId))
        return
    try:
        for rtb in rtbs:
            cli_logger.print("Deleting route table: {} ...".format(rtb.id))
            table = ec2.RouteTable(rtb.id)
            table.delete()
    except boto3.exceptions.Boto3Error as e:
        cli_logger.error("Failed to delete route table. {}", str(e))
        raise e
    return


def release_elastic_ip_address(ec2_client, allocationId, retry=5):
    while retry > 0:
        try:
            ec2_client.release_address(AllocationId=allocationId)
        except botocore.exceptions.ClientError as e:
            error_code = get_boto_error_code(e)
            if error_code and error_code == "InvalidAllocationID.NotFound":
                cli_logger.warning("Warning: {}", str(e))
                return

            retry = retry - 1
            if retry > 0:
                cli_logger.warning("Remaining {} tries to release elastic ip address for NAT Gateway...".format(retry))
                time.sleep(60)
            else:
                cli_logger.error("Failed to release elastic ip address for NAT Gateway. {}", str(e))
                raise e

    cli_logger.print("Successfully released elastic ip address for NAT Gateway.")


def _delete_nat_gateway(workspace_name, ec2_client, VpcId):
    """ Remove nat-gateway and release elastic IP """
    nat_gateways = get_workspace_nat_gateways(workspace_name, ec2_client, VpcId)
    if len(nat_gateways) == 0:
        cli_logger.print("No NAT Gateways for workspace were found under this VPC: {} ...".format(VpcId))
        return
    try:
        for nat in nat_gateways:
            cli_logger.print("Deleting NAT Gateway: {} ...".format(nat["NatGatewayId"]))
            ec2_client.delete_nat_gateway(NatGatewayId=nat["NatGatewayId"])
            cli_logger.print("Deleting elastic instance : {} ...".format(nat["NatGatewayAddresses"][0]["AllocationId"]))
            release_elastic_ip_address(ec2_client, nat["NatGatewayAddresses"][0]["AllocationId"])
    except boto3.exceptions.Boto3Error as e:
        cli_logger.error("Failed to delete NAT Gateway. {}", str(e))
        raise e
    return


def _delete_security_group(config, VpcId):
    """ Delete any security-groups """
    sg = _get_security_group(config, VpcId, SECURITY_GROUP_TEMPLATE.format(config["workspace_name"]))
    if sg is None:
        cli_logger.print("No security groups for workspace were found under this VPC: {} ...".format(VpcId))
        return
    try:
        cli_logger.print("Deleting security group: {}".format(sg.id))
        sg.delete()
    except boto3.exceptions.Boto3Error as e:
        cli_logger.error("Failed to delete security group. {}", str(e))
        raise e
    return


def _delete_vpc(ec2, ec2_client, VpcId):
    """ Delete the VPC """
    VpcIds = [vpc["VpcId"] for vpc in ec2_client.describe_vpcs()["Vpcs"] if vpc["VpcId"] == VpcId]
    if len(VpcIds) == 0:
        cli_logger.print("This VPC: {} doesn't exist. ".format(VpcId))
        return

    vpc_resource = ec2.Vpc(VpcId)
    try:
        cli_logger.print("Deleting VPC: {}".format(vpc_resource.id))
        vpc_resource.delete()
    except Exception as e:
        cli_logger.error("Failed to delete VPC. {}", str(e))
        raise e
    return


def _create_vpc(config,  ec2):
    cli_logger.print("Creating workspace VPC...")
    # create vpc
    try:
        vpc = ec2.create_vpc(CidrBlock='10.10.0.0/16')
        vpc.modify_attribute(EnableDnsSupport={'Value': True})
        vpc.modify_attribute(EnableDnsHostnames={'Value': True})
        vpc.create_tags(Tags=[{'Key': 'Name', 'Value': 'cloudtik-{}-vpc'.format(config["workspace_name"])}])
        cli_logger.print("Successfully created workspace VPC: cloudtik-{}-vpc ...".format(config["workspace_name"]))
    except Exception as e:
        cli_logger.error("Failed to create workspace VPC. {}", str(e))
        raise e

    return vpc


def _create_subnet(config, cidr,  vpc):
    cli_logger.print("Creating subnet for VPC: {} with CIDR: {} ...".format(vpc.id, cidr))
    try:
        subnet = vpc.create_subnet(CidrBlock=cidr)
        cli_logger.print("Successfully created subnet: cloudtik-{}-subnet ...".format(config["workspace_name"]))
    except Exception as e:
        cli_logger.error("Failed to create subnet. {}", str(e))
        raise e
    return subnet


def _create_and_configure_subnets(config, vpc):
    subnets = []
    cidr_list = _configure_subnets_cidr(vpc)
    for i in range(0, len(cidr_list)):
        subnets.append(_create_subnet(config, cidr_list[i], vpc))

    assert len(subnets) == 2, "We must create 2 subnets for VPC: {}!".format(vpc.id)

    for i in range(0, len(cidr_list)):
        # make the first subnet as public subnet and rename it
        if i == 0:
            subnets[0].meta.client.modify_subnet_attribute(SubnetId=subnets[0].id, MapPublicIpOnLaunch={"Value": True})
            subnets[0].create_tags(Tags=[{'Key': 'Name',
                                          'Value': 'cloudtik-{}-public-subnet'.format(config["workspace_name"])}])
        else:
            subnets[i].create_tags(Tags=[{'Key': 'Name',
                                          'Value': 'cloudtik-{}-private-subnet'.format(config["workspace_name"])}])
    return subnets


def _create_internet_gateway(config, ec2, vpc):
    cli_logger.print("Creating Internet Gateway for the VPC: {}...".format(vpc.id))
    try:
        igw = ec2.create_internet_gateway()
        igw.create_tags(Tags=[{'Key': 'Name', 'Value': 'cloudtik-{}-internet-gateway'.format(config["workspace_name"])}])
        igw.attach_to_vpc(VpcId=vpc.id)
        cli_logger.print("Successfully created Internet Gateway: cloudtik-{}-internet-gateway ...".format(config["workspace_name"]))
    except Exception as e:
        cli_logger.error("Failed to create Internet Gateway. {}", str(e))
        try:
            cli_logger.print("Try to find the existing Internet Gateway...")
            igws = [igw for igw in vpc.internet_gateways.all()]
            igw = igws[0]
            cli_logger.print("Existing internet gateway found. Will use this one.")
        except Exception:
            raise e
    return igw


def _create_elastic_ip(ec2_client):
    # Allocate Elastic IP
    eip_for_nat_gateway = ec2_client.allocate_address(Domain='vpc')
    return eip_for_nat_gateway


def wait_nat_creation(ec2_client, nat_gateway_id):
    """
    Check if successful state is reached every 15 seconds until a successful state is reached.
    An error is returned after 40 failed checks.
    """
    try:
        waiter = ec2_client.get_waiter('nat_gateway_available')
        waiter.wait(NatGatewayIds=[nat_gateway_id])
    except Exception as e:
        cli_logger.abort('Could not create the NAT gateway.')
        raise


def _create_nat_gateway(config, ec2_client, vpc, subnet):
    cli_logger.print("Creating NAT Gateway for subnet: {}...".format(subnet.id))
    try:
        eip_for_nat_gateway = _create_elastic_ip(ec2_client)
        allocation_id = eip_for_nat_gateway['AllocationId']

        # create NAT Gateway and associate with Elastic IP
        nat_gw = ec2_client.create_nat_gateway(SubnetId=subnet.id, AllocationId=allocation_id)['NatGateway']
        nat_gw_id = nat_gw['NatGatewayId']
        ec2_client.create_tags(Tags=[{'Key': 'Name', 'Value': 'cloudtik-{}-nat-gateway'.format(config["workspace_name"])}],
                        Resources=[nat_gw_id])
        wait_nat_creation(ec2_client, nat_gw_id)
        cli_logger.print("Successfully created NAT Gateway: cloudtik-{}-nat-gateway ...".format(config["workspace_name"]))
    except Exception as e:
        cli_logger.error("Failed to create NAT Gateway. {}", str(e))
        try:
            cli_logger.print("Try to find the existing NAT Gateway...")
            nat_gws = [nat for nat in ec2_client.describe_nat_gateways()['NatGateways'] if nat["VpcId"] == vpc.id]
            nat_gw = nat_gws[0]
            cli_logger.print(
                "Found an existing NAT Gateway. Will use this one")
        except Exception:
            raise e

    return nat_gw


def _update_route_table_for_public_subnet(config, ec2, ec2_client, vpc, subnet, igw):
    cli_logger.print("Updating public subnet route table: {}...".format(subnet.id))
    public_route_tables = get_workspace_public_route_tables(config["workspace_name"], ec2, vpc.id)
    if len(public_route_tables) > 0:
        public_route_table = public_route_tables[0]
    else:
        public_route_table = list(vpc.route_tables.all())[0]
        public_route_table.create_tags(Tags=[{'Key': 'Name', 'Value': 'cloudtik-{}-public-route-table'.format(config["workspace_name"])}])

    # add a default route, for Public Subnet, pointing to Internet Gateway
    try:
        ec2_client.create_route(RouteTableId=public_route_table.id, DestinationCidrBlock='0.0.0.0/0', GatewayId=igw.id)
    except Exception as e:
        cli_logger.error("Failed to create route table. {}", str(e))
        cli_logger.print(
            "Update the rules for route table:{}.".format(public_route_table.id))
        ec2_client.delete_route(RouteTableId=public_route_table.id, DestinationCidrBlock='0.0.0.0/0')
        ec2_client.create_route(RouteTableId=public_route_table.id, DestinationCidrBlock='0.0.0.0/0', GatewayId=igw.id)
    public_route_table.associate_with_subnet(SubnetId=subnet.id)


def _create_route_table_for_private_subnet(config, ec2, vpc, subnet):
    cli_logger.print("Updating private subnet route table: {}...".format(subnet.id))
    private_route_tables = get_workspace_private_route_tables(config["workspace_name"], ec2, vpc.id)
    if len(private_route_tables) > 0:
        private_route_table = private_route_tables[0]
    else:
        private_route_table = ec2.create_route_table(VpcId=vpc.id)
        private_route_table.create_tags(
            Tags=[{'Key': 'Name', 'Value': 'cloudtik-{}-private-route-table'.format(config["workspace_name"])}])

    private_route_table.associate_with_subnet(SubnetId=subnet.id)
    return private_route_table


def _configure_workspace(config):
    ec2 = _resource("ec2", config)
    ec2_client = _client("ec2", config)
    workspace_name = config["workspace_name"]
    use_internal_ips = config["provider"].get("use_internal_ips", False)

    current_step = 1
    total_steps = NUM_AWS_WORKSPACE_CREATION_STEPS

    try:
        with cli_logger.group("Creating workspace: {}", workspace_name):
            with cli_logger.group(
                    "Creating instance profile",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _configure_workspace_instance_profile(config, workspace_name)

            _configure_network_resources(config, ec2, ec2_client,
                                         current_step, total_steps)
    except Exception as e:
        cli_logger.error("Failed to create workspace. {}", str(e))
        raise e

    cli_logger.print(
        "Successfully created workspace: {}.",
        cf.bold(workspace_name))

    return config


def _configure_workspace_instance_profile(config, workspace_name):
    instance_profile_name = _get_workspace_instance_profile_name(workspace_name)
    instance_role_name = "cloudtik-{}-role".format(workspace_name)

    cli_logger.print("Creating instance profile: {}...".format(instance_profile_name))
    _create_or_update_instance_profile(config, instance_profile_name,
                                       instance_role_name)
    cli_logger.print("Successfully created and configured instance profile.")


def _get_workspace_instance_profile_name(workspace_name):
    return "cloudtik-{}-profile".format(workspace_name)


def _configure_network_resources(config, ec2, ec2_client,
                                 current_step, total_steps):
    workspace_name = config["workspace_name"]

    # create VPC
    with cli_logger.group(
            "Creating VPC",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        vpc = _configure_vpc(config, workspace_name, ec2, ec2_client)

    # create subnets
    with cli_logger.group(
            "Creating subnets",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        subnets = _create_and_configure_subnets(config, vpc)

    # TODO check whether we need to create new internet gateway? Maybe existing vpc contains internet subnets
    # create internet gateway for public subnets
    with cli_logger.group(
            "Creating Internet gateway",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        internet_gateway = _create_internet_gateway(config, ec2, vpc)

    # add internet_gateway into public route table
    with cli_logger.group(
            "Updating route table",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _update_route_table_for_public_subnet(config, ec2, ec2_client, vpc, subnets[0], internet_gateway)
        # create private route table for private subnets
        private_route_table = _create_route_table_for_private_subnet(config, ec2, vpc, subnets[-1])

    # create NAT gateway for private subnets
    with cli_logger.group(
            "Creating NAT gateway",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        nat_gateway = _create_nat_gateway(config, ec2_client, vpc, subnets[0])
        # Create a default route pointing to NAT Gateway for private subnets
        ec2_client.create_route(RouteTableId=private_route_table.id, DestinationCidrBlock='0.0.0.0/0',
                                NatGatewayId=nat_gateway['NatGatewayId'])

    with cli_logger.group(
            "Creating security group",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _upsert_security_group(config, vpc.id)

    return config


def _configure_vpc(config, workspace_name, ec2, ec2_client):
    use_internal_ips = config["provider"].get("use_internal_ips", False)
    if use_internal_ips:
        # No need to create new vpc
        VpcId = get_current_vpc(config)
        vpc = ec2.Vpc(id=VpcId)
        vpc.create_tags(Tags=[{'Key': 'Name', 'Value': 'cloudtik-{}-vpc'.format(workspace_name)}])
    else:

        # Need to create a new vpc
        if get_workspace_vpc_id(config["workspace_name"], ec2_client) is None:
            vpc = _create_vpc(config, ec2)
        else:
            raise RuntimeError("There is a same name VPC for workspace: {}, "
                               "if you want to create a new workspace with the same name, "
                               "you need to execute workspace delete first!".format(workspace_name))
    return vpc


def _configure_subnets_cidr(vpc):
    cidr_list = []
    subnets = list(vpc.subnets.all())
    vpc_CidrBlock = vpc.cidr_block
    ip = vpc_CidrBlock.split("/")[0].split(".")

    if len(subnets) == 0:
        for i in range(0, 2):
            cidr_list.append(ip[0] + "." + ip[1] + "." + str(i) + ".0/24")
    else:
        cidr_blocks = [subnet.cidr_block for subnet in subnets]
        for i in range(0, 256):
            tmp_cidr_block = ip[0] + "." + ip[1] + "." + str(i) + ".0/24"

            if check_cidr_conflict(tmp_cidr_block, cidr_blocks):
                cidr_list.append(tmp_cidr_block)
                print("Choose CIDR: {}".format(tmp_cidr_block))

            if len(cidr_list) == 2:
                break

    return cidr_list


def get_current_vpc(config):
    client = _client("ec2", config)
    ip_address = get_node_ip_address(address="8.8.8.8:53")
    VpcId = ""
    for  Reservation in  client.describe_instances().get("Reservations"):
        for instance  in Reservation["Instances"]:
            if instance.get("PrivateIpAddress", "") == ip_address:
                VpcId = instance["VpcId"]

    return VpcId


def _configure_subnet(config):
    ec2 = _resource("ec2", config)
    use_internal_ips = config["provider"].get("use_internal_ips", False)

    # If head or worker security group is specified, filter down to subnets
    # belonging to the same VPC as the security group.
    sg_ids = []
    for node_type in config["available_node_types"].values():
        node_config = node_type["node_config"]
        sg_ids.extend(node_config.get("SecurityGroupIds", []))
    if sg_ids:
        vpc_id_of_sg = _get_vpc_id_of_sg(sg_ids, config)
    else:
        vpc_id_of_sg = None

    try:
        candidate_subnets = ec2.subnets.all()
        if vpc_id_of_sg:
            candidate_subnets = [
                s for s in candidate_subnets if s.vpc_id == vpc_id_of_sg
            ]
        subnets = sorted(
            (s for s in candidate_subnets if s.state == "available" and (
                use_internal_ips or s.map_public_ip_on_launch)),
            reverse=True,  # sort from Z-A
            key=lambda subnet: subnet.availability_zone)
    except botocore.exceptions.ClientError as exc:
        handle_boto_error(exc, "Failed to fetch available subnets from AWS.")
        raise exc

    if not subnets:
        cli_logger.abort(
            "No usable subnets found, try manually creating an instance in "
            "your specified region to populate the list of subnets "
            "and trying this again.\n"
            "Note that the subnet must map public IPs "
            "on instance launch unless you set `use_internal_ips: true` in "
            "the `provider` config.")

    if "availability_zone" in config["provider"]:
        azs = config["provider"]["availability_zone"].split(",")
        subnets = [
            s for az in azs  # Iterate over AZs first to maintain the ordering
            for s in subnets if s.availability_zone == az
        ]
        if not subnets:
            cli_logger.abort(
                "No usable subnets matching availability zone {} found.\n"
                "Choose a different availability zone or try "
                "manually creating an instance in your specified region "
                "to populate the list of subnets and trying this again.",
                config["provider"]["availability_zone"])

    # Use subnets in only one VPC, so that _configure_security_groups only
    # needs to create a security group in this one VPC. Otherwise, we'd need
    # to set up security groups in all of the user's VPCs and set up networking
    # rules to allow traffic between these groups.
    # See pull #14868.
    subnet_ids = [
        s.subnet_id for s in subnets if s.vpc_id == subnets[0].vpc_id
    ]
    # map from node type key -> source of SubnetIds field
    subnet_src_info = {}
    _set_config_info(subnet_src=subnet_src_info)
    for key, node_type in config["available_node_types"].items():
        node_config = node_type["node_config"]
        if "SubnetIds" not in node_config:
            subnet_src_info[key] = "default"
            node_config["SubnetIds"] = subnet_ids
        else:
            subnet_src_info[key] = "config"

    return config


def _configure_subnet_from_workspace(config):
    ec2 = _resource("ec2", config)
    ec2_client = _client("ec2", config)
    workspace_name = config["workspace_name"]
    use_internal_ips = config["provider"].get("use_internal_ips", False)

    VpcId = get_workspace_vpc_id(workspace_name, ec2_client)
    public_subnet_ids = [public_subnet.id for public_subnet in get_workspace_public_subnets(workspace_name, ec2, VpcId)]
    private_subnet_ids = [private_subnet.id for private_subnet in get_workspace_private_subnets(workspace_name, ec2, VpcId)]

    # map from node type key -> source of SubnetIds field
    subnet_src_info = {}
    _set_config_info(subnet_src=subnet_src_info)

    for key, node_type in config["available_node_types"].items():
        node_config = node_type["node_config"]
        if key == config["head_node_type"]:
            if use_internal_ips:
                node_config["SubnetIds"] = private_subnet_ids
            else:
                node_config["SubnetIds"] = public_subnet_ids
        else:
            node_config["SubnetIds"] = private_subnet_ids
        subnet_src_info[key] = "workspace"

    return config


def _get_vpc_id_of_sg(sg_ids: List[str], config: Dict[str, Any]) -> str:
    """Returns the VPC id of the security groups with the provided security
    group ids.

    Errors if the provided security groups belong to multiple VPCs.
    Errors if no security group with any of the provided ids is identified.
    """
    # sort security group IDs to support deterministic unit test stubbing
    sg_ids = sorted(set(sg_ids))

    ec2 = _resource("ec2", config)
    filters = [{"Name": "group-id", "Values": sg_ids}]
    security_groups = ec2.security_groups.filter(Filters=filters)
    vpc_ids = [sg.vpc_id for sg in security_groups]
    vpc_ids = list(set(vpc_ids))

    multiple_vpc_msg = "All security groups specified in the cluster config "\
        "should belong to the same VPC."
    cli_logger.doassert(len(vpc_ids) <= 1, multiple_vpc_msg)
    assert len(vpc_ids) <= 1, multiple_vpc_msg

    no_sg_msg = "Failed to detect a security group with id equal to any of "\
        "the configured SecurityGroupIds."
    cli_logger.doassert(len(vpc_ids) > 0, no_sg_msg)
    assert len(vpc_ids) > 0, no_sg_msg

    return vpc_ids[0]


def _configure_security_group(config):
    # map from node type key -> source of SecurityGroupIds field
    security_group_info_src = {}
    _set_config_info(security_group_src=security_group_info_src)

    for node_type_key in config["available_node_types"]:
        security_group_info_src[node_type_key] = "config"

    node_types_to_configure = [
        node_type_key
        for node_type_key, node_type in config["available_node_types"].items()
        if "SecurityGroupIds" not in node_type["node_config"]
    ]
    if not node_types_to_configure:
        return config  # have user-defined groups
    head_node_type = config["head_node_type"]
    if config["head_node_type"] in node_types_to_configure:
        # configure head node security group last for determinism
        # in tests
        node_types_to_configure.remove(head_node_type)
        node_types_to_configure.append(head_node_type)
    security_groups = _upsert_security_groups(config, node_types_to_configure)

    for node_type_key in node_types_to_configure:
        node_config = config["available_node_types"][node_type_key][
            "node_config"]
        sg = security_groups[node_type_key]
        node_config["SecurityGroupIds"] = [sg.id]
        security_group_info_src[node_type_key] = "default"

    return config


def _configure_security_group_from_workspace(config):
    ec2_client = _client("ec2", config)
    workspace_name = config["workspace_name"]
    VpcId = get_workspace_vpc_id(workspace_name, ec2_client)
    # map from node type key -> source of SecurityGroupIds field
    security_group_info_src = {}
    _set_config_info(security_group_src=security_group_info_src)
    sg = get_workspace_security_group(config,  VpcId, workspace_name)

    for node_type_key in config["available_node_types"].keys():
        node_config = config["available_node_types"][node_type_key][
            "node_config"]
        node_config["SecurityGroupIds"] = [sg.id]
        security_group_info_src[node_type_key] = "workspace"

    return config


def _configure_ami(config):
    """Provide helpful message for missing ImageId for node configuration."""

    # map from node type key -> source of ImageId field
    ami_src_info = {key: "config" for key in config["available_node_types"]}
    _set_config_info(ami_src=ami_src_info)

    region = config["provider"]["region"]
    default_ami = DEFAULT_AMI.get(region)
    if not default_ami:
        cli_logger.abort("Not support on this region: {}. Please use one of these regions {}".
                         format(region, sorted(DEFAULT_AMI.keys())))

    for key, node_type in config["available_node_types"].items():
        node_config = node_type["node_config"]
        node_config["ImageId"] = default_ami

    return config


def _upsert_security_groups(config, node_types):
    security_groups = _get_or_create_vpc_security_groups(config, node_types)
    _upsert_security_group_rules(config, security_groups)

    return security_groups


def _upsert_security_group(config, VpcId):
    cli_logger.print("Updating security group for VPC: {}...".format(VpcId))
    security_group = _create_security_group(config, VpcId,
                                            SECURITY_GROUP_TEMPLATE.format(config["workspace_name"]))
    _add_security_group_rules(config, security_group)

    return security_group


def _update_security_group(config, VpcId):
    cli_logger.print("Updating security group for VPC: {}...".format(VpcId))
    security_group = get_workspace_security_group(config, VpcId,
                                 SECURITY_GROUP_TEMPLATE.format(config["workspace_name"]))
    _add_security_group_rules(config, security_group)

    return security_group


def _get_or_create_vpc_security_groups(conf, node_types):
    # Figure out which VPC each node_type is in...
    ec2 = _resource("ec2", conf)
    node_type_to_vpc = {
        node_type: _get_vpc_id_or_die(
            ec2,
            conf["available_node_types"][node_type]["node_config"]["SubnetIds"]
            [0],
        )
        for node_type in node_types
    }

    # Generate the name of the security group we're looking for...
    expected_sg_name = conf["provider"] \
        .get("security_group", {}) \
        .get("GroupName", SECURITY_GROUP_TEMPLATE.format(conf["cluster_name"]))

    # Figure out which security groups with this name exist for each VPC...
    vpc_to_existing_sg = {
        sg.vpc_id: sg
        for sg in _get_security_groups(
            conf,
            node_type_to_vpc.values(),
            [expected_sg_name],
        )
    }

    # Lazily create any security group we're missing for each VPC...
    vpc_to_sg = LazyDefaultDict(
        partial(_create_security_group, conf, group_name=expected_sg_name),
        vpc_to_existing_sg,
    )

    # Then return a mapping from each node_type to its security group...
    return {
        node_type: vpc_to_sg[vpc_id]
        for node_type, vpc_id in node_type_to_vpc.items()
    }


@lru_cache()
def _get_vpc_id_or_die(ec2, subnet_id):
    subnet = list(
        ec2.subnets.filter(Filters=[{
            "Name": "subnet-id",
            "Values": [subnet_id]
        }]))

    # TODO: better error message
    cli_logger.doassert(len(subnet) == 1, "Subnet ID not found: {}", subnet_id)
    assert len(subnet) == 1, "Subnet ID not found: {}".format(subnet_id)
    subnet = subnet[0]
    return subnet.vpc_id


def _get_security_group(config, vpc_id, group_name):
    security_group = _get_security_groups(config, [vpc_id], [group_name])
    return None if not security_group else security_group[0]


def _get_security_groups(config, vpc_ids, group_names):
    unique_vpc_ids = list(set(vpc_ids))
    unique_group_names = set(group_names)

    ec2 = _resource("ec2", config)
    existing_groups = list(
        ec2.security_groups.filter(Filters=[{
            "Name": "vpc-id",
            "Values": unique_vpc_ids
        }]))
    filtered_groups = [
        sg for sg in existing_groups if sg.group_name in unique_group_names
    ]
    return filtered_groups


def _create_security_group(config, VpcId, group_name):
    client = _client("ec2", config)
    client.create_security_group(
        Description="Auto-created security group for workers",
        GroupName=group_name,
        VpcId=VpcId)
    security_group = _get_security_group(config, VpcId, group_name)
    cli_logger.doassert(security_group,
                        "Failed to create security group")  # err msg

    cli_logger.verbose(
        "Created new security group {}",
        cf.bold(security_group.group_name),
        _tags=dict(id=security_group.id))
    cli_logger.doassert(security_group,
                        "Failed to create security group")  # err msg
    assert security_group, "Failed to create security group"
    return security_group


def _upsert_security_group_rules(conf, security_groups):
    sgids = {sg.id for sg in security_groups.values()}

    # Update sgids to include user-specified security groups.
    # This is necessary if the user specifies the head node type's security
    # groups but not the worker's, or vice-versa.
    for node_type in conf["available_node_types"]:
        sgids.update(conf["available_node_types"][node_type].get(
            "SecurityGroupIds", []))

    # sort security group items for deterministic inbound rule config order
    # (mainly supports more precise stub-based boto3 unit testing)
    for node_type, sg in sorted(security_groups.items()):
        sg = security_groups[node_type]
        if not sg.ip_permissions:
            _update_inbound_rules(sg, sgids, conf)


def _add_security_group_rules(conf, security_group):
    _update_inbound_rules(security_group, {security_group.id}, conf)


def _update_inbound_rules(target_security_group, sgids, config):
    extended_rules = config["provider"] \
        .get("security_group", {}) \
        .get("IpPermissions", [])
    ip_permissions = _create_default_inbound_rules(config, sgids, extended_rules)
    old_ip_permisssions = target_security_group.ip_permissions
    if len(old_ip_permisssions) != 0:
        target_security_group.revoke_ingress(IpPermissions=old_ip_permisssions)
    target_security_group.authorize_ingress(IpPermissions=ip_permissions)


def _create_default_inbound_rules(config, sgids, extended_rules=None):
    if extended_rules is None:
        extended_rules = []
    intra_cluster_rules = _create_default_intra_cluster_inbound_rules(sgids)
    ssh_rules = _create_default_ssh_inbound_rules(sgids, config)
    merged_rules = itertools.chain(
        intra_cluster_rules,
        ssh_rules,
        extended_rules,
    )
    return list(merged_rules)


def _create_default_intra_cluster_inbound_rules(intra_cluster_sgids):
    return [{
        "FromPort": -1,
        "ToPort": -1,
        "IpProtocol": "-1",
        "UserIdGroupPairs": [
            {
                "GroupId": security_group_id
            } for security_group_id in sorted(intra_cluster_sgids)
            # sort security group IDs for deterministic IpPermission models
            # (mainly supports more precise stub-based boto3 unit testing)
        ]
    }]


def _create_default_ssh_inbound_rules(sgids, config):
    vpc_id_of_sg = _get_vpc_id_of_sg(sgids, config)
    ec2 = _resource("ec2", config)
    vpc = ec2.Vpc(vpc_id_of_sg)
    vpc_cidr = vpc.cidr_block
    return [{
        "FromPort": 22,
        "ToPort": 22,
        "IpProtocol": "tcp",
        "IpRanges": [{
            "CidrIp": vpc_cidr
        }]
    }]


def _get_role(role_name, config):
    iam = _resource("iam", config)
    role = iam.Role(role_name)
    try:
        role.load()
        return role
    except botocore.exceptions.ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "NoSuchEntity":
            return None
        else:
            handle_boto_error(
                exc, "Failed to fetch IAM role data for {} from AWS.",
                cf.bold(role_name))
            raise exc


def _get_instance_profile(profile_name, config):
    iam = _resource("iam", config)
    profile = iam.InstanceProfile(profile_name)
    try:
        profile.load()
        return profile
    except botocore.exceptions.ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "NoSuchEntity":
            return None
        else:
            handle_boto_error(
                exc,
                "Failed to fetch IAM instance profile data for {} from AWS.",
                cf.bold(profile_name))
            raise exc


def _get_key(key_name, config):
    ec2 = _resource("ec2", config)
    try:
        for key in ec2.key_pairs.filter(Filters=[{
                "Name": "key-name",
                "Values": [key_name]
        }]):
            if key.name == key_name:
                return key
    except botocore.exceptions.ClientError as exc:
        handle_boto_error(exc, "Failed to fetch EC2 key pair {} from AWS.",
                          cf.bold(key_name))
        raise exc


def _configure_from_launch_template(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges any launch template data referenced by the node config of all
    available node type's into their parent node config. Any parameters
    specified in node config override the same parameters in the launch
    template, in compliance with the behavior of the ec2.create_instances
    API.

    Args:
        config (Dict[str, Any]): config to bootstrap
    Returns:
        config (Dict[str, Any]): The input config with all launch template
        data merged into the node config of all available node types. If no
        launch template data is found, then the config is returned
        unchanged.
    Raises:
        ValueError: If no launch template is found for any launch
        template [name|id] and version, or more than one launch template is
        found.
    """
    # create a copy of the input config to modify
    config = copy.deepcopy(config)
    node_types = config["available_node_types"]

    # iterate over sorted node types to support deterministic unit test stubs
    for name, node_type in sorted(node_types.items()):
        node_types[name] = _configure_node_type_from_launch_template(
            config, node_type)
    return config


def _configure_node_type_from_launch_template(
        config: Dict[str, Any], node_type: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges any launch template data referenced by the given node type's
    node config into the parent node config. Any parameters specified in
    node config override the same parameters in the launch template.

    Args:
        config (Dict[str, Any]): config to bootstrap
        node_type (Dict[str, Any]): node type config to bootstrap
    Returns:
        node_type (Dict[str, Any]): The input config with all launch template
        data merged into the node config of the input node type. If no
        launch template data is found, then the config is returned
        unchanged.
    Raises:
        ValueError: If no launch template is found for the given launch
        template [name|id] and version, or more than one launch template is
        found.
    """
    # create a copy of the input config to modify
    node_type = copy.deepcopy(node_type)

    node_cfg = node_type["node_config"]
    if "LaunchTemplate" in node_cfg:
        node_type["node_config"] = \
            _configure_node_cfg_from_launch_template(config, node_cfg)
    return node_type


def _configure_node_cfg_from_launch_template(
        config: Dict[str, Any], node_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges any launch template data referenced by the given node type's
    node config into the parent node config. Any parameters specified in
    node config override the same parameters in the launch template.

    Note that this merge is simply a bidirectional dictionary update, from
    the node config to the launch template data, and from the launch
    template data to the node config. Thus, the final result captures the
    relative complement of launch template data with respect to node config,
    and allows all subsequent config bootstrapping code paths to act as
    if the complement was explicitly specified in the user's node config. A
    deep merge of nested elements like tag specifications isn't required
    here, since the AWSNodeProvider's ec2.create_instances call will do this
    for us after it fetches the referenced launch template data.

    Args:
        config (Dict[str, Any]): config to bootstrap
        node_cfg (Dict[str, Any]): node config to bootstrap
    Returns:
        node_cfg (Dict[str, Any]): The input node config merged with all launch
        template data. If no launch template data is found, then the node
        config is returned unchanged.
    Raises:
        ValueError: If no launch template is found for the given launch
        template [name|id] and version, or more than one launch template is
        found.
    """
    # create a copy of the input config to modify
    node_cfg = copy.deepcopy(node_cfg)

    ec2 = _client("ec2", config)
    kwargs = copy.deepcopy(node_cfg["LaunchTemplate"])
    template_version = str(kwargs.pop("Version", "$Default"))
    # save the launch template version as a string to prevent errors from
    # passing an integer to ec2.create_instances in AWSNodeProvider
    node_cfg["LaunchTemplate"]["Version"] = template_version
    kwargs["Versions"] = [template_version] if template_version else []

    template = ec2.describe_launch_template_versions(**kwargs)
    lt_versions = template["LaunchTemplateVersions"]
    if len(lt_versions) != 1:
        raise ValueError(f"Expected to find 1 launch template but found "
                         f"{len(lt_versions)}")

    lt_data = template["LaunchTemplateVersions"][0]["LaunchTemplateData"]
    # override launch template parameters with explicit node config parameters
    lt_data.update(node_cfg)
    # copy all new launch template parameters back to node config
    node_cfg.update(lt_data)

    return node_cfg


def _configure_from_network_interfaces(config: Dict[str, Any]) \
        -> Dict[str, Any]:
    """
    Copies all network interface subnet and security group IDs up to their
    parent node config for each available node type.

    Args:
        config (Dict[str, Any]): config to bootstrap
    Returns:
        config (Dict[str, Any]): The input config with all network interface
        subnet and security group IDs copied into the node config of all
        available node types. If no network interfaces are found, then the
        config is returned unchanged.
    Raises:
        ValueError: If [1] subnet and security group IDs exist at both the
        node config and network interface levels, [2] any network interface
        doesn't have a subnet defined, or [3] any network interface doesn't
        have a security group defined.
    """
    # create a copy of the input config to modify
    config = copy.deepcopy(config)

    node_types = config["available_node_types"]
    for name, node_type in node_types.items():
        node_types[name] = _configure_node_type_from_network_interface(
            node_type)
    return config


def _configure_node_type_from_network_interface(node_type: Dict[str, Any]) \
        -> Dict[str, Any]:
    """
    Copies all network interface subnet and security group IDs up to the
    parent node config for the given node type.

    Args:
        node_type (Dict[str, Any]): node type config to bootstrap
    Returns:
        node_type (Dict[str, Any]): The input config with all network interface
        subnet and security group IDs copied into the node config of the
        given node type. If no network interfaces are found, then the
        config is returned unchanged.
    Raises:
        ValueError: If [1] subnet and security group IDs exist at both the
        node config and network interface levels, [2] any network interface
        doesn't have a subnet defined, or [3] any network interface doesn't
        have a security group defined.
    """
    # create a copy of the input config to modify
    node_type = copy.deepcopy(node_type)

    node_cfg = node_type["node_config"]
    if "NetworkInterfaces" in node_cfg:
        node_type["node_config"] = \
            _configure_subnets_and_groups_from_network_interfaces(node_cfg)
    return node_type


def _configure_subnets_and_groups_from_network_interfaces(
        node_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Copies all network interface subnet and security group IDs into their
    parent node config.

    Args:
        node_cfg (Dict[str, Any]): node config to bootstrap
    Returns:
        node_cfg (Dict[str, Any]): node config with all copied network
        interface subnet and security group IDs
    Raises:
        ValueError: If [1] subnet and security group IDs exist at both the
        node config and network interface levels, [2] any network interface
        doesn't have a subnet defined, or [3] any network interface doesn't
        have a security group defined.
    """
    # create a copy of the input config to modify
    node_cfg = copy.deepcopy(node_cfg)

    # If NetworkInterfaces are defined, SubnetId and SecurityGroupIds
    # can't be specified in the same node type config.
    conflict_keys = ["SubnetId", "SubnetIds", "SecurityGroupIds"]
    if any(conflict in node_cfg for conflict in conflict_keys):
        raise ValueError(
            "If NetworkInterfaces are defined, subnets and security groups "
            "must ONLY be given in each NetworkInterface.")
    subnets = _subnets_in_network_config(node_cfg)
    if not all(subnets):
        raise ValueError(
            "NetworkInterfaces are defined but at least one is missing a "
            "subnet. Please ensure all interfaces have a subnet assigned.")
    security_groups = _security_groups_in_network_config(node_cfg)
    if not all(security_groups):
        raise ValueError(
            "NetworkInterfaces are defined but at least one is missing a "
            "security group. Please ensure all interfaces have a security "
            "group assigned.")
    node_cfg["SubnetIds"] = subnets
    node_cfg["SecurityGroupIds"] = list(itertools.chain(*security_groups))

    return node_cfg


def _subnets_in_network_config(config: Dict[str, Any]) -> List[str]:
    """
    Returns all subnet IDs found in the given node config's network interfaces.

    Args:
        config (Dict[str, Any]): node config
    Returns:
        subnet_ids (List[str]): List of subnet IDs for all network interfaces,
        or an empty list if no network interfaces are defined. An empty string
        is returned for each missing network interface subnet ID.
    """
    return [
        ni.get("SubnetId", "") for ni in config.get("NetworkInterfaces", [])
    ]


def _security_groups_in_network_config(config: Dict[str, Any]) \
        -> List[List[str]]:
    """
    Returns all security group IDs found in the given node config's network
    interfaces.

    Args:
        config (Dict[str, Any]): node config
    Returns:
        security_group_ids (List[List[str]]): List of security group ID lists
        for all network interfaces, or an empty list if no network interfaces
        are defined. An empty list is returned for each missing network
        interface security group list.
    """
    return [ni.get("Groups", []) for ni in config.get("NetworkInterfaces", [])]


def verify_s3_storage(provider_config: Dict[str, Any]):
    s3_storage = provider_config.get("aws_s3_storage")
    if s3_storage is None:
        return

    s3 = boto3.client(
        's3',
        aws_access_key_id=s3_storage["s3.access.key.id"],
        aws_secret_access_key=s3_storage["s3.secret.access.key"]
    )

    try:
        s3.list_objects(Bucket=s3_storage["s3.bucket"], Delimiter='/')
    except botocore.exceptions.ClientError as e:
        raise StorageTestingError("Error happens when verifying S3 storage configurations. "
                                  "If you want to go without passing the verification, "
                                  "set 'verify_cloud_storage' to False under provider config. "
                                  "Error: {}.".format(e.message)) from None


def _client(name, config):
    return _resource(name, config).meta.client


def _resource(name, config):
    region = config["provider"]["region"]
    aws_credentials = config["provider"].get("aws_credentials", {})
    return resource_cache(name, region, **aws_credentials)
