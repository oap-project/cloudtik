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

from cloudtik.core._private.providers import _PROVIDER_PRETTY_NAMES
from cloudtik.core._private.cli_logger import cli_logger, cf
from cloudtik.core._private.event_system import (CreateClusterEvent,
                                                  global_event_system)
from cloudtik.core._private.services import get_node_ip_address

from cloudtik.providers._private.aws.utils import LazyDefaultDict, \
    handle_boto_error, resource_cache

logger = logging.getLogger(__name__)

CLOUDTIK = "cloudtik"
CLOUDTIK_DEFAULT_INSTANCE_PROFILE = CLOUDTIK + "-v1"
CLOUDTIK_DEFAULT_IAM_ROLE = CLOUDTIK + "-v1"
SECURITY_GROUP_TEMPLATE = CLOUDTIK + "-{}"

DEFAULT_AMI_NAME = "AWS Deep Learning AMI (Ubuntu 18.04) V30.0"

# Obtained from https://aws.amazon.com/marketplace/pp/B07Y43P7X5 on 8/4/2020.
DEFAULT_AMI = {
    "us-east-1": "ami-029510cec6d69f121",  # US East (N. Virginia)
    "us-east-2": "ami-08bf49c7b3a0c761e",  # US East (Ohio)
    "us-west-1": "ami-0cc472544ce594a19",  # US West (N. California)
    "us-west-2": "ami-0a2363a9cff180a64",  # US West (Oregon)
    "ca-central-1": "ami-0a871851b2ab39f01",  # Canada (Central)
    "eu-central-1": "ami-049fb1ea198d189d7",  # EU (Frankfurt)
    "eu-west-1": "ami-0abcbc65f89fb220e",  # EU (Ireland)
    "eu-west-2": "ami-0755b39fd4dab7cbe",  # EU (London)
    "eu-west-3": "ami-020485d8df1d45530",  # EU (Paris)
    "sa-east-1": "ami-058a6883cbdb4e599",  # SA (Sao Paulo)
}

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
        key_pair_name = ("{}_{}".format(CLOUDTIK, region)
                         if key_name is None else key_name)
        return (key_pair_name,
                os.path.expanduser("~/.ssh/{}.pem".format(key_pair_name)))

    key_pair_name = ("{}_{}_{}".format(CLOUDTIK, i, region)
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


def delete_workspace_aws(config):
    delete_vpc(config)
    # _delete_security_group(config)
    return None



def bootstrap_workspace_aws(config):
    # create a copy of the input config to modify
    config = copy.deepcopy(config)
    # create vpc according to use_internal_ip
    config = _configure_vpc(config)

    # Cluster workers should be in a security group that permits traffic within
    # the group, and also SSH access from outside.
    config = _configure_security_group(config)

    return config


def bootstrap_aws(config):
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
    _check_ami(config)

    return config


def _configure_iam_role(config):
    head_node_type = config["head_node_type"]
    head_node_config = config["available_node_types"][head_node_type][
        "node_config"]
    if "IamInstanceProfile" in head_node_config:
        _set_config_info(head_instance_profile_src="config")
        return config
    _set_config_info(head_instance_profile_src="default")

    instance_profile_name = CLOUDTIK_DEFAULT_INSTANCE_PROFILE
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
        role_name = CLOUDTIK_DEFAULT_IAM_ROLE
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
    # Add IAM role to "head_node" field so that it is applied only to
    # the head node -- not to workers with the same node type as the head.
    config["head_node"]["IamInstanceProfile"] = {"Arn": profile.arn}

    return config


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


def _delete_internet_gateway(ec2, vpcid):
  """ Detach and delete the internet-gateway """
  vpc_resource = ec2.Vpc(vpcid)
  igws = vpc_resource.internet_gateways.all()
  if igws:
    for igw in igws:
      try:
        print("Detaching and Removing igw-id: ", igw.id)
        igw.detach_from_vpc(
          VpcId=vpcid
        )
        igw.delete(
          # DryRun=True
        )
      except boto3.exceptions.Boto3Error as e:
        print(e)


def _delete_subnets(ec2, vpcid):
    return


def _delete_route_table(ec2, vpcid):
    """ Delete the route-tables """
    vpc_resource = ec2.Vpc(vpcid)
    rtbs = vpc_resource.route_tables.all()
    if rtbs:
        try:
            for rtb in rtbs:
                assoc_attr = [rtb.associations_attribute for rtb in rtbs]
                if [rtb_ass[0]['RouteTableId'] for rtb_ass in assoc_attr if rtb_ass[0]['Main'] == True]:
                    print(rtb.id + " is the main route table, continue...")
                    continue
                print("Removing rtb-id: ", rtb.id)
                table = ec2.RouteTable(rtb.id)
                table.delete(
                    # DryRun=True
                )
        except boto3.exceptions.Boto3Error as e:
            print(e)

    return

def _del_security_group(ec2, vpcid):
    return


def _delete_vpc(ec2, vpcid):
    return


def delete_vpc(config):
    ec2 = _resource("ec2", config)
    ec2_client = _client("ec2", config)
    use_internal_ips = config["provider"].get("use_internal_ips", False)
    VpcId = config["provider"]["VpcId"]
    """
     Do the work - order of operation
     1.) Delete the internet-gateway
     2.) Delete subnets
     3.) Delete route-tables
     4.) Delete security-groups
     5.) Delete the VPC 
     """
    _delete_internet_gateway(ec2, VpcId)
    _delete_subnets(ec2, VpcId)
    _delete_route_table(ec2, VpcId)
    _del_security_group(ec2, VpcId)
    _delete_vpc(ec2, VpcId)


def _create_vpc(config,  ec2):
    cli_logger.print("Start to create customer vpc...")
    # create vpc
    try:
        vpc = ec2.create_vpc(CidrBlock='10.10.0.0/16')
        vpc.modify_attribute(EnableDnsSupport={'Value': True})
        vpc.modify_attribute(EnableDnsHostnames={'Value': True})
        vpc.create_tags(Tags=[{'Key': 'Name', 'Value': 'cloudtik_{}_vpc'.format(config["workspace_name"])}])
        cli_logger.print("Have successfully create vpc: cloudtik_{}_vpc ...".format(config["workspace_name"]))
    except Exception as e:
        # todo: add better exception info
        cli_logger.verbose_error("{}", str(e))
        cli_logger.abort(
            "Cannot create vpc... please check weather you have reach the  maximum number of VPCs.")
    return vpc


def _create_subnet(config, cidr,  vpc):
    cli_logger.print("Start to create subnet for the vpc: {} with CIDR: {} ...".format(vpc.id, cidr))
    try:
        subnet = vpc.create_subnet(CidrBlock=cidr)
        cli_logger.print("Have successfully create subnet: cloudtik_{}_subnet ...".format(config["workspace_name"]))
    except Exception as e:
        cli_logger.verbose_error("{}", str(e))
        cli_logger.abort(
            "Cannot create subnet, please check whether the CIDR has conflict with other subnets...")
    return subnet


def _create_and_configure_subnets(config, vpc):
    subnets = []
    cidr_list = _configure_subnets_cidr(vpc.cidr_block)
    for i in range(0, len(cidr_list)):
        subnets.append(_create_subnet(config, cidr_list[i], vpc))

    assert len(subnets) == 3, "We must create 3 subnets for VPC: {}!".format(vpc.id)

    for i in range(0, len(cidr_list)):
        # make the first subnet as public subnet and rename it
        if i == 0:
            subnets[0].meta.client.modify_subnet_attribute(SubnetId=subnets[0].id, MapPublicIpOnLaunch={"Value": True})
            subnets[0].create_tags(Tags=[{'Key': 'Name',
                                          'Value': 'cloudtik_{}_public_subnet'.format(config["workspace_name"])}])
        else:
            subnets[i].create_tags(Tags=[{'Key': 'Name',
                                          'Value': 'cloudtik_{}_private_subnet_{}'.format(config["workspace_name"], i)}])
    return subnets


def _create_internet_gateway(config, ec2, vpc):
    cli_logger.print("Start to create internet gateway for the vpc: {}...".format(vpc.id))
    try:
        igw = ec2.create_internet_gateway()
        igw.create_tags(Tags=[{'Key': 'Name', 'Value': 'cloudtik_{}_internet_gateway'.format(config["workspace_name"])}])
        igw.attach_to_vpc(VpcId=vpc.id)
        cli_logger.print("Have successfully create internet gateway: cloudtik_{}_internet_gateway ...".format(config["workspace_name"]))
    except Exception as e:
        cli_logger.verbose_error("{}", str(e))
        cli_logger.abort(
            "Cannot create internet gateway... "
            "please check weather you have reach the  maximum number of internet gateways...")
    return igw


def _create_elastic_ip(ec2_client):
    # Allocate Elastic IP
    eip_for_nat_gateway = ec2_client.allocate_address(Domain='vpc')
    return eip_for_nat_gateway


def _create_nat_gateway(config, ec2_client, subnet):
    cli_logger.print("Start to create nat gateway for the subnet: {}...".format(subnet.id))
    try:
        eip_for_nat_gateway = _create_elastic_ip(ec2_client)
        allocation_id = eip_for_nat_gateway['AllocationId']

        # create NAT Gateway and associate with Elastic IP
        nat_gw = ec2_client.create_nat_gateway(SubnetId=subnet.id, AllocationId=allocation_id)
        nat_gw_id = nat_gw['NatGateway']['NatGatewayId']

        nat_gw.create_tags(Tags=[{'Key': 'Name', 'Value': 'cloudtik_{}_nat_gateway'.format(config["workspace_name"])}])
        cli_logger.print("Have successfully create nat gateway: cloudtik_{}_nat_gateway ...".format(config["workspace_name"]))
    except Exception as e:
        cli_logger.verbose_error("{}", str(e))
        cli_logger.abort(
            "Cannot create nat gateway... "
            "please check weather you have reach the  maximum number of nat gateways...")
    return nat_gw


def _update_route_table_for_public_subnet(config, ec2_client, vpc, subnet, igw):
    public_route_table = list(vpc.route_tables.all())[0]
    public_route_table.create_tags(Tags=[{'Key': 'Name', 'Value': 'cloudtik_{}_public_route_table'.format(config["workspace_name"])}])
    # add a default route, for Public Subnet, pointing to Internet Gateway
    ec2_client.create_route(RouteTableId=public_route_table.id, DestinationCidrBlock='0.0.0.0/0', GatewayId=igw.id)
    public_route_table.associate_with_subnet(SubnetId=subnet.id)


def _create_route_table_for_private_subnet(config, ec2, vpc, subnets):
    private_route_table = ec2.create_route_table(VpcId=vpc.id)
    private_route_table.create_tags(
        Tags=[{'Key': 'Name', 'Value': 'cloudtik_{}_private_route_table'.format(config["workspace_name"])}])
    private_route_table.associate_with_subnet(SubnetId=subnets[1].id)
    private_route_table.associate_with_subnet(SubnetId=subnets[2].id)


def _configure_vpc(config):
    ec2 = _resource("ec2", config)
    ec2_client = _client("ec2", config)
    use_internal_ips = config["provider"].get("use_internal_ips", False)

    if use_internal_ips:
        # No need to create new vpc
        VpcId = get_current_vpc(config)
        vpc = ec2.Vpc(id=VpcId)
        vpc.create_tags(Tags=[{'Key': 'Name', 'Value': 'cloudtik_{}_vpc'.format(config["workspace_name"])}])
    else:
        # Need to create a new vpc
        vpc = _create_vpc(config, ec2)

    # create subnets
    subnets = _create_and_configure_subnets(config, vpc)

    # TODO check whether we need to create new internet gateway? Maybe existing vpc contains internet subnets
    # create internet gateway for public subnets
    internet_gateway = _create_internet_gateway(config, ec2, vpc)

    # add internet_gateway into public route table
    _update_route_table_for_public_subnet(config, ec2_client, vpc, subnets[0], internet_gateway)

    # create private route table for private subnets
    private_route_table = _create_route_table_for_private_subnet(config, ec2, vpc, subnets)

    # create nate_gatway for private subnets
    nat_gateway = _create_nat_gateway(config, ec2_client, subnets[0])

    # Create a default route pointing to NAT Gateway for private subnets
    ec2_client.create_route(RouteTableId=private_route_table.id, DestinationCidrBlock='0.0.0.0/0',
                            NatGatewayId=nat_gateway['NatGateway']['NatGatewayId'])

    # TODO: Maybe need to record these settings on AWS resource? Need to verify later.
    for key, node_type in config["available_node_types"].items():
        node_config = node_type["node_config"]
        if key == config["head_node_type"]:
            if use_internal_ips:
                node_config["SubnetIds"] = [subnets[1], subnets[2]]
            else:
                node_config["SubnetIds"] = [subnets[0]]
        else:
            node_config["SubnetIds"] = [subnets[1], subnets[2]]

    config["provider"]["VpcId"] = subnets[0].vpc_id

    return config


def _configure_subnets_cidr(vpc_CidrBlock):
    cidr_list = []
    ip = vpc_CidrBlock.split("/")[0].split(".")
    for i in range(0, 3):
        cidr_list.append(ip[0] + "." + ip[1] + "." + str(i) + ".0/24")
    return cidr_list


def get_current_vpc(config):
    client = _client("ec2", config)
    ip_address = get_node_ip_address(address="8.8.8.8:53")
    VpcId = ""
    for  Reservation in  client.describe_instances().get("Reservations"):
        for instance  in Reservation["Instances"]:
            if instance["PrivateIpAddress"] == ip_address:
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


def _check_ami(config):
    """Provide helpful message for missing ImageId for node configuration."""

    # map from node type key -> source of ImageId field
    ami_src_info = {key: "config" for key in config["available_node_types"]}
    _set_config_info(ami_src=ami_src_info)

    region = config["provider"]["region"]
    default_ami = DEFAULT_AMI.get(region)
    if not default_ami:
        # If we do not provide a default AMI for the given region, noop.
        return

    for key, node_type in config["available_node_types"].items():
        node_config = node_type["node_config"]
        node_ami = node_config.get("ImageId", "").lower()
        if node_ami in ["", "latest_dlami"]:
            node_config["ImageId"] = default_ami
            ami_src_info[key] = "dlami"


def _upsert_security_groups(config, node_types):
    security_groups = _get_or_create_vpc_security_groups(config, node_types)
    _upsert_security_group_rules(config, security_groups)

    return security_groups


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


def _create_security_group(config, vpc_id, group_name):
    client = _client("ec2", config)
    client.create_security_group(
        Description="Auto-created security group for workers",
        GroupName=group_name,
        VpcId=vpc_id)
    security_group = _get_security_group(config, vpc_id, group_name)
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


def _update_inbound_rules(target_security_group, sgids, config):
    extended_rules = config["provider"] \
        .get("security_group", {}) \
        .get("IpPermissions", [])
    ip_permissions = _create_default_inbound_rules(sgids, extended_rules)
    target_security_group.authorize_ingress(IpPermissions=ip_permissions)


def _create_default_inbound_rules(sgids, extended_rules=None):
    if extended_rules is None:
        extended_rules = []
    intracluster_rules = _create_default_intracluster_inbound_rules(sgids)
    ssh_rules = _create_default_ssh_inbound_rules()
    merged_rules = itertools.chain(
        intracluster_rules,
        ssh_rules,
        extended_rules,
    )
    return list(merged_rules)


def _create_default_intracluster_inbound_rules(intracluster_sgids):
    return [{
        "FromPort": -1,
        "ToPort": -1,
        "IpProtocol": "-1",
        "UserIdGroupPairs": [
            {
                "GroupId": security_group_id
            } for security_group_id in sorted(intracluster_sgids)
            # sort security group IDs for deterministic IpPermission models
            # (mainly supports more precise stub-based boto3 unit testing)
        ]
    }]


def _create_default_ssh_inbound_rules():
    return [{
        "FromPort": 22,
        "ToPort": 22,
        "IpProtocol": "tcp",
        "IpRanges": [{
            "CidrIp": "0.0.0.0/0"
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


def _client(name, config):
    return _resource(name, config).meta.client


def _resource(name, config):
    region = config["provider"]["region"]
    aws_credentials = config["provider"].get("aws_credentials", {})
    return resource_cache(name, region, **aws_credentials)
