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
import random
import string

import boto3
import botocore

from cloudtik.core.tags import CLOUDTIK_TAG_NODE_KIND, NODE_KIND_HEAD, CLOUDTIK_TAG_CLUSTER_NAME
from cloudtik.core._private.providers import _PROVIDER_PRETTY_NAMES
from cloudtik.core._private.cli_logger import cli_logger, cf
from cloudtik.core._private.services import get_node_ip_address
from cloudtik.core._private.utils import check_cidr_conflict, is_use_internal_ip, \
    is_managed_cloud_storage, is_use_managed_cloud_storage, is_managed_cloud_database, is_use_managed_cloud_database, \
    is_worker_role_for_cloud_storage, is_use_working_vpc, is_use_peering_vpc, is_peering_firewall_allow_ssh_only, \
    is_peering_firewall_allow_working_subnet, is_gpu_runtime
from cloudtik.core.workspace_provider import Existence, CLOUDTIK_MANAGED_CLOUD_STORAGE, \
    CLOUDTIK_MANAGED_CLOUD_STORAGE_URI, CLOUDTIK_MANAGED_CLOUD_DATABASE, CLOUDTIK_MANAGED_CLOUD_DATABASE_ENDPOINT, \
    CLOUDTIK_MANAGED_CLOUD_DATABASE_PORT
from cloudtik.providers._private.aws.utils import LazyDefaultDict, \
    handle_boto_error, get_boto_error_code, _get_node_info, BOTO_MAX_RETRIES, _resource, \
    _resource_client, _make_resource, _make_resource_client, make_ec2_client, export_aws_s3_storage_config, \
    get_aws_s3_storage_config, get_aws_s3_storage_config_for_update, _working_node_client, _working_node_resource, \
    get_aws_cloud_storage_uri, AWS_S3_BUCKET, _make_client, get_aws_database_config, export_aws_database_config, \
    get_aws_database_config_for_update, AWS_DATABASE_ENDPOINT
from cloudtik.providers._private.utils import StorageTestingError

logger = logging.getLogger(__name__)

AWS_RESOURCE_NAME_PREFIX = "cloudtik"
SECURITY_GROUP_TEMPLATE = AWS_RESOURCE_NAME_PREFIX + "-{}"

AWS_WORKSPACE_VERSION_TAG_NAME = "cloudtik-workspace-version"
AWS_WORKSPACE_VERSION_CURRENT = "1"

AWS_WORKSPACE_VPC_NAME = AWS_RESOURCE_NAME_PREFIX + "-{}-vpc"
AWS_WORKSPACE_VPC_PEERING_NAME = AWS_RESOURCE_NAME_PREFIX + "-{}-vpc-peering-connection"
AWS_WORKSPACE_VPC_S3_ENDPOINT = AWS_RESOURCE_NAME_PREFIX + "-{}-vpc-endpoint-s3"

AWS_WORKSPACE_DB_SUBNET_GROUP_NAME = "cloudtik-{}-db-subnet-group"
AWS_WORKSPACE_DATABASE_NAME = "cloudtik-{}-db"

DEFAULT_AMI_NAME_PREFIX = "ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server"

# Obtained from https://aws.amazon.com/marketplace/pp/B07Y43P7X5 on 8/4/2020.
DEFAULT_AMI = {
    "us-east-1": "ami-0149b2da6ceec4bb0",  # US East (N. Virginia)
    "us-east-2": "ami-0d5bf08bc8017c83b",  # US East (Ohio)
    "us-west-1": "ami-03f6d497fceb40069",  # US West (N. California)
    "us-west-2": "ami-0c09c7eb16d3e8e70",  # US West (Oregon)
    "af-south-1": "ami-0fffe3a460634f60c",  # Africa (Cape Town)
    "ap-east-1": "ami-09800b995a7e41703",  # Asia Pacific (Hong Kong)
    "ap-south-1": "ami-024c319d5d14b463e",  # Asia Pacific (Mumbai)
    "ap-northeast-1": "ami-09b18720cb71042df",  # Asia Pacific (Tokyo)
    "ap-northeast-2": "ami-07d16c043aa8e5153",  # Asia Pacific (Seoul),
    "ap-northeast-3": "ami-09d2f3a31110c6ad4",  # Asia Pacific (Osaka),
    "ap-southeast-1": "ami-00e912d13fbb4f225",  # Asia Pacific (Singapore)
    "ap-southeast-2": "ami-055166f8a8041fbf1",  # Asia Pacific (Sydney),
    "ap-southeast-3": "ami-06704743af22a1200",  # Asia Pacific (Jakarta)
    "ca-central-1": "ami-043a72cf696697251",  # Canada (Central)
    "eu-central-1": "ami-06148e0e81e5187c8",  # EU (Frankfurt)
    "eu-west-1": "ami-0fd8802f94ed1c969",  # EU (Ireland)
    "eu-west-2": "ami-04842bc62789b682e",  # EU (London)
    "eu-west-3": "ami-064736ff8301af3ee",  # EU (Paris)
    "eu-south-1": "ami-0e825b1b63ff6b36a",  # EU (Milan)
    "eu-north-1": "ami-00b696228b0185ffe",  # EU (Stockholm)
    "me-south-1": "ami-00df83d12eb4b3c4e",  # Middle East (Bahrain)
    "sa-east-1": "ami-00742e66d44c13cd9",  # SA (Sao Paulo)
}

DEFAULT_AMI_NAME_PREFIX_GPU = "AWS Deep Learning Base AMI GPU CUDA 11 (Ubuntu 20.04)"

DEFAULT_AMI_GPU = {
    "us-east-1": "ami-02ea7c238b7ba36af",
    "us-east-2": "ami-0bc3c221aef20fb80",
    "us-west-1": "ami-0c936c23b91cb09f3",
    "us-west-2": "ami-0044186f8cba64624",
    "af-south-1": "ami-057c0d8ac71c6945a",
    "ap-east-1": "ami-04f68b82b64589073",
    "ap-south-1": "ami-0a7218430dc6fb541",
    "ap-northeast-1": "ami-051cb3b8c84c0fe88",
    "ap-northeast-2": "ami-0d80b63a196840f6c",
    "ap-northeast-3": "ami-0877936062e0e6ec4",
    "ap-southeast-1": "ami-091708b0547aef03d",
    "ap-southeast-2": "ami-0182289040a1ed516",
    "ap-southeast-3": "ami-0126ebdde927850a6",
    "ca-central-1": "ami-00eea8631708145cc",
    "eu-central-1": "ami-09671ce2422ed8f83",
    "eu-west-1": "ami-0ea7e93c8626716f4",
    "eu-west-2": "ami-0246db3bcf68e5f9b",
    "eu-west-3": "ami-0a8da46354e76997e",
    "eu-south-1": "ami-0840c9dff1f1d6312",
    "eu-north-1": "ami-018358e3bd01b00a3",
    "me-south-1": "ami-0fecc2978a274d09f",
    "sa-east-1": "ami-0cfdee64f7ca89369",
}

AWS_VPC_SUBNETS_COUNT = 2
AWS_VPC_PUBLIC_SUBNET_INDEX = 0

AWS_WORKSPACE_NUM_CREATION_STEPS = 8
AWS_WORKSPACE_NUM_DELETION_STEPS = 9
AWS_WORKSPACE_NUM_UPDATE_STEPS = 1
AWS_WORKSPACE_TARGET_RESOURCES = 10

AWS_MANAGED_STORAGE_S3_BUCKET = "aws.managed.storage.s3.bucket"

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
        key_pair_name = ("{}_aws_{}".format(AWS_RESOURCE_NAME_PREFIX, region)
                         if key_name is None else key_name)
        return (key_pair_name,
                os.path.expanduser("~/.ssh/{}.pem".format(key_pair_name)))

    key_pair_name = ("{}_aws_{}_{}".format(AWS_RESOURCE_NAME_PREFIX, region, i)
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


def list_ec2_instances(region: str, aws_credentials: Dict[str, Any] = None
                       ) -> List[Dict[str, Any]]:
    """Get all instance-types/resources available in the user's AWS region.
    Args:
        region (str): the region of the AWS provider. e.g., "us-west-2".
    Returns:
        final_instance_types: a list of instances. An example of one element in
        the list:
            {'InstanceType': 'm5a.xlarge', 'ProcessorInfo':
            {'SupportedArchitectures': ['x86_64'], 'SustainedClockSpeedInGhz':
            2.5},'VCpuInfo': {'DefaultVCpus': 4, 'DefaultCores': 2,
            'DefaultThreadsPerCore': 2, 'ValidCores': [2],
            'ValidThreadsPerCore': [1, 2]}, 'MemoryInfo': {'SizeInMiB': 16384},
            ...}

    """
    final_instance_types = []
    ec2 = make_ec2_client(
        region=region,
        max_retries=BOTO_MAX_RETRIES,
        aws_credentials=aws_credentials)
    instance_types = ec2.describe_instance_types()
    final_instance_types.extend(copy.deepcopy(instance_types["InstanceTypes"]))
    while "NextToken" in instance_types:
        instance_types = ec2.describe_instance_types(
            NextToken=instance_types["NextToken"])
        final_instance_types.extend(
            copy.deepcopy(instance_types["InstanceTypes"]))

    return final_instance_types


def get_latest_ami_id(cluster_config: Dict[str, Any], is_gpu):
    name_filter = DEFAULT_AMI_NAME_PREFIX_GPU if is_gpu else DEFAULT_AMI_NAME_PREFIX
    try:
        ec2 = make_ec2_client(
            region=cluster_config["provider"]["region"],
            max_retries=BOTO_MAX_RETRIES,
            aws_credentials=cluster_config["provider"].get("aws_credentials"))
        response = ec2.describe_images(Owners=["amazon"],
                                       Filters=[{'Name': 'name', 'Values': [name_filter + "*"]},
                                                {"Name": "is-public", "Values": ["true"]},
                                                {"Name": "state", "Values": ["available"]}
                                                ])
        images = response.get('Images', [])
        if len(images) > 0:
            images.sort(key=lambda item: item['CreationDate'], reverse=True)
            ami_id = images[0]["ImageId"]
            return ami_id
        else:
            return None
    except Exception as e:
        cli_logger.warning(
            "Error when getting latest AWS AMI information: {}", str(e))
        return None


def post_prepare_aws(config: Dict[str, Any]) -> Dict[str, Any]:
    try:
        config = fill_available_node_types_resources(config)
    except Exception as exc:
        cli_logger.warning(
            "Failed to detect node resources. Make sure you have properly configured the AWS credentials: {}.",
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
    instances_list = list_ec2_instances(
        cluster_config["provider"]["region"],
        cluster_config["provider"].get("aws_credentials"))
    instances_dict = {
        instance["InstanceType"]: instance
        for instance in instances_list
    }

    # Update the instance information to node type
    available_node_types = cluster_config["available_node_types"]
    for node_type in available_node_types:
        instance_type = available_node_types[node_type]["node_config"][
            "InstanceType"]
        if instance_type in instances_dict:
            cpus = instances_dict[instance_type]["VCpuInfo"][
                "DefaultVCpus"]
            detected_resources = {"CPU": cpus}

            memory_total = instances_dict[instance_type]["MemoryInfo"][
                "SizeInMiB"]
            memory_total_in_bytes = int(memory_total) * 1024 * 1024
            detected_resources["memory"] = memory_total_in_bytes

            gpus = instances_dict[instance_type].get("GpuInfo",
                                                     {}).get("Gpus")
            if gpus is not None:
                assert len(gpus) == 1
                gpu_name = gpus[0]["Name"]
                detected_resources.update({
                    "GPU": gpus[0]["Count"],
                    f"accelerator_type:{gpu_name}": 1
                })

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
                             " is not available in AWS region: " +
                             cluster_config["provider"]["region"] + ".")
    return cluster_config


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
    vpc = describe_workspace_vpc(workspace_name, ec2_client)
    if vpc is None:
        return None

    return vpc["VpcId"]


def get_workspace_vpc(workspace_name, ec2_client, ec2):
    vpc_id = get_workspace_vpc_id(workspace_name, ec2_client)
    if vpc_id is None:
        return None
    return ec2.Vpc(vpc_id)


def get_workspace_vpc_name(workspace_name):
    return AWS_WORKSPACE_VPC_NAME.format(workspace_name)


def get_workspace_vpc_peering_name(workspace_name):
    return AWS_WORKSPACE_VPC_PEERING_NAME.format(workspace_name)


def get_workspace_vpc_s3_endpoint(workspace_name):
    return AWS_WORKSPACE_VPC_S3_ENDPOINT.format(workspace_name)


def describe_workspace_vpc(workspace_name, ec2_client):
    vpc_name = get_workspace_vpc_name(workspace_name)
    vpcs = [vpc for vpc in ec2_client.describe_vpcs()["Vpcs"] if not vpc.get("Tags") is None]
    workspace_vpcs = [vpc for vpc in vpcs
             for tag in vpc["Tags"]
                if tag['Key'] == 'Name' and tag['Value'] == vpc_name]

    if len(workspace_vpcs) == 0:
        return None
    elif len(workspace_vpcs) == 1:
        return workspace_vpcs[0]
    else:
        raise RuntimeError("The workspace {} should not have more than one VPC.".format(workspace_name))


def get_workspace_private_subnets(workspace_name, ec2, vpc_id):
    vpc = ec2.Vpc(vpc_id)
    return _get_workspace_private_subnets(workspace_name, vpc)


def get_workspace_public_subnets(workspace_name, ec2, vpc_id):
    vpc = ec2.Vpc(vpc_id)
    return _get_workspace_public_subnets(workspace_name, vpc)


def _get_workspace_private_subnets(workspace_name, vpc):
    return _get_workspace_subnets(workspace_name, vpc, "cloudtik-{}-private-subnet")


def _get_workspace_public_subnets(workspace_name, vpc):
    return _get_workspace_subnets(workspace_name, vpc, "cloudtik-{}-public-subnet")


def _get_workspace_subnets(workspace_name, vpc, name_pattern):
    subnets = [subnet for subnet in vpc.subnets.all() if subnet.tags]
    workspace_subnets = [subnet for subnet in subnets
                                 for tag in subnet.tags
                                 if tag['Key'] == 'Name' and tag['Value'].startswith(
                                    name_pattern.format(workspace_name))]
    return workspace_subnets


def get_workspace_nat_gateways(workspace_name, ec2_client, vpc_id):
    nat_gateways = [nat for nat in get_vpc_nat_gateways(ec2_client, vpc_id)
                    if nat.get("Tags") is not None]
    workspace_nat_gateways = [nat for nat in nat_gateways
                              for tag in nat['Tags']
                              if tag['Key'] == 'Name' and tag['Value'] == "cloudtik-{}-nat-gateway".format(
                                workspace_name)]
    return workspace_nat_gateways


def get_vpc_nat_gateways(ec2_client, vpc_id):
    return [nat for nat in ec2_client.describe_nat_gateways()['NatGateways']
            if nat["VpcId"] == vpc_id and nat["State"] != 'deleted']


def _get_workspace_route_table_ids(workspace_name, ec2, vpc_id):
    vpc = ec2.Vpc(vpc_id)
    rtbs = [rtb for rtb in vpc.route_tables.all() if rtb.tags]

    workspace_rtb_ids = [rtb.id for rtb in rtbs
                         for tag in rtb.tags
                         if tag['Key'] == 'Name' and
                         "cloudtik-{}".format(workspace_name) in tag['Value']]

    return workspace_rtb_ids


def get_workspace_private_route_tables(workspace_name, ec2, vpc_id):
    vpc = ec2.Vpc(vpc_id)
    rtbs = [rtb for rtb in vpc.route_tables.all() if rtb.tags]

    workspace_private_rtbs = [rtb for rtb in rtbs
                              for tag in rtb.tags
                              if tag['Key'] == 'Name' and tag['Value'] == "cloudtik-{}-private-route-table".format(
                                workspace_name)]

    return workspace_private_rtbs


def get_vpc_route_tables(vpc):
    vpc_rtbs = [rtb for rtb in vpc.route_tables.all() if rtb.vpc_id == vpc.id]

    return vpc_rtbs


def get_workspace_public_route_tables(workspace_name, ec2, vpc_id):
    vpc = ec2.Vpc(vpc_id)
    rtbs = [rtb for rtb in vpc.route_tables.all() if rtb.tags]

    workspace_public_rtbs = [rtb for rtb in rtbs
                             for tag in rtb.tags
                             if tag['Key'] == 'Name' and tag['Value'] == "cloudtik-{}-public-route-table".format(
                               workspace_name)]

    return workspace_public_rtbs


def get_workspace_security_group(config, vpc_id, workspace_name):
    return _get_security_group(
        config["provider"], vpc_id, SECURITY_GROUP_TEMPLATE.format(workspace_name))


def get_workspace_internet_gateways(workspace_name, ec2, vpc_id):
    igws = [igw for igw in get_vpc_internet_gateways(ec2, vpc_id) if igw.tags]

    workspace_igws = [igw for igw in igws
                      for tag in igw.tags
                      if tag['Key'] == 'Name' and tag['Value'] == "cloudtik-{}-internet-gateway".format(
                        workspace_name)]

    return workspace_igws


def get_vpc_internet_gateways(ec2, vpc_id):
    vpc = ec2.Vpc(vpc_id)
    return list(vpc.internet_gateways.all())


def get_vpc_endpoint_for_s3(ec2_client, vpc_id, workspace_name):
    s3_endpoint_name = get_workspace_vpc_s3_endpoint(workspace_name)
    vpc_endpoint = ec2_client.describe_vpc_endpoints(Filters=[
        {'Name': 'vpc-id', 'Values': [vpc_id]},
        {'Name': 'tag:Name', 'Values': [s3_endpoint_name]}
    ])
    return vpc_endpoint['VpcEndpoints']


def check_aws_workspace_existence(config):
    ec2 = _resource("ec2", config)
    ec2_client = _resource_client("ec2", config)
    workspace_name = config["workspace_name"]
    managed_cloud_storage = is_managed_cloud_storage(config)
    managed_cloud_database = is_managed_cloud_database(config)
    use_peering_vpc = is_use_peering_vpc(config)

    existing_resources = 0
    target_resources = AWS_WORKSPACE_TARGET_RESOURCES
    if managed_cloud_storage:
        target_resources += 1
    if managed_cloud_database:
        target_resources += 1
    if use_peering_vpc:
        target_resources += 1

    """
         Do the work - order of operation:
         Check VPC
         Check private subnets
         Check public subnets
         Check nat-gateways
         Check route-tables
         Check Internet-gateways
         Check security-group
         Check VPC endpoint for s3
         Check VPC peering if needed
         Instance profiles
         Check S3 bucket
         Check database instance
    """
    skipped_resources = 0
    vpc_id = get_workspace_vpc_id(workspace_name, ec2_client)
    if vpc_id is not None:
        existing_resources += 1
        # Network resources that depending on VPC
        if len(get_workspace_private_subnets(workspace_name, ec2, vpc_id)) >= AWS_VPC_SUBNETS_COUNT - 1:
            existing_resources += 1
        if len(get_workspace_public_subnets(workspace_name, ec2, vpc_id)) > 0:
            existing_resources += 1
        if len(get_workspace_nat_gateways(workspace_name, ec2_client, vpc_id)) > 0:
            existing_resources += 1
        elif len(get_vpc_nat_gateways(ec2_client, vpc_id)) > 0:
            existing_resources += 1
            skipped_resources += 1
        if len(get_workspace_private_route_tables(workspace_name, ec2, vpc_id)) > 0:
            existing_resources += 1
        if len(get_workspace_internet_gateways(workspace_name, ec2, vpc_id)) > 0:
            existing_resources += 1
        elif len(get_vpc_internet_gateways(ec2, vpc_id)) > 0:
            existing_resources += 1
            skipped_resources += 1
        if get_workspace_security_group(config, vpc_id, workspace_name) is not None:
            existing_resources += 1
        if len(get_vpc_endpoint_for_s3(ec2_client, vpc_id, workspace_name)) > 0:
            existing_resources += 1
        if use_peering_vpc:
            if get_workspace_vpc_peering_connection(config) is not None:
                existing_resources += 1

    if _get_head_instance_profile(config) is not None:
        existing_resources += 1

    if _get_worker_instance_profile(config) is not None:
        existing_resources += 1

    cloud_storage_existence = False
    if managed_cloud_storage:
        if get_workspace_s3_bucket(config, workspace_name) is not None:
            existing_resources += 1
            cloud_storage_existence = True

    cloud_database_existence = False
    if managed_cloud_database:
        if get_workspace_database_instance(config, workspace_name) is not None:
            existing_resources += 1
            cloud_database_existence = True

    if existing_resources <= skipped_resources:
        return Existence.NOT_EXIST
    elif existing_resources == target_resources:
        return Existence.COMPLETED
    else:
        if existing_resources == skipped_resources + 1 and cloud_storage_existence:
            return Existence.STORAGE_ONLY
        elif existing_resources == skipped_resources + 1 and cloud_database_existence:
            return Existence.DATABASE_ONLY
        elif existing_resources == skipped_resources + 2 and cloud_storage_existence \
                and cloud_database_existence:
            return Existence.STORAGE_AND_DATABASE_ONLY
        return Existence.IN_COMPLETED


def check_aws_workspace_integrity(config):
    existence = check_aws_workspace_existence(config)
    return True if existence == Existence.COMPLETED else False


def get_aws_workspace_info(config):
    managed_cloud_storage = is_managed_cloud_storage(config)
    managed_cloud_database = is_managed_cloud_database(config)

    info = {}
    if managed_cloud_storage:
        get_aws_managed_cloud_storage_info(
            config, config["provider"], info)

    if managed_cloud_database:
        get_aws_managed_cloud_database_info(
            config, config["provider"], info)
    return info


def get_aws_managed_cloud_storage_info(config, cloud_provider, info):
    workspace_name = config["workspace_name"]
    bucket = get_managed_s3_bucket(cloud_provider, workspace_name)
    managed_bucket_name = None if bucket is None else bucket.name

    if managed_bucket_name is not None:
        aws_cloud_storage = {AWS_S3_BUCKET: managed_bucket_name}
        managed_cloud_storage = {
            AWS_MANAGED_STORAGE_S3_BUCKET: managed_bucket_name,
            CLOUDTIK_MANAGED_CLOUD_STORAGE_URI: get_aws_cloud_storage_uri(aws_cloud_storage)
        }
        info[CLOUDTIK_MANAGED_CLOUD_STORAGE] = managed_cloud_storage


def get_aws_managed_cloud_database_info(config, cloud_provider, info):
    workspace_name = config["workspace_name"]
    database_instance = get_managed_database_instance(
        cloud_provider, workspace_name)
    if database_instance is not None:
        endpoint = database_instance['Endpoint']
        managed_cloud_database_info = {
            CLOUDTIK_MANAGED_CLOUD_DATABASE_ENDPOINT: endpoint['Address'],
            CLOUDTIK_MANAGED_CLOUD_DATABASE_PORT: endpoint['Port']
        }
        info[CLOUDTIK_MANAGED_CLOUD_DATABASE] = managed_cloud_database_info


def update_aws_workspace(
        config,
        delete_managed_storage: bool = False,
        delete_managed_database: bool = False):
    workspace_name = config["workspace_name"]
    managed_cloud_storage = is_managed_cloud_storage(config)
    managed_cloud_database = is_managed_cloud_database(config)

    current_step = 1
    total_steps = AWS_WORKSPACE_NUM_UPDATE_STEPS
    if managed_cloud_storage or delete_managed_storage:
        total_steps += 1
    if managed_cloud_database or delete_managed_database:
        total_steps += 1

    try:
        with cli_logger.group("Updating workspace: {}", workspace_name):
            with cli_logger.group(
                    "Updating workspace firewalls",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                update_workspace_firewalls(config)

            if managed_cloud_storage:
                with cli_logger.group(
                        "Creating managed cloud storage...",
                        _numbered=("[]", current_step, total_steps)):
                    current_step += 1
                    _create_workspace_cloud_storage(config, workspace_name)
            else:
                if delete_managed_storage:
                    with cli_logger.group(
                            "Deleting managed cloud storage",
                            _numbered=("[]", current_step, total_steps)):
                        current_step += 1
                        _delete_workspace_cloud_storage(config, workspace_name)

            if managed_cloud_database:
                with cli_logger.group(
                        "Creating managed database",
                        _numbered=("[]", current_step, total_steps)):
                    current_step += 1
                    _create_workspace_cloud_database(config, workspace_name)
            else:
                if delete_managed_database:
                    with cli_logger.group(
                            "Deleting managed database",
                            _numbered=("[]", current_step, total_steps)):
                        current_step += 1
                        _delete_workspace_cloud_database(config, workspace_name)

    except Exception as e:
        cli_logger.error("Failed to update workspace with the name {}. "
                         "You need to delete and try create again. {}", workspace_name, str(e))
        raise e

    cli_logger.success(
        "Successfully updated workspace: {}.",
        cf.bold(workspace_name))


def update_workspace_firewalls(config):
    ec2_client = _resource_client("ec2", config)
    workspace_name = config["workspace_name"]
    vpc_id = get_workspace_vpc_id(workspace_name, ec2_client)
    if vpc_id is None:
        raise RuntimeError("The workspace: {} doesn't exist!".format(config["workspace_name"]))

    try:
        cli_logger.print("Updating the firewalls of workspace...")
        _update_security_group(config, vpc_id)
    except Exception as e:
        cli_logger.error(
            "Failed to update the firewalls of workspace {}. {}", workspace_name, str(e))
        raise e

    cli_logger.print(
        "Successfully updated the firewalls of workspace: {}.",
        cf.bold(workspace_name))


def delete_aws_workspace(
        config,
        delete_managed_storage: bool = False,
        delete_managed_database: bool = False):
    ec2 = _resource("ec2", config)
    ec2_client = _resource_client("ec2", config)
    workspace_name = config["workspace_name"]
    use_peering_vpc = is_use_peering_vpc(config)
    managed_cloud_storage = is_managed_cloud_storage(config)
    managed_cloud_database = is_managed_cloud_database(config)
    vpc_id = get_workspace_vpc_id(workspace_name, ec2_client)

    current_step = 1
    total_steps = AWS_WORKSPACE_NUM_DELETION_STEPS
    if vpc_id is None:
        total_steps = 1
    else:
        if use_peering_vpc:
            total_steps += 1
    if managed_cloud_storage and delete_managed_storage:
        total_steps += 1
    if managed_cloud_database and delete_managed_database:
        total_steps += 1

    try:
        with cli_logger.group("Deleting workspace: {}", workspace_name):
            # Delete in a reverse way of creating
            if managed_cloud_storage and delete_managed_storage:
                with cli_logger.group(
                        "Deleting S3 bucket",
                        _numbered=("[]", current_step, total_steps)):
                    current_step += 1
                    _delete_workspace_cloud_storage(config, workspace_name)

            if managed_cloud_database and delete_managed_database:
                with cli_logger.group(
                        "Deleting managed database",
                        _numbered=("[]", current_step, total_steps)):
                    current_step += 1
                    _delete_workspace_cloud_database(config, workspace_name)

            with cli_logger.group(
                    "Deleting instance profile",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _delete_workspace_instance_profile(config, workspace_name)

            if vpc_id:
                _delete_network_resources(config, workspace_name,
                                          ec2, ec2_client, vpc_id,
                                          current_step, total_steps)

    except Exception as e:
        cli_logger.error(
            "Failed to delete workspace {}. {}", workspace_name, str(e))
        raise e

    cli_logger.success(
            "Successfully deleted workspace: {}.",
            cf.bold(workspace_name))
    return None


def _delete_workspace_instance_profile(config, workspace_name):
    current_step = 1
    total_steps = 2

    with cli_logger.group(
            "Deleting instance profile for head",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_instance_profile_for_head(config, workspace_name)

    with cli_logger.group(
            "Deleting instance profile for worker",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_instance_profile_for_worker(config, workspace_name)


def _delete_instance_profile_for_head(config, workspace_name):
    head_instance_profile_name = _get_head_instance_profile_name(workspace_name)
    head_instance_role_name = "cloudtik-{}-head-role".format(workspace_name)
    _delete_instance_profile(config, head_instance_profile_name, head_instance_role_name)


def _delete_instance_profile_for_worker(config, workspace_name):
    worker_instance_profile_name = _get_worker_instance_profile_name(workspace_name)
    worker_instance_role_name = "cloudtik-{}-worker-role".format(workspace_name)
    _delete_instance_profile(config, worker_instance_profile_name, worker_instance_role_name)


def _delete_workspace_cloud_database(config, workspace_name):
    _delete_managed_cloud_database(
        config["provider"], workspace_name)


def _delete_managed_cloud_database(provider_config, workspace_name):
    current_step = 1
    total_steps = 2

    with cli_logger.group(
            "Deleting managed database instance",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_managed_database_instance(provider_config, workspace_name)

    with cli_logger.group(
            "Deleting DB subnet group",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_db_subnet_group(provider_config, workspace_name)


def _delete_db_subnet_group(provider_config, workspace_name):
    db_subnet_group = get_workspace_db_subnet_group(provider_config, workspace_name)
    if db_subnet_group is None:
        cli_logger.print("No DB subnet group for the workspace were found. Skip deletion.")
        return

    rds_client = _make_client("rds", provider_config)
    try:
        db_subnet_group_name = db_subnet_group.get("DBSubnetGroupName")
        cli_logger.print("Deleting DB subnet group: {}...".format(db_subnet_group_name))
        rds_client.delete_db_subnet_group(
            DBSubnetGroupName=db_subnet_group_name
        )
        cli_logger.print("Successfully deleted DB subnet group: {}.".format(db_subnet_group_name))
    except boto3.exceptions.Boto3Error as e:
        cli_logger.error("Failed to delete DB subnet group. {}", str(e))
        raise e


def _delete_managed_database_instance(provider_config, workspace_name):
    rds_client = _make_client("rds", provider_config)
    db_instance = get_managed_database_instance(provider_config, workspace_name)
    if db_instance is None:
        cli_logger.warning("No managed database instance were found for workspace. Skip deletion.")
        return

    try:
        db_instance_identifier = db_instance.get("DBInstanceIdentifier")
        cli_logger.print("Deleting database instance: {}...".format(db_instance_identifier))
        rds_client.delete_db_instance(
            DBInstanceIdentifier=db_instance_identifier,
            SkipFinalSnapshot=True
        )
        wait_db_instance_deletion(rds_client, db_instance_identifier)
        cli_logger.print("Successfully deleted database instance: {}.".format(db_instance_identifier))
    except boto3.exceptions.Boto3Error as e:
        cli_logger.error("Failed to delete database instance. {}", str(e))
        raise e


def _delete_workspace_cloud_storage(config, workspace_name):
    _delete_managed_cloud_storage(config["provider"], workspace_name)


def _delete_managed_cloud_storage(cloud_provider, workspace_name):
    bucket = get_managed_s3_bucket(cloud_provider, workspace_name)
    if bucket is None:
        cli_logger.warning("No S3 bucket with the name found.")
        return

    try:
        cli_logger.print("Deleting S3 bucket: {}...".format(bucket.name))
        bucket.objects.all().delete()
        bucket.delete()
        cli_logger.print("Successfully deleted S3 bucket: {}.".format(bucket.name))
    except boto3.exceptions.Boto3Error as e:
        cli_logger.error("Failed to delete S3 bucket. {}", str(e))
        raise e


def _delete_network_resources(config, workspace_name,
                              ec2, ec2_client, vpc_id,
                              current_step, total_steps):
    use_working_vpc = is_use_working_vpc(config)
    use_peering_vpc = is_use_peering_vpc(config)

    """
         Do the work - order of operation:
         Delete vpc peering connection
         Delete private subnets
         Delete route-tables for private subnets
         Delete nat-gateway for private subnets
         Delete public subnets
         Delete internet gateway
         Delete security group
         Delete VPC endpoint for S3
         Delete vpc
    """

    # delete vpc peering connection
    if use_peering_vpc:
        with cli_logger.group(
                "Deleting VPC peering connection",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _delete_workspace_vpc_peering_connection_and_routes(config, ec2, ec2_client)

    # delete private subnets
    with cli_logger.group(
            "Deleting private subnet",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_private_subnets(workspace_name, ec2, vpc_id)

    # delete route tables for private subnets
    with cli_logger.group(
            "Deleting route table",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_route_table(workspace_name, ec2, vpc_id)

    # delete nat-gateway
    with cli_logger.group(
            "Deleting NAT gateway",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_nat_gateways(workspace_name, ec2_client, vpc_id)

    # delete public subnets
    with cli_logger.group(
            "Deleting public subnet",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_public_subnets(workspace_name, ec2, vpc_id)

    # delete internet gateway
    with cli_logger.group(
            "Deleting Internet gateway",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_internet_gateway(workspace_name, ec2, vpc_id)

    # delete security group
    with cli_logger.group(
            "Deleting security group",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_workspace_security_group(config, vpc_id)

    # delete vpc endpoint for s3
    with cli_logger.group(
            "Deleting VPC endpoint for S3",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_vpc_endpoint_for_s3(ec2_client, vpc_id, workspace_name)

    # delete vpc
    with cli_logger.group(
            "Deleting VPC",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        if not use_working_vpc:
            _delete_vpc(ec2, ec2_client, vpc_id)
        else:
            # deleting the tags we created on working vpc
            _delete_vpc_tags(ec2, ec2_client, vpc_id, workspace_name)


def create_aws_workspace(config):
    # create a copy of the input config to modify
    config = copy.deepcopy(config)
    
    # create workspace
    config = _create_workspace(config)

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


def bootstrap_aws(config):
    workspace_name = config.get("workspace_name", "")
    if workspace_name == "":
        raise RuntimeError("Workspace name is not specified in cluster configuration.")

    config = bootstrap_aws_from_workspace(config)
    return config


def bootstrap_aws_from_workspace(config):
    if not check_aws_workspace_integrity(config):
        workspace_name = config["workspace_name"]
        cli_logger.abort("AWS workspace {} doesn't exist or is in wrong state.", workspace_name)

    # create a copy of the input config to modify
    config = copy.deepcopy(config)

    # Used internally to store head IAM role.
    config["head_node"] = {}

    # If a LaunchTemplate is provided, extract the necessary fields for the
    # config stages below.
    config = _configure_from_launch_template(config)

    # The head node needs to have an IAM role that allows it to create further
    # EC2 instances.
    config = _configure_iam_role_from_workspace(config)

    # Set s3.bucket if use_managed_cloud_storage=False
    config = _configure_cloud_storage_from_workspace(config)

    # Set database parameters if use_managed_cloud_database
    config = _configure_cloud_database_from_workspace(config)

    # Configure SSH access, using an existing key pair if possible.
    config = _configure_key_pair(config)

    # Pick a reasonable subnet if not specified by the user.
    config = _configure_subnet_from_workspace(config)

    # Cluster workers should be in a security group that permits traffic within
    # the group, and also SSH access from outside.
    config = _configure_security_group_from_workspace(config)

    # Provide a helpful message for missing AMI.
    config = _configure_ami(config)

    config = _configure_prefer_spot_node(config)
    return config


def bootstrap_aws_workspace(config):
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

    if "security_group" not in provider_config:
        provider_config["security_group"] = {}
    security_group_config = provider_config["security_group"]

    if "IpPermissions" not in security_group_config:
        security_group_config["IpPermissions"] = []
    ip_permissions = security_group_config["IpPermissions"]
    ip_permission = {
        "IpProtocol": "tcp",
        "FromPort": 22,
        "ToPort": 22,
        "IpRanges": [{"CidrIp": allowed_ssh_source} for allowed_ssh_source in allowed_ssh_sources]
    }
    ip_permissions.append(ip_permission)


def get_workspace_head_nodes(config):
    return _get_workspace_head_nodes(
        config["provider"], config["workspace_name"])


def _get_workspace_head_nodes(provider_config, workspace_name):
    ec2 = _make_resource("ec2", provider_config)
    ec2_client = _make_resource_client("ec2", provider_config)
    vpc_id = get_workspace_vpc_id(workspace_name, ec2_client)
    if vpc_id is None:
        raise RuntimeError(
            "Failed to get the VPC. The workspace {} doesn't exist or is in the wrong state.".format(
                workspace_name
            ))

    filters = [
        {
           "Name": "vpc-id",
            "Values": [vpc_id]
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


def _configure_cloud_storage_from_workspace(config):
    use_managed_cloud_storage = is_use_managed_cloud_storage(config)
    if use_managed_cloud_storage:
        _configure_managed_cloud_storage_from_workspace(config, config["provider"])

    return config


def _configure_managed_cloud_storage_from_workspace(config, cloud_provider):
    workspace_name = config["workspace_name"]
    s3_bucket = get_managed_s3_bucket(cloud_provider, workspace_name)
    if s3_bucket is None:
        cli_logger.abort("No managed s3 bucket was found. If you want to use managed s3 bucket, "
                         "you should set managed_cloud_storage equal to True when you creating workspace.")

    cloud_storage = get_aws_s3_storage_config_for_update(config["provider"])
    cloud_storage[AWS_S3_BUCKET] = s3_bucket.name


def _configure_cloud_database_from_workspace(config):
    use_managed_cloud_database = is_use_managed_cloud_database(config)
    if use_managed_cloud_database:
        _configure_managed_cloud_database_from_workspace(
            config, config["provider"])

    return config


def _configure_managed_cloud_database_from_workspace(config, cloud_provider):
    workspace_name = config["workspace_name"]
    database_instance = get_managed_database_instance(cloud_provider, workspace_name)
    if database_instance is None:
        cli_logger.abort("No managed database was found. If you want to use managed database, "
                         "you should set managed_cloud_database equal to True when you creating workspace.")

    endpoint = database_instance['Endpoint']
    database_config = get_aws_database_config_for_update(config["provider"])
    database_config[AWS_DATABASE_ENDPOINT] = endpoint['Address']
    database_config["port"] = endpoint['Port']
    if "username" not in database_config:
        database_config["username"] = database_instance['MasterUsername']


def _configure_iam_role_from_workspace(config):
    worker_role_for_cloud_storage = is_worker_role_for_cloud_storage(config)
    if worker_role_for_cloud_storage:
        return _configure_iam_role_for_cluster(config)
    else:
        return _configure_iam_role_for_head(config)


def _configure_iam_role_for_head(config):
    head_node_type = config["head_node_type"]
    head_node_config = config["available_node_types"][head_node_type][
        "node_config"]
    if "IamInstanceProfile" in head_node_config:
        _set_config_info(head_instance_profile_src="config")
        return config
    _set_config_info(head_instance_profile_src="workspace")

    instance_profile_name = _get_head_instance_profile_name(
        config["workspace_name"])
    profile = _get_instance_profile(instance_profile_name, config)
    if not profile:
        raise RuntimeError("Workspace instance profile: {} not found!".format(instance_profile_name))

    # Add IAM role to "head_node" field so that it is applied only to
    # the head node -- not to workers with the same node type as the head.
    config["head_node"]["IamInstanceProfile"] = {"Arn": profile.arn}

    return config


def _configure_iam_role_for_cluster(config):
    _set_config_info(head_instance_profile_src="workspace")

    head_instance_profile_name = _get_head_instance_profile_name(
        config["workspace_name"])
    head_profile = _get_instance_profile(head_instance_profile_name, config)

    worker_instance_profile_name = _get_worker_instance_profile_name(
        config["workspace_name"])
    worker_profile = _get_instance_profile(worker_instance_profile_name, config)

    if not head_profile:
        raise RuntimeError("Workspace head instance profile: {} not found!".format(head_instance_profile_name))
    if not worker_profile:
        raise RuntimeError("Workspace worker instance profile: {} not found!".format(worker_instance_profile_name))

    for key, node_type in config["available_node_types"].items():
        node_config = node_type["node_config"]
        if key == config["head_node_type"]:
            node_config["IamInstanceProfile"] = {"Arn": head_profile.arn}
        else:
            node_config["IamInstanceProfile"] = {"Arn": worker_profile.arn}

    return config


def _create_or_update_instance_profile(config, instance_profile_name, instance_role_name, is_head=True):
    profile = _get_instance_profile(instance_profile_name, config)

    if profile is None:
        cli_logger.verbose(
            "Creating new IAM instance profile {} for use as the default.",
            cf.bold(instance_profile_name))
        client = _resource_client("iam", config)
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
            if is_head:
                attach_policy_arns = [
                    "arn:aws:iam::aws:policy/AmazonEC2FullAccess",
                    "arn:aws:iam::aws:policy/AmazonS3FullAccess",
                    "arn:aws:iam::aws:policy/IAMFullAccess"
                ]
            else:
                attach_policy_arns = [
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

    client = _resource_client("iam", config)
    # Deletes the specified instance profile. The instance profile must not have an associated role.
    client.delete_instance_profile(
        InstanceProfileName=instance_profile_name)

    cli_logger.print("Successfully deleted instance profile.")


def _delete_instance_profile_role(config, instance_profile_role):
    _delete_iam_role(config["provider"], instance_profile_role)


def _delete_iam_role(cloud_provider, role_name):
    role = _get_iam_role(role_name, cloud_provider)
    if role is None:
        cli_logger.print("IAM role {} doesn't exist", role_name)
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


def _delete_internet_gateway(workspace_name, ec2, vpc_id):
    """ Detach and delete the internet-gateway """
    igws = get_workspace_internet_gateways(workspace_name, ec2, vpc_id)

    if len(igws) == 0:
        cli_logger.print("No Internet Gateways for workspace were found under this VPC: {}...".format(vpc_id))
        return
    for igw in igws:
        try:
            cli_logger.print("Detaching and deleting Internet Gateway: {}...".format(igw.id))
            igw.detach_from_vpc(VpcId=vpc_id)
            igw.delete()
        except boto3.exceptions.Boto3Error as e:
            cli_logger.error("Failed to detach or delete Internet Gateway. {}", str(e))
            raise e


def _delete_private_subnets(workspace_name, ec2, vpc_id):
    """ Delete custom private subnets """
    subnets = get_workspace_private_subnets(workspace_name, ec2, vpc_id)

    if len(subnets) == 0:
        cli_logger.print("No subnets for workspace were found under this VPC: {}...".format(vpc_id))
        return
    try:
        for subnet in subnets:
            cli_logger.print("Deleting private subnet: {}...".format(subnet.id))
            subnet.delete()
            cli_logger.print("Successfully deleted private subnet: {}.".format(subnet.id))
    except boto3.exceptions.Boto3Error as e:
        cli_logger.error("Failed to delete private subnet. {}", str(e))
        raise e


def _delete_public_subnets(workspace_name, ec2, vpc_id):
    """ Delete custom public subnets """
    subnets = get_workspace_public_subnets(workspace_name, ec2, vpc_id)

    if len(subnets) == 0:
        cli_logger.print("No subnets for workspace were found under this VPC: {}...".format(vpc_id))
        return
    try:
        for subnet in subnets:
            cli_logger.print("Deleting public subnet: {}...".format(subnet.id))
            subnet.delete()
            cli_logger.print("Successfully deleted public subnet: {}.".format(subnet.id))
    except boto3.exceptions.Boto3Error as e:
        cli_logger.error("Failed to delete public subnet. {}", str(e))
        raise e


def _delete_route_table(workspace_name, ec2, vpc_id):
    """ Delete the route-tables for private subnets """
    rtbs = get_workspace_private_route_tables(workspace_name, ec2, vpc_id)
    if len(rtbs) == 0:
        cli_logger.print("No route tables for workspace were found under this VPC: {}...".format(vpc_id))
        return
    try:
        for rtb in rtbs:
            cli_logger.print("Deleting route table: {}...".format(rtb.id))
            table = ec2.RouteTable(rtb.id)
            table.delete()
            cli_logger.print("Successfully deleted route table: {}.".format(rtb.id))
    except boto3.exceptions.Boto3Error as e:
        cli_logger.error("Failed to delete route table. {}", str(e))
        raise e


def release_elastic_ip_address(ec2_client, allocation_id, retry=5):
    while retry > 0:
        try:
            ec2_client.release_address(AllocationId=allocation_id)
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


def _delete_nat_gateways(workspace_name, ec2_client, vpc_id):
    """ Remove nat-gateway and release elastic IP """
    nat_gateways = get_workspace_nat_gateways(workspace_name, ec2_client, vpc_id)
    if len(nat_gateways) == 0:
        cli_logger.print("No NAT Gateways for workspace were found under this VPC: {}...".format(vpc_id))
        return

    for nat in nat_gateways:
        _delete_nat_gateway(nat, ec2_client)


def _delete_nat_gateway_resource(nat, ec2_client):
    try:
        cli_logger.print("Deleting NAT Gateway: {}...".format(nat["NatGatewayId"]))
        ec2_client.delete_nat_gateway(NatGatewayId=nat["NatGatewayId"])
        cli_logger.print("Successfully deleted NAT Gateway: {}.".format(nat["NatGatewayId"]))
    except boto3.exceptions.Boto3Error as e:
        cli_logger.error("Failed to delete NAT Gateway. {}", str(e))
        raise e


def _delete_elastic_ip_address(nat, ec2_client):
    cli_logger.print("Releasing elastic ip address : {}...".format(nat["NatGatewayAddresses"][0]["AllocationId"]))
    release_elastic_ip_address(ec2_client, nat["NatGatewayAddresses"][0]["AllocationId"])
    cli_logger.print("Successfully released elastic ip address for NAT Gateway.")


def _delete_nat_gateway(nat, ec2_client):
    current_step = 1
    total_steps = 2

    with cli_logger.group(
            "Deleting NAT Gateway",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_nat_gateway_resource(nat, ec2_client)

    with cli_logger.group(
            "Releasing elastic ip address",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_elastic_ip_address(nat, ec2_client)


def _delete_workspace_security_group(config, vpc_id):
    security_group_name = SECURITY_GROUP_TEMPLATE.format(config["workspace_name"])
    _delete_security_group(
        config["provider"], vpc_id, security_group_name
    )


def _delete_security_group(provider_config, vpc_id, security_group_name):
    """ Delete any security-groups """
    sg = _get_security_group(
        provider_config, vpc_id, security_group_name)
    if sg is None:
        cli_logger.print("No security groups for workspace were found under this VPC: {}...".format(vpc_id))
        return
    try:
        cli_logger.print("Deleting security group: {}...".format(sg.id))
        sg.delete()
        cli_logger.print("Successfully deleted security group: {}.".format(sg.id))
    except boto3.exceptions.Boto3Error as e:
        cli_logger.error("Failed to delete security group. {}", str(e))
        raise e


def _get_vpc(ec2, ec2_client, vpc_id):
    vpc_ids = [vpc["VpcId"] for vpc in ec2_client.describe_vpcs()["Vpcs"] if vpc["VpcId"] == vpc_id]
    if len(vpc_ids) == 0:
        return None

    vpc = ec2.Vpc(vpc_id)
    return vpc


def _delete_vpc(ec2, ec2_client, vpc_id):
    """ Delete the VPC """
    vpc = _get_vpc(ec2, ec2_client, vpc_id)
    if vpc is None:
        cli_logger.print("The VPC: {} doesn't exist.".format(vpc_id))
        return
    try:
        cli_logger.print("Deleting VPC: {}...".format(vpc.id))
        vpc.delete()
        cli_logger.print("Successfully deleted VPC: {}.".format(vpc.id))
    except Exception as e:
        cli_logger.error("Failed to delete VPC. {}", str(e))
        raise e


def _delete_vpc_tags(ec2, ec2_client, vpc_id, workspace_name):
    vpc = _get_vpc(ec2, ec2_client, vpc_id)
    if vpc is None:
        cli_logger.print("The VPC: {} doesn't exist.".format(vpc_id))
        return
    try:
        cli_logger.print("Deleting VPC tags: {}...".format(vpc.id))
        vpc_name = get_workspace_vpc_name(workspace_name)
        ec2_client.delete_tags(
            Resources=[vpc_id],
            Tags=[
                {'Key': 'Name', 'Value': vpc_name},
                {'Key': AWS_WORKSPACE_VERSION_TAG_NAME}
            ]
        )
        cli_logger.print("Successfully deleted VPC tags: {}.".format(vpc.id))
    except Exception as e:
        cli_logger.error("Failed to delete VPC tags. {}", str(e))
        raise e


def _delete_vpc_endpoint_for_s3(ec2_client, vpc_id, workspace_name):
    endpoint_ids = [endpoint['VpcEndpointId'] for endpoint in get_vpc_endpoint_for_s3(ec2_client, vpc_id, workspace_name)]
    if len(endpoint_ids) == 0:
        cli_logger.print("No VPC endpoint for S3 was found in workspace.")
        return
    try:
        cli_logger.print("Deleting VPC endpoint for S3: {}...".format(endpoint_ids))
        ec2_client.delete_vpc_endpoints(
                        VpcEndpointIds=endpoint_ids
                        )
        cli_logger.print("Successfully deleted VPC endpoint for S3: {}.".format(endpoint_ids))
    except Exception as e:
        cli_logger.error("Failed to delete VPC endpoint for S3. {}", str(e))
        raise e


def _delete_workspace_vpc_peering_connection_and_routes(config, ec2, ec2_client):
    current_step = 1
    total_steps = 2

    with cli_logger.group(
            "Deleting routes for VPC peering connection",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_routes_for_workspace_vpc_peering_connection(config, ec2, ec2_client)

    with cli_logger.group(
            "Deleting VPC peering connection",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_workspace_vpc_peering_connection(config, ec2_client)


def _delete_routes_for_workspace_vpc_peering_connection(config, ec2, ec2_client):
    workspace_name = config["workspace_name"]
    current_ec2_client = _working_node_client('ec2', config)
    current_vpc = get_current_vpc(config)
    workspace_vpc = get_workspace_vpc(workspace_name, ec2_client, ec2)
    current_vpc_route_tables = get_vpc_route_tables(current_vpc)
    workspace_vpc_route_tables = get_vpc_route_tables(workspace_vpc)
    vpc_peering_connection = get_workspace_vpc_peering_connection(config)
    if vpc_peering_connection is None:
        cli_logger.print("No VPC peering connection was found in workspace. Skip delete "
                         "routes for workspace vpc peering connection.")
        return
    vpc_peering_connection_id = vpc_peering_connection['VpcPeeringConnectionId']

    for current_vpc_route_table in current_vpc_route_tables:
        if len([route for route in current_vpc_route_table.routes if
                route._destination_cidr_block == workspace_vpc.cidr_block
                and route.vpc_peering_connection_id == vpc_peering_connection_id]) == 0:
            cli_logger.print(
                "No route of current VPC route table {} need to be delete.".format(
                    current_vpc_route_table.id))
            continue
        try:
            current_ec2_client.delete_route(RouteTableId=current_vpc_route_table.id, DestinationCidrBlock=workspace_vpc.cidr_block)
            cli_logger.print(
                "Successfully delete the route about VPC peering connection for current VPC route table {}.".format(
                    current_vpc_route_table.id))
        except Exception as e:
            cli_logger.error(
                "Failed to delete the route about VPC peering connection for current VPC route table. {}", str(e))
            raise e

    for workspace_vpc_route_table in workspace_vpc_route_tables:
        if len([route for route in workspace_vpc_route_table.routes if
                route._destination_cidr_block == current_vpc.cidr_block
                and route.vpc_peering_connection_id == vpc_peering_connection_id]) == 0:
            cli_logger.print(
                "No route of workspace VPC route table {} need to be delete.".format(
                    workspace_vpc_route_table.id))
            continue
        try:
            ec2_client.delete_route(RouteTableId=workspace_vpc_route_table.id,
                                    DestinationCidrBlock=current_vpc.cidr_block)
            cli_logger.print(
                "Successfully delete the route about VPC peering connection for workspace VPC route table.".format(
                    workspace_vpc_route_table.id))
        except Exception as e:
            cli_logger.error(
                "Failed to delete the route about VPC peering connection for workspace VPC route table. {}", str(e))
            raise e


def _delete_workspace_vpc_peering_connection(config, ec2_client):
    vpc_peering_connection = get_workspace_vpc_peering_connection(config)
    if vpc_peering_connection is None:
        cli_logger.print("No VPC peering connection was found in workspace.")
        return
    vpc_peering_connection_id = vpc_peering_connection['VpcPeeringConnectionId']
    try:
        cli_logger.print("Deleting VPC peering connection for : {}...".format(vpc_peering_connection_id))
        ec2_client.delete_vpc_peering_connection(
            VpcPeeringConnectionId=vpc_peering_connection_id
        )
        waiter = ec2_client.get_waiter('vpc_peering_connection_deleted')
        waiter.wait(VpcPeeringConnectionIds=[vpc_peering_connection_id])
        cli_logger.print("Successfully deleted VPC peering connection for: {}.".format(vpc_peering_connection_id))
    except Exception as e:
        cli_logger.error("Failed to delete VPC peering connection. {}", str(e))
        raise e


def _create_vpc(config, ec2, ec2_client):
    workspace_name = config["workspace_name"]
    vpc_name = get_workspace_vpc_name(workspace_name)
    cli_logger.print("Creating workspace VPC: {}...", vpc_name)
    # create vpc
    cidr_block = '10.0.0.0/16'
    if is_use_peering_vpc(config):
        current_vpc = get_current_vpc(config)
        cidr_block = _configure_peering_vpc_cidr_block(current_vpc)

    try:
        vpc = ec2.create_vpc(
            CidrBlock=cidr_block,
            TagSpecifications=[
                {
                    'ResourceType': 'vpc',
                    'Tags': [
                        {
                            'Key': 'Name',
                            'Value': vpc_name
                        },
                        {
                            'Key': AWS_WORKSPACE_VERSION_TAG_NAME,
                            'Value': AWS_WORKSPACE_VERSION_CURRENT
                        },
                    ]
                },
            ]
        )

        waiter = ec2_client.get_waiter('vpc_exists')
        waiter.wait(VpcIds=[vpc.id])

        vpc.modify_attribute(EnableDnsSupport={'Value': True})
        vpc.modify_attribute(EnableDnsHostnames={'Value': True})
        cli_logger.print("Successfully created workspace VPC: {}.", vpc_name)
    except Exception as e:
        cli_logger.error("Failed to create workspace VPC. {}", str(e))
        raise e

    return vpc


def get_existing_routes_cidr_block(route_tables):
    existing_routes_cidr_block = set()
    for route_table in route_tables:
        for route in route_table.routes:
            if route._destination_cidr_block != '0.0.0.0/0':
                existing_routes_cidr_block.add(route._destination_cidr_block)
    return existing_routes_cidr_block


def _configure_peering_vpc_cidr_block(current_vpc):
    current_vpc_cidr_block = current_vpc.cidr_block
    current_vpc_route_tables = get_vpc_route_tables(current_vpc)

    existing_routes_cidr_block = get_existing_routes_cidr_block(current_vpc_route_tables)
    existing_routes_cidr_block.add(current_vpc_cidr_block)

    ip = current_vpc_cidr_block.split("/")[0].split(".")

    for  i in range(0, 256):
        tmp_cidr_block = ip[0] + "." + str(i) + ".0.0/16"

        if check_cidr_conflict(tmp_cidr_block, existing_routes_cidr_block):
            cli_logger.print("Successfully found cidr block for peering VPC.")
            return tmp_cidr_block

    cli_logger.abort("Failed to find non-conflicted cidr block for peering VPC.")


def _describe_availability_zones(ec2_client):
    response = ec2_client.describe_availability_zones()
    availability_zones = [zone["ZoneName"] for zone in response['AvailabilityZones']
                          if zone["State"] == 'available' and zone["ZoneType"] == 'availability-zone']
    return availability_zones


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


def _create_and_configure_subnets(config, ec2_client, vpc):
    workspace_name = config["workspace_name"]
    subnets = []
    cidr_list = _configure_subnets_cidr(vpc)
    cidr_len = len(cidr_list)

    availability_zones = set(_describe_availability_zones(ec2_client))
    used_availability_zones = set()
    default_availability_zone = None
    last_availability_zone = None

    for i in range(0, cidr_len):
        cidr_block = cidr_list[i]
        subnet_type = "public" if i == 0 else "private"
        with cli_logger.group(
                "Creating {} subnet", subnet_type,
                _numbered=("()", i + 1, cidr_len)):
            try:
                if i == 0:
                    subnet = _create_public_subnet(workspace_name, vpc, cidr_block)
                    default_availability_zone = subnet.availability_zone
                else:
                    if last_availability_zone is None:
                        last_availability_zone = default_availability_zone

                    subnet = _create_private_subnet(workspace_name, vpc, cidr_block,
                                                    last_availability_zone)

                    last_availability_zone = _next_availability_zone(
                        availability_zones, used_availability_zones, last_availability_zone)

            except Exception as e:
                cli_logger.error("Failed to create {} subnet. {}", subnet_type, str(e))
                raise e
            subnets.append(subnet)

    assert len(subnets) == AWS_VPC_SUBNETS_COUNT, "We must create {} subnets for VPC: {}!".format(
        AWS_VPC_SUBNETS_COUNT, vpc.id)
    return subnets


def _create_public_subnet(workspace_name, vpc, cidr_block):
    cli_logger.print("Creating public subnet for VPC: {} with CIDR: {}...".format(vpc.id, cidr_block))
    subnet_name = 'cloudtik-{}-public-subnet'.format(workspace_name)
    subnet = vpc.create_subnet(
        CidrBlock=cidr_block,
        TagSpecifications=[
            {
                'ResourceType': 'subnet',
                'Tags': [
                    {
                        'Key': 'Name',
                        'Value': subnet_name
                    },
                ]
            },
        ]
    )
    subnet.meta.client.modify_subnet_attribute(SubnetId=subnet.id,
                                               MapPublicIpOnLaunch={"Value": True})
    cli_logger.print("Successfully created public subnet: {}.".format(subnet_name))
    return subnet


def _create_private_subnet(workspace_name, vpc, cidr_block, availability_zone):
    if availability_zone is None:
        cli_logger.print("Creating private subnet for VPC: {} with CIDR: {}...".format(
            vpc.id, cidr_block))
    else:
        cli_logger.print("Creating private subnet for VPC: {} with CIDR: {} in {}...".format(
            vpc.id, cidr_block, availability_zone))

    subnet_name = 'cloudtik-{}-private-subnet'.format(workspace_name)
    tag_specs = [
            {
                'ResourceType': 'subnet',
                'Tags': [
                    {
                        'Key': 'Name',
                        'Value': subnet_name
                    },
                ]
            },
        ]
    if availability_zone is None:
        subnet = vpc.create_subnet(
            CidrBlock=cidr_block,
            TagSpecifications=tag_specs
        )
    else:
        subnet = vpc.create_subnet(
            CidrBlock=cidr_block,
            TagSpecifications=tag_specs,
            AvailabilityZone=availability_zone
        )
    cli_logger.print("Successfully created private subnet: {}.".format(subnet_name))
    return subnet


def _create_internet_gateway(config, ec2, ec2_client, vpc):
    cli_logger.print("Creating Internet Gateway for the VPC: {}...".format(vpc.id))
    try:
        igw = ec2.create_internet_gateway(
            TagSpecifications=[
                {
                    'ResourceType': 'internet-gateway',
                    'Tags': [
                        {
                            'Key': 'Name',
                            'Value': 'cloudtik-{}-internet-gateway'.format(config["workspace_name"])
                        },
                    ]
                },
            ]
        )

        waiter = ec2_client.get_waiter('internet_gateway_exists')
        waiter.wait(InternetGatewayIds=[igw.id])

        igw.attach_to_vpc(VpcId=vpc.id)
        cli_logger.print("Successfully created Internet Gateway: cloudtik-{}-internet-gateway.".format(config["workspace_name"]))
    except Exception as e:
        cli_logger.error("Failed to create Internet Gateway. {}", str(e))
        try:
            cli_logger.print("Try to find the existing Internet Gateway...")
            igws = [igw for igw in vpc.internet_gateways.all()]
        except Exception:
            raise e

        if len(igws) > 0:
            igw = igws[0]
            cli_logger.print("Existing internet gateway found. Will use this one.")
        else:
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


def wait_db_instance_creation(rds_client, db_instance_identifier):
    """
    Check if successful state is reached every 30 seconds until a successful state is reached.
    An error is returned after 40 failed checks.
    """
    try:
        waiter = rds_client.get_waiter('db_instance_available')
        waiter.wait(
            DBInstanceIdentifier=db_instance_identifier,
            WaiterConfig={
                'Delay': 30,
                'MaxAttempts': 40
            }
        )
    except Exception as e:
        cli_logger.abort('Could not create the database instance.')
        raise


def wait_db_instance_deletion(rds_client, db_instance_identifier):
    """
    Check if successful state is reached every 30 seconds until a successful state is reached.
    An error is returned after 40 failed checks.
    """
    try:
        waiter = rds_client.get_waiter('db_instance_deleted')
        waiter.wait(
            DBInstanceIdentifier=db_instance_identifier,
            WaiterConfig={
                'Delay': 30,
                'MaxAttempts': 40
            }
        )
    except Exception as e:
        cli_logger.abort('Could not create the database instance.')
        raise


def _create_and_configure_nat_gateway(
        config, ec2_client, vpc, subnet, private_route_table):
    current_step = 1
    total_steps = 2

    with cli_logger.group(
            "Creating NAT Gateway",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        nat_gateway = _create_nat_gateway(config, ec2_client, vpc, subnet)

    with cli_logger.group(
            "Configuring NAT Gateway route for private subnet",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _configure_nat_gateway_route_for_private_subnet(
            ec2_client, nat_gateway, private_route_table)


def _configure_nat_gateway_route_for_private_subnet(
        ec2_client, nat_gateway, private_route_table):
    cli_logger.print("Configuring NAT Gateway route for private subnet...")
    # Create a default route pointing to NAT Gateway for private subnets
    ec2_client.create_route(RouteTableId=private_route_table.id, DestinationCidrBlock='0.0.0.0/0',
                            NatGatewayId=nat_gateway['NatGatewayId'])
    cli_logger.print("Successfully configured NAT Gateway route for private subnet.")


def _create_nat_gateway(config, ec2_client, vpc, subnet):
    cli_logger.print("Creating NAT Gateway for subnet: {}...".format(subnet.id))
    try:
        eip_for_nat_gateway = _create_elastic_ip(ec2_client)
        allocation_id = eip_for_nat_gateway['AllocationId']

        # create NAT Gateway and associate with Elastic IP
        nat_gw = ec2_client.create_nat_gateway(
            SubnetId=subnet.id,
            AllocationId=allocation_id,
            TagSpecifications=[
                {
                    'ResourceType': 'natgateway',
                    'Tags': [
                        {
                            'Key': 'Name',
                            'Value': 'cloudtik-{}-nat-gateway'.format(config["workspace_name"])
                        },
                    ]
                },
            ]
        )['NatGateway']
        nat_gw_id = nat_gw['NatGatewayId']
        wait_nat_creation(ec2_client, nat_gw_id)
        cli_logger.print("Successfully created NAT Gateway: cloudtik-{}-nat-gateway.".format(config["workspace_name"]))
    except Exception as e:
        cli_logger.error("Failed to create NAT Gateway. {}", str(e))
        try:
            cli_logger.print("Try to find the existing NAT Gateway...")
            nat_gws = get_vpc_nat_gateways(ec2_client, vpc.id)
        except Exception:
            raise e

        if len(nat_gws) > 0:
            nat_gw = nat_gws[0]
            cli_logger.print("Found an existing NAT Gateway. Will use this one")
        else:
            raise e

    return nat_gw


def _create_and_configure_vpc_peering_connection(config, ec2, ec2_client):
    current_step = 1
    total_steps = 3

    with cli_logger.group(
            "Creating VPC peering connection",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_workspace_vpc_peering_connection(config, ec2, ec2_client)

    with cli_logger.group(
            "Accepting VPC peering connection",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _accept_workspace_vpc_peering_connection(config, ec2_client)

    with cli_logger.group(
            "Update route tables for the VPC peering connection",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _update_route_tables_for_workspace_vpc_peering_connection(config, ec2, ec2_client)


def _update_route_tables_for_workspace_vpc_peering_connection(config, ec2, ec2_client):
    current_ec2_client = _working_node_client('ec2', config)
    workspace_name = config["workspace_name"]
    current_vpc = get_current_vpc(config)
    workspace_vpc = get_workspace_vpc(workspace_name, ec2_client, ec2)
    current_vpc_route_tables = get_vpc_route_tables(current_vpc)
    workspace_vpc_route_tables = get_vpc_route_tables(workspace_vpc)

    vpc_peering_connection = get_workspace_vpc_peering_connection(config)
    if vpc_peering_connection is None:
        cli_logger.abort(
            "No vpc_peering_connection found for workspace: {}.".format(workspace_name))

    for current_vpc_route_table in current_vpc_route_tables:
        try:
            current_ec2_client.create_route(
                RouteTableId=current_vpc_route_table.id, DestinationCidrBlock=workspace_vpc.cidr_block,
                VpcPeeringConnectionId=vpc_peering_connection['VpcPeeringConnectionId'])
            cli_logger.print(
                "Successfully add route destination to current VPC route table {} with workspace VPC CIDR block.".format(
                    current_vpc_route_table.id))
        except Exception as e:
            cli_logger.error(
                "Failed to add route destination to current VPC route table with workspace VPC CIDR block. {}", str(e))
            raise e

    for workspace_vpc_route_table in workspace_vpc_route_tables:
        try:
            ec2_client.create_route(
                RouteTableId=workspace_vpc_route_table.id, DestinationCidrBlock=current_vpc.cidr_block,
                VpcPeeringConnectionId=vpc_peering_connection['VpcPeeringConnectionId'])
            cli_logger.print(
                "Successfully add route destination to workspace VPC route table {} with current VPC CIDR block.".format(
                    workspace_vpc_route_table.id))
        except Exception as e:
            cli_logger.error(
                "Failed to add route destination to workspace VPC route table with current VPC CIDR block. {}", str(e))
            raise e


def wait_for_workspace_vpc_peering_connection_active(config):
    workspace_name = config["workspace_name"]
    vpc_peer_name = get_workspace_vpc_peering_name(workspace_name)
    current_ec2_client = _working_node_client('ec2', config)
    retry = 20
    while retry > 0:
        response = current_ec2_client.describe_vpc_peering_connections(Filters=[
            {'Name': 'status-code', 'Values': ['active']},
            {'Name': 'tag:Name', 'Values': [vpc_peer_name]}
        ])
        vpc_peering_connections = response['VpcPeeringConnections']
        if len(vpc_peering_connections) == 0:
            retry = retry - 1
        else:
            return True
        if retry > 0:
            cli_logger.warning(
                "Remaining {} tries to wait for workspace vpc_peering_connection active...".format(retry))
            time.sleep(1)
        else:
            cli_logger.abort("Failed to wait for workspace vpc_peering_connection active.")


def _accept_workspace_vpc_peering_connection(config, ec2_client):
    workspace_name = config["workspace_name"]
    vpc_peer_name = get_workspace_vpc_peering_name(workspace_name)
    pending_vpc_peering_connection = get_workspace_pending_acceptance_vpc_peering_connection(config, ec2_client)
    if pending_vpc_peering_connection is None:
        cli_logger.abort(
            "No pending-acceptance vpc peering connection was found for the workspace: {}.".format(workspace_name))
    try:
        ec2_client.accept_vpc_peering_connection(
            VpcPeeringConnectionId=pending_vpc_peering_connection['VpcPeeringConnectionId'],
        )
        wait_for_workspace_vpc_peering_connection_active(config)
        cli_logger.print(
            "Successfully accepted VPC peering connection: {}.", vpc_peer_name)
    except Exception as e:
        cli_logger.error("Failed to accept VPC peering connection. {}", str(e))
        raise e


def get_workspace_vpc_peering_connection(config):
    workspace_name = config["workspace_name"]
    vpc_peer_name = get_workspace_vpc_peering_name(workspace_name)
    current_ec2_client = _working_node_client('ec2', config)
    response = current_ec2_client.describe_vpc_peering_connections(Filters=[
        {'Name': 'status-code', 'Values': ['active']},
        {'Name': 'tag:Name', 'Values': [vpc_peer_name]}
    ])
    vpc_peering_connections = response['VpcPeeringConnections']
    return None if len(vpc_peering_connections) == 0 else vpc_peering_connections[0]


def get_workspace_pending_acceptance_vpc_peering_connection(config, ec2_client):
    workspace_name = config["workspace_name"]
    vpc_peer_name = get_workspace_vpc_peering_name(workspace_name)
    current_ec2_client = _working_node_client('ec2', config)
    retry = 20
    while retry > 0:
        response = current_ec2_client.describe_vpc_peering_connections(Filters=[
            {'Name': 'status-code', 'Values': ['pending-acceptance']},
            {'Name': 'tag:Name', 'Values': [vpc_peer_name]}
        ])
        vpc_peering_connections = response['VpcPeeringConnections']
        if len(vpc_peering_connections) == 0:
            retry = retry - 1
        else:
            return vpc_peering_connections[0]

        if retry > 0:
            cli_logger.warning("Remaining {} tries to get workspace pending_acceptance vpc_peering_connection...".format(retry))
            time.sleep(1)
        else:
            cli_logger.abort("Failed to get workspace pending_acceptance vpc_peering_connection.")


def _create_workspace_vpc_peering_connection(config, ec2, ec2_client):
    current_ec2_client = _working_node_client('ec2', config)
    workspace_name = config["workspace_name"]
    vpc_peer_name = get_workspace_vpc_peering_name(workspace_name)
    region = config["provider"]["region"]
    current_vpc = get_current_vpc(config)
    workspace_vpc = get_workspace_vpc(workspace_name, ec2_client, ec2)
    cli_logger.print("Creating VPC peering connection.")
    try:
        response = current_ec2_client.create_vpc_peering_connection(
            VpcId=current_vpc.vpc_id,
            PeerVpcId=workspace_vpc.vpc_id,
            PeerRegion=region,
            TagSpecifications=[{'ResourceType': "vpc-peering-connection",
                                "Tags": [{'Key': 'Name',
                                          'Value': vpc_peer_name
                                          }]}],
        )
        vpc_peering_connection_id = response['VpcPeeringConnection']['VpcPeeringConnectionId']
        waiter = current_ec2_client.get_waiter('vpc_peering_connection_exists')
        waiter.wait(VpcPeeringConnectionIds=[vpc_peering_connection_id])

        cli_logger.print(
            "Successfully created VPC peering connection: {}.", vpc_peer_name)
    except Exception as e:
        cli_logger.error("Failed to create VPC peering connection. {}", str(e))
        raise e


def _create_vpc_endpoint_for_s3(config, ec2, ec2_client, vpc):
    cli_logger.print("Creating VPC endpoint for S3: {}...".format(vpc.id))
    try:
        region = config["provider"]["region"]
        workspace_name = config["workspace_name"]
        s3_endpoint_name = get_workspace_vpc_s3_endpoint(workspace_name)
        route_table_ids = _get_workspace_route_table_ids(workspace_name, ec2, vpc.id)

        ec2_client.create_vpc_endpoint(
            VpcEndpointType='Gateway',
            VpcId=vpc.id,
            ServiceName='com.amazonaws.{}.s3'.format(region),
            RouteTableIds=route_table_ids,
            TagSpecifications=[{'ResourceType': "vpc-endpoint",
                                "Tags": [{'Key': 'Name',
                                          'Value': s3_endpoint_name
                                          }]}],
        )
        cli_logger.print(
            "Successfully created VPC endpoint for S3: {}.", s3_endpoint_name)
    except Exception as e:
        cli_logger.error("Failed to create Vpc Endpoint for S3. {}", str(e))
        raise e


def _create_or_update_route_tables(
        config, ec2, ec2_client, vpc, subnets, internet_gateway):
    current_step = 1
    total_steps = 2

    with cli_logger.group(
            "Updating route table for public subnet",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _update_route_table_for_public_subnet(
            config, ec2, ec2_client, vpc, subnets[AWS_VPC_PUBLIC_SUBNET_INDEX], internet_gateway)

    with cli_logger.group(
            "Creating route table for private subnet",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        # create private route table for private subnets
        private_subnets = subnets[1:]
        private_route_table = _create_route_table_for_private_subnets(
            config, ec2, vpc, private_subnets)

    return private_route_table


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
            "Update the rules for route table: {}.".format(public_route_table.id))
        ec2_client.delete_route(RouteTableId=public_route_table.id, DestinationCidrBlock='0.0.0.0/0')
        ec2_client.create_route(RouteTableId=public_route_table.id, DestinationCidrBlock='0.0.0.0/0', GatewayId=igw.id)
    public_route_table.associate_with_subnet(SubnetId=subnet.id)
    cli_logger.print("Successfully updated public subnet route table: {}.".format(subnet.id))


def _create_route_table_for_private_subnets(config, ec2, vpc, subnets):
    route_table_name = 'cloudtik-{}-private-route-table'.format(config["workspace_name"])
    cli_logger.print("Creating private subnet route table: {}...".format(route_table_name))
    private_route_tables = get_workspace_private_route_tables(config["workspace_name"], ec2, vpc.id)
    if len(private_route_tables) > 0:
        private_route_table = private_route_tables[0]
        cli_logger.print("Found existing private subnet route table. Skip creation.")
    else:
        private_route_table = ec2.create_route_table(
            VpcId=vpc.id,
            TagSpecifications=[
                {
                    'ResourceType': 'route-table',
                    'Tags': [
                        {
                            'Key': 'Name',
                            'Value': route_table_name
                        },
                    ]
                },
            ]
        )
        cli_logger.print("Successfully created private subnet route table: {}...".format(route_table_name))

    for subnet in subnets:
        cli_logger.print("Updating private subnet route table: {}...".format(subnet.id))
        private_route_table.associate_with_subnet(SubnetId=subnet.id)
        cli_logger.print("Successfully updated private subnet route table: {}.".format(subnet.id))
    return private_route_table


def _create_workspace(config):
    ec2 = _resource("ec2", config)
    ec2_client = _resource_client("ec2", config)
    workspace_name = config["workspace_name"]
    managed_cloud_storage = is_managed_cloud_storage(config)
    managed_cloud_database = is_managed_cloud_database(config)
    use_peering_vpc = is_use_peering_vpc(config)

    current_step = 1
    total_steps = AWS_WORKSPACE_NUM_CREATION_STEPS
    if managed_cloud_storage:
        total_steps += 1
    if managed_cloud_database:
        total_steps += 1
    if use_peering_vpc:
        total_steps += 1

    try:
        with cli_logger.group("Creating workspace: {}", workspace_name):
            current_step = _create_network_resources(config, ec2, ec2_client,
                                                     current_step, total_steps)

            with cli_logger.group(
                    "Creating instance profile",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _create_workspace_instance_profile(config, workspace_name)

            if managed_cloud_storage:
                with cli_logger.group(
                        "Creating S3 bucket",
                        _numbered=("[]", current_step, total_steps)):
                    current_step += 1
                    _create_workspace_cloud_storage(config, workspace_name)

            if managed_cloud_database:
                with cli_logger.group(
                        "Creating managed database",
                        _numbered=("[]", current_step, total_steps)):
                    current_step += 1
                    _create_workspace_cloud_database(config, workspace_name)

    except Exception as e:
        cli_logger.error("Failed to create workspace with the name {}. "
                         "You need to delete and try create again. {}", workspace_name, str(e))
        raise e

    cli_logger.success(
        "Successfully created workspace: {}.",
        cf.bold(workspace_name))

    return config


def _create_workspace_instance_profile(config, workspace_name):
    current_step = 1
    total_steps = 2

    with cli_logger.group(
            "Creating instance profile for head",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_instance_profile_for_head(config, workspace_name)

    with cli_logger.group(
            "Creating instance profile for worker",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_instance_profile_for_worker(config, workspace_name)


def _create_instance_profile_for_head(config, workspace_name):
    head_instance_profile_name = _get_head_instance_profile_name(workspace_name)
    head_instance_role_name = "cloudtik-{}-head-role".format(workspace_name)
    cli_logger.print("Creating head instance profile: {}...".format(head_instance_profile_name))
    _create_or_update_instance_profile(config, head_instance_profile_name,
                                       head_instance_role_name)
    cli_logger.print("Successfully created and configured head instance profile.")


def _create_instance_profile_for_worker(config, workspace_name):
    worker_instance_profile_name = _get_worker_instance_profile_name(workspace_name)
    worker_instance_role_name = "cloudtik-{}-worker-role".format(workspace_name)
    cli_logger.print("Creating worker instance profile: {}...".format(worker_instance_profile_name))
    _create_or_update_instance_profile(config, worker_instance_profile_name,
                                       worker_instance_role_name, is_head=False)
    cli_logger.print("Successfully created and configured worker instance profile.")


def _create_workspace_cloud_storage(config, workspace_name):
    _create_managed_cloud_storage(config["provider"], workspace_name)


def _create_managed_cloud_storage(cloud_provider, workspace_name):
    # If the managed cloud storage for the workspace already exists
    # Skip the creation step
    bucket = get_managed_s3_bucket(cloud_provider, workspace_name)
    if bucket is not None:
        cli_logger.print("S3 bucket for the workspace already exists. Skip creation.")
        return

    s3 = _make_resource("s3", cloud_provider)
    s3_client = _make_resource_client("s3", cloud_provider)
    region = cloud_provider["region"]
    suffix = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))
    bucket_name = "cloudtik-{workspace_name}-{region}-{suffix}".format(
        workspace_name=workspace_name,
        region=region,
        suffix=suffix
    )

    cli_logger.print("Creating S3 bucket for the workspace: {}...".format(workspace_name))
    try:
        if region == 'us-east-1':
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint': region})

        # Enable server side encryption by default
        s3_client.put_bucket_encryption(Bucket=bucket_name, ServerSideEncryptionConfiguration={
            'Rules': [{'ApplyServerSideEncryptionByDefault': {'SSEAlgorithm': 'AES256'}}, ]})
        cli_logger.print(
            "Successfully created S3 bucket: {}.".format(bucket_name))
    except Exception as e:
        cli_logger.error("Failed to create S3 bucket. {}", str(e))
        raise e


def _create_db_subnet_group(
        cloud_provider, workspace_name, subnet_ids):
    rds_client = _make_client("rds", cloud_provider)
    db_subnet_group = get_workspace_db_subnet_group(cloud_provider, workspace_name)
    if db_subnet_group is not None:
        cli_logger.print("The DB subnet group for the workspace already exists. Skip creation.")
        return

    db_subnet_group_name = AWS_WORKSPACE_DB_SUBNET_GROUP_NAME.format(workspace_name)

    cli_logger.print("Creating DB subnet group for workspace: {}...".format(workspace_name))
    try:
        rds_client.create_db_subnet_group(
            DBSubnetGroupName=db_subnet_group_name,
            DBSubnetGroupDescription='CloudTik workspace DB subnet group',
            SubnetIds=subnet_ids
        )
    except Exception as e:
        cli_logger.error("Failed to create DB subnet group. {}", str(e))
        raise e


def _create_workspace_cloud_database(config, workspace_name):
    cloud_provider = config["provider"]

    ec2 = _make_resource("ec2", cloud_provider)
    ec2_client = _make_resource_client("ec2", cloud_provider)
    vpc = get_workspace_vpc(workspace_name, ec2_client, ec2)
    subnet_ids = [subnet.id for subnet in vpc.subnets.all()]
    security_group = get_workspace_security_group(
        config, vpc.vpc_id, workspace_name)

    _create_managed_cloud_database(
        cloud_provider, workspace_name,
        subnet_ids, security_group.id
    )


def _create_managed_cloud_database(
        cloud_provider, workspace_name,
        subnet_ids, security_group_id):
    current_step = 1
    total_steps = 2

    with cli_logger.group(
            "Creating DB subnet group",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_db_subnet_group(
            cloud_provider, workspace_name, subnet_ids)

    with cli_logger.group(
            "Creating managed database instance",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_managed_database_instance(
            cloud_provider, workspace_name, security_group_id)


def _create_managed_database_instance(
        cloud_provider, workspace_name, security_group_id):
    # If the managed cloud database for the workspace already exists
    # Skip the creation step
    db_instance = get_managed_database_instance(cloud_provider, workspace_name)
    if db_instance is not None:
        cli_logger.print("Managed database instance for the workspace already exists. Skip creation.")
        return

    rds_client = _make_client("rds", cloud_provider)
    db_instance_identifier = AWS_WORKSPACE_DATABASE_NAME.format(workspace_name)
    db_subnet_group = AWS_WORKSPACE_DB_SUBNET_GROUP_NAME.format(workspace_name)
    database_config = get_aws_database_config(cloud_provider, {})

    cli_logger.print("Creating database instance for the workspace: {}...".format(workspace_name))
    try:
        rds_client.create_db_instance(
            DBInstanceIdentifier=db_instance_identifier,
            DBInstanceClass=database_config.get("instance_class", "db.t3.xlarge"),
            Engine="mysql",
            StorageType=database_config.get("storage_type", "gp2"),
            AllocatedStorage=database_config.get("allocated_storage", 50),
            MasterUsername=database_config.get('username', "cloudtik"),
            MasterUserPassword=database_config.get('password', "cloudtik"),
            VpcSecurityGroupIds=[
                security_group_id
            ],
            DBSubnetGroupName=db_subnet_group,
            PubliclyAccessible=False
        )
        wait_db_instance_creation(rds_client, db_instance_identifier)
    except Exception as e:
        cli_logger.error("Failed to create database instance. {}", str(e))
        raise e


def _get_head_instance_profile_name(workspace_name):
    return "cloudtik-{}-head-profile".format(workspace_name)


def _get_worker_instance_profile_name(workspace_name):
    return "cloudtik-{}-worker-profile".format(workspace_name)


def _get_head_instance_profile(config):
    workspace_name = config["workspace_name"]
    head_instance_profile_name = _get_head_instance_profile_name(workspace_name)
    return _get_instance_profile(head_instance_profile_name, config)


def _get_worker_instance_profile(config):
    workspace_name = config["workspace_name"]
    worker_instance_profile_name = _get_worker_instance_profile_name(workspace_name)
    return _get_instance_profile(worker_instance_profile_name, config)


def _create_network_resources(config, ec2, ec2_client,
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
        subnets = _create_and_configure_subnets(config, ec2_client, vpc)

    # TODO check whether we need to create new internet gateway? Maybe existing vpc contains internet subnets
    # create internet gateway for public subnets
    with cli_logger.group(
            "Creating Internet gateway",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        internet_gateway = _create_internet_gateway(config, ec2, ec2_client, vpc)

    # add internet_gateway into public route table
    with cli_logger.group(
            "Updating route tables",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        private_route_table = _create_or_update_route_tables(
            config, ec2, ec2_client, vpc, subnets, internet_gateway)

    # create NAT gateway for private subnets
    with cli_logger.group(
            "Creating and configuring NAT gateway",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_and_configure_nat_gateway(
            config, ec2_client, vpc, subnets[AWS_VPC_PUBLIC_SUBNET_INDEX], private_route_table)

    # create VPC endpoint for S3
    with cli_logger.group(
            "Creating VPC endpoint for S3",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_vpc_endpoint_for_s3(config, ec2, ec2_client, vpc)

    with cli_logger.group(
            "Creating security group",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _upsert_security_group(config, vpc.id)

    if is_use_peering_vpc(config):
        with cli_logger.group(
                "Creating VPC peering connection",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _create_and_configure_vpc_peering_connection(config, ec2, ec2_client)

    return current_step


def _configure_vpc(config, workspace_name, ec2, ec2_client):
    use_working_vpc = is_use_working_vpc(config)
    if use_working_vpc:
        # No need to create new vpc
        vpc_name = get_workspace_vpc_name(workspace_name)
        vpc = get_current_vpc(config)
        vpc.create_tags(Tags=[
            {'Key': 'Name', 'Value': vpc_name},
            {'Key': AWS_WORKSPACE_VERSION_TAG_NAME, 'Value': AWS_WORKSPACE_VERSION_CURRENT}
        ])
        cli_logger.print("Using the existing VPC: {} for workspace. Skip creation.".format(vpc.id))
    else:
        # Need to create a new vpc
        if get_workspace_vpc_id(config["workspace_name"], ec2_client) is None:
            vpc = _create_vpc(config, ec2, ec2_client)
        else:
            raise RuntimeError("There is a same name VPC for workspace: {}, "
                               "if you want to create a new workspace with the same name, "
                               "you need to execute workspace delete first!".format(workspace_name))
    return vpc


def _configure_subnets_cidr(vpc):
    cidr_list = []
    subnets = list(vpc.subnets.all())
    vpc_cidr_block = vpc.cidr_block
    ip = vpc_cidr_block.split("/")[0].split(".")

    if len(subnets) == 0:
        for i in range(0, AWS_VPC_SUBNETS_COUNT):
            cidr_list.append(ip[0] + "." + ip[1] + "." + str(i) + ".0/24")
    else:
        cidr_blocks = [subnet.cidr_block for subnet in subnets]
        for i in range(0, 256):
            tmp_cidr_block = ip[0] + "." + ip[1] + "." + str(i) + ".0/24"

            if check_cidr_conflict(tmp_cidr_block, cidr_blocks):
                cidr_list.append(tmp_cidr_block)

            if len(cidr_list) == AWS_VPC_SUBNETS_COUNT:
                break

    return cidr_list


def get_current_vpc_id(config):
    client = _working_node_client('ec2', config)
    ip_address = get_node_ip_address(address="8.8.8.8:53")
    vpc_id = None
    for Reservation in client.describe_instances().get("Reservations"):
        for instance in Reservation["Instances"]:
            if instance.get("PrivateIpAddress", "") == ip_address:
                vpc_id = instance["VpcId"]

    if vpc_id is None:
        raise RuntimeError("Failed to get the VPC for the current machine. "
                           "Please make sure your current machine is an AWS virtual machine.")
    return vpc_id


def get_current_vpc(config):
    current_vpc_id = get_current_vpc_id(config)
    ec2 = _working_node_resource('ec2', config)
    current_vpc = ec2.Vpc(id=current_vpc_id)
    return current_vpc


def _configure_subnet_from_workspace(config):
    ec2 = _resource("ec2", config)
    ec2_client = _resource_client("ec2", config)
    workspace_name = config["workspace_name"]
    use_internal_ips = is_use_internal_ip(config)

    vpc_id = get_workspace_vpc_id(workspace_name, ec2_client)
    public_subnets = get_workspace_public_subnets(workspace_name, ec2, vpc_id)
    private_subnets = get_workspace_private_subnets(workspace_name, ec2, vpc_id)
    public_subnet_ids = [public_subnet.id for public_subnet in public_subnets]
    private_subnet_ids = [private_subnet.id for private_subnet in private_subnets]

    # We need to make sure the first private subnet is the same availability zone with the first public subnet
    if not use_internal_ips and len(public_subnet_ids) > 0:
        availability_zone = public_subnets[0].availability_zone
        for private_subnet in private_subnets:
            if availability_zone == private_subnet.availability_zone:
                private_subnet_ids.remove(private_subnet.id)
                private_subnet_ids.insert(0, private_subnet.id)
                break

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


def _configure_security_group_from_workspace(config):
    ec2_client = _resource_client("ec2", config)
    workspace_name = config["workspace_name"]
    vpc_id = get_workspace_vpc_id(workspace_name, ec2_client)
    # map from node type key -> source of SecurityGroupIds field
    security_group_info_src = {}
    _set_config_info(security_group_src=security_group_info_src)
    sg = get_workspace_security_group(config,  vpc_id, workspace_name)

    for node_type_key in config["available_node_types"].keys():
        node_config = config["available_node_types"][node_type_key][
            "node_config"]
        node_config["SecurityGroupIds"] = [sg.id]
        security_group_info_src[node_type_key] = "workspace"

    return config


def _get_default_ami(config, default_ami, is_gpu):
    if default_ami is not None:
        return default_ami

    default_ami = get_latest_ami_id(config, is_gpu)
    if not default_ami:
        region = config["provider"]["region"]
        cli_logger.warning(
            "Can not get latest ami information in this region: {}. Will use default ami id".format(region))
        default_ami = DEFAULT_AMI_GPU.get(region) if is_gpu else DEFAULT_AMI.get(region)
        if not default_ami:
            cli_logger.abort("Not support on this region: {}. Please use one of these regions {}".
                             format(region, sorted(DEFAULT_AMI.keys())))
    return default_ami


def _configure_ami(config):
    """Provide helpful message for missing ImageId for node configuration."""

    # map from node type key -> source of ImageId field
    ami_src_info = {key: "config" for key in config["available_node_types"]}
    _set_config_info(ami_src=ami_src_info)

    is_gpu = is_gpu_runtime(config)

    default_ami = None
    for key, node_type in config["available_node_types"].items():
        node_config = node_type["node_config"]
        image_id = node_config.get("ImageId", "")
        if image_id == "":
            # Only set to default ami if not specified by the user
            default_ami = _get_default_ami(config, default_ami, is_gpu)
            node_config["ImageId"] = default_ami

    return config


def _upsert_security_group(config, vpc_id):
    current_step = 1
    total_steps = 2

    with cli_logger.group(
            "Creating security group for VPC",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        security_group = _create_workspace_security_group(config, vpc_id)

    with cli_logger.group(
            "Configuring rules for security group",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _add_security_group_rules(config, security_group)

    return security_group


def _create_workspace_security_group(config, vpc_id):
    group_name = SECURITY_GROUP_TEMPLATE.format(config["workspace_name"])
    return _create_security_group(config["provider"], vpc_id, group_name)


def _update_security_group(config, vpc_id):
    security_group = get_workspace_security_group(config, vpc_id, config["workspace_name"])
    _add_security_group_rules(config, security_group)
    return security_group


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


def _get_security_group(provider_config, vpc_id, group_name):
    security_group = _get_security_groups(provider_config, [vpc_id], [group_name])
    return None if not security_group else security_group[0]


def _get_security_groups(provider_config, vpc_ids, group_names):
    unique_vpc_ids = list(set(vpc_ids))
    unique_group_names = set(group_names)

    ec2 = _make_resource("ec2", provider_config)
    existing_groups = list(
        ec2.security_groups.filter(Filters=[{
            "Name": "vpc-id",
            "Values": unique_vpc_ids
        }]))
    filtered_groups = [
        sg for sg in existing_groups if sg.group_name in unique_group_names
    ]
    return filtered_groups


def wait_security_group_creation(ec2_client, vpc_id, group_name):
    waiter = ec2_client.get_waiter('security_group_exists')
    waiter.wait(Filters=[
        {
            'Name': 'vpc-id',
            'Values': [vpc_id]
        },
        {
            'Name': 'group-name',
            'Values': [group_name]
        }
    ])


def _create_security_group(provider_config, vpc_id, group_name):
    cli_logger.print("Creating security group for VPC: {}...".format(group_name))
    client = _make_resource_client("ec2", provider_config)
    client.create_security_group(
        Description="Auto-created security group for workers",
        GroupName=group_name,
        VpcId=vpc_id)

    # Wait for creation
    wait_security_group_creation(client, vpc_id, group_name)

    security_group = _get_security_group(
        provider_config, vpc_id, group_name)
    cli_logger.doassert(security_group,
                        "Failed to create security group")

    cli_logger.print("Successfully created security group: {}.".format(group_name))
    return security_group


def _add_security_group_rules(config, security_group):
    cli_logger.print("Updating rules for security group: {}...".format(security_group.id))
    security_group_ids = {security_group.id}
    extended_rules = config["provider"] \
        .get("security_group", {}) \
        .get("IpPermissions", [])
    ip_permissions = _create_default_inbound_rules(
        config, security_group_ids, extended_rules)
    _update_inbound_rules(security_group, ip_permissions)
    cli_logger.print("Successfully updated rules for security group.")


def _update_inbound_rules(target_security_group, ip_permissions):
    old_ip_permissions = target_security_group.ip_permissions
    if len(old_ip_permissions) != 0:
        target_security_group.revoke_ingress(IpPermissions=old_ip_permissions)
    target_security_group.authorize_ingress(IpPermissions=ip_permissions)


def _create_default_inbound_rules(config, sgids, extended_rules=None):
    if extended_rules is None:
        extended_rules = []
    intra_cluster_rules = _create_default_intra_cluster_inbound_rules(sgids)
    ssh_rules = _create_default_ssh_inbound_rules(sgids, config)

    if is_use_peering_vpc(config) and is_peering_firewall_allow_working_subnet(config):
        extended_rules += _create_allow_working_node_inbound_rules(config)

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


def _create_allow_working_node_inbound_rules(config):
    allow_ssh_only = is_peering_firewall_allow_ssh_only(config)
    vpc = get_current_vpc(config)
    working_vpc_cidr = vpc.cidr_block
    return [{
        "FromPort": 22 if allow_ssh_only else 0,
        "ToPort": 22 if allow_ssh_only else 65535,
        "IpProtocol": "tcp",
        "IpRanges": [{
            "CidrIp": working_vpc_cidr
        }]
    }]


def _get_role(role_name, config):
    return _get_iam_role(role_name, config["provider"])


def _get_iam_role(role_name, provider_config):
    iam = _make_resource("iam", provider_config)
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


def get_workspace_s3_bucket(config, workspace_name):
    return get_managed_s3_bucket(config["provider"], workspace_name)


def get_managed_s3_bucket(provider_config, workspace_name):
    s3 = _make_resource("s3", provider_config)
    region = provider_config["region"]
    bucket_name_prefix = "cloudtik-{workspace_name}-{region}-".format(
        workspace_name=workspace_name,
        region=region
    )

    cli_logger.verbose("Getting s3 bucket with prefix: {}.".format(bucket_name_prefix))
    for bucket in s3.buckets.all():
        if bucket_name_prefix in bucket.name:
            cli_logger.verbose("Successfully get the s3 bucket: {}.".format(bucket.name))
            return bucket

    cli_logger.verbose_error("Failed to get the s3 bucket for workspace.")
    return None


def get_workspace_database_instance(config, workspace_name):
    return get_managed_database_instance(config["provider"], workspace_name)


def get_managed_database_instance(provider_config, workspace_name):
    rds_client = _make_client("rds", provider_config)
    db_instances = [db_instance for db_instance in rds_client.describe_db_instances().get("DBInstances", [])
                    if db_instance.get('DBInstanceStatus') == 'available']
    db_instance_identifier = AWS_WORKSPACE_DATABASE_NAME.format(workspace_name)
    cli_logger.verbose("Getting the managed database with identifier: {}.".format(db_instance_identifier))
    for db_instance in db_instances:
        if db_instance.get('DBInstanceIdentifier') == db_instance_identifier:
            cli_logger.verbose("Successfully get the managed database: {}.".format(db_instance_identifier))
            return db_instance

    cli_logger.verbose_error("Failed to get the managed database for workspace.")
    return None


def get_workspace_db_subnet_group(provider_config, workspace_name):
    rds_client = _make_client("rds", provider_config)
    db_subnet_groups = [db_subnet_group for db_subnet_group in rds_client.describe_db_subnet_groups().get(
        "DBSubnetGroups", []) if db_subnet_group.get('SubnetGroupStatus') == 'Complete']
    db_subnet_group_name = AWS_WORKSPACE_DB_SUBNET_GROUP_NAME.format(workspace_name)
    cli_logger.verbose("Getting the workspace DB subnet group: {}.".format(db_subnet_group_name))
    for db_subnet_group in db_subnet_groups:
        if db_subnet_group.get('DBSubnetGroupName') == db_subnet_group_name:
            cli_logger.verbose("Successfully get the workspace DB subnet group: {}.".format(db_subnet_group_name))
            return db_subnet_group

    cli_logger.verbose_error("Failed to get the workspace DB subnet group.")
    return None


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
    node_types = config["available_node_types"]

    # iterate over sorted node types to support deterministic unit test stubs
    for name, node_type in sorted(node_types.items()):
        node_cfg = node_type["node_config"]
        if "LaunchTemplate" in node_cfg:
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

    ec2 = _resource_client("ec2", config)
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


def verify_s3_storage(provider_config: Dict[str, Any]):
    s3_storage = get_aws_s3_storage_config(provider_config)
    if s3_storage is None:
        return

    s3 = boto3.client(
        's3',
        aws_access_key_id=s3_storage.get("s3.access.key.id"),
        aws_secret_access_key=s3_storage.get("s3.secret.access.key")
    )

    try:
        s3.list_objects(Bucket=s3_storage[AWS_S3_BUCKET], Delimiter='/')
    except botocore.exceptions.ClientError as e:
        raise StorageTestingError("Error happens when verifying S3 storage configurations. "
                                  "If you want to go without passing the verification, "
                                  "set 'verify_cloud_storage' to False under provider config. "
                                  "Error: {}.".format(e.message)) from None


def get_cluster_name_from_head(head_node) -> Optional[str]:
    for tag in head_node.tags:
        tag_key = tag.get("Key")
        if tag_key == CLOUDTIK_TAG_CLUSTER_NAME:
            return tag.get("Value")
    return None


def list_aws_clusters(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    head_nodes = get_workspace_head_nodes(config)
    clusters = {}
    for head_node in head_nodes:
        cluster_name = get_cluster_name_from_head(head_node)
        if cluster_name:
            clusters[cluster_name] = _get_node_info(head_node)
    return clusters


def with_aws_environment_variables(provider_config, node_type_config: Dict[str, Any], node_id: str):
    config_dict = {}
    export_aws_s3_storage_config(provider_config, config_dict)
    export_aws_database_config(provider_config, config_dict)
    return config_dict
