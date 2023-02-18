import logging
from functools import partial
import copy
import os
import stat
import subprocess
import random
import string
import time
import itertools
from typing import Any, Dict, Optional, List

from Tea.exceptions import UnretryableException

from cloudtik.core._private.cli_logger import cli_logger, cf
from cloudtik.core._private.services import get_node_ip_address
from cloudtik.core._private.utils import check_cidr_conflict, get_cluster_uri, is_use_internal_ip, \
    is_managed_cloud_storage, is_use_managed_cloud_storage, is_worker_role_for_cloud_storage, is_use_working_vpc, \
    is_use_peering_vpc, is_peering_firewall_allow_ssh_only, is_peering_firewall_allow_working_subnet, format_exception_message
from cloudtik.core.tags import CLOUDTIK_TAG_CLUSTER_NAME, CLOUDTIK_TAG_NODE_KIND, NODE_KIND_HEAD
from cloudtik.core.workspace_provider import Existence, CLOUDTIK_MANAGED_CLOUD_STORAGE, \
    CLOUDTIK_MANAGED_CLOUD_STORAGE_URI
from cloudtik.providers._private.aliyun.utils import AcsClient, export_aliyun_oss_storage_config, \
    get_aliyun_oss_storage_config, get_aliyun_oss_storage_config_for_update, ALIYUN_OSS_BUCKET, _get_node_info, \
    get_aliyun_cloud_storage_uri

from cloudtik.providers._private.aliyun.utils import OssClient, EcsClient, RamClient, VpcClient, VpcPeerClient, check_resource_status
from cloudtik.providers._private.utils import StorageTestingError

# instance status
PENDING = "Pending"
RUNNING = "Running"
STARTING = "Starting"
STOPPING = "Stopping"
STOPPED = "Stopped"

logger = logging.getLogger(__name__)

ALIYUN_DEFAULT_IMAGE_FAMILY = "acs:ubuntu_20_04_64"

ALIYUN_DEFAULT_IMAGE_BY_REGION = {
}
ALIYUN_DEFAULT_IMAGE_ID = "ubuntu_20_04_x64_20G_alibase_20221228.vhd"

ALIYUN_WORKSPACE_NUM_CREATION_STEPS = 5
ALIYUN_WORKSPACE_NUM_DELETION_STEPS = 7
ALIYUN_WORKSPACE_TARGET_RESOURCES = 8
ALIYUN_VPC_SWITCHES_COUNT=2

ALIYUN_RESOURCE_NAME_PREFIX = "cloudtik"
ALIYUN_WORKSPACE_VPC_NAME = ALIYUN_RESOURCE_NAME_PREFIX + "-{}-vpc"
ALIYUN_WORKSPACE_PUBLIC_VSWITCH_NAME = ALIYUN_RESOURCE_NAME_PREFIX + "-{}-public-vswitch"
ALIYUN_WORKSPACE_PRIVATE_VSWITCH_NAME = ALIYUN_RESOURCE_NAME_PREFIX + "-{}-private-vswitch"
ALIYUN_WORKSPACE_NAT_VSWITCH_NAME = ALIYUN_RESOURCE_NAME_PREFIX + "-{}-nat-vswitch"
ALIYUN_WORKSPACE_SECURITY_GROUP_NAME = ALIYUN_RESOURCE_NAME_PREFIX + "-{}-security-group"
ALIYUN_WORKSPACE_EIP_NAME = ALIYUN_RESOURCE_NAME_PREFIX + "-{}-eip"
ALIYUN_WORKSPACE_NAT_GATEWAY_NAME = ALIYUN_RESOURCE_NAME_PREFIX + "-{}-nat"
ALIYUN_WORKSPACE_SNAT_ENTRY_NAME = ALIYUN_RESOURCE_NAME_PREFIX + "-{}-snat"
ALIYUN_WORKSPACE_VPC_PEERING_ROUTE_ENTRY_NAME = ALIYUN_RESOURCE_NAME_PREFIX + "-{}-route-entry"
ALIYUN_WORKSPACE_VPC_PEERING_NAME = ALIYUN_RESOURCE_NAME_PREFIX + "-{}-vpc-peering-connection"

ALIYUN_WORKSPACE_VERSION_TAG_NAME = "cloudtik-workspace-version"
ALIYUN_WORKSPACE_VERSION_CURRENT = "1"

ALIYUN_MANAGED_STORAGE_OSS_BUCKET = "aliyun.managed.storage.oss.bucket"

HEAD_ROLE_ATTACH_POLICIES = [
    "AliyunECSFullAccess",
    "AliyunOSSFullAccess",
    "AliyunRAMFullAccess"
]

WORKER_ROLE_ATTACH_POLICIES = [
    "AliyunOSSFullAccess",
]

MAX_POLLS = 15
MAX_POLLS_NAT = MAX_POLLS * 8
POLL_INTERVAL = 1

def bootstrap_aliyun(config):
    workspace_name = config.get("workspace_name", "")
    if workspace_name == "":
        raise RuntimeError("Workspace name is not specified in cluster configuration.")

    config = bootstrap_aliyun_from_workspace(config)
    return config


def bootstrap_aliyun_from_workspace(config):
    if not check_aliyun_workspace_integrity(config):
        workspace_name = config["workspace_name"]
        cli_logger.abort("Alibaba Cloud workspace {} doesn't exist or is in wrong state.", workspace_name)

    # create vpc
    # _get_or_create_vpc(config)
    # create security group id
    # _get_or_create_security_group(config)
    # create vswitch
    # _get_or_create_vswitch(config)
    # create key pair
    # _get_or_import_key_pair(config)

    # create a copy of the input config to modify
    config = copy.deepcopy(config)

    # Used internally to store head IAM role.
    config["head_node"] = {}

    # If a LaunchTemplate is provided, extract the necessary fields for the
    # config stages below.
    config = _configure_from_launch_template(config)

    # The head node needs to have an RAM role that allows it to create further
    # ECS instances.
    config = _configure_ram_role_from_workspace(config)

    # Set oss.bucket if use_managed_cloud_storage
    config = _configure_cloud_storage_from_workspace(config)

    # Configure SSH access, using an existing key pair if possible.
    config = _configure_key_pair(config)

    # Pick a reasonable subnet if not specified by the user.
    config = _configure_vswitch_from_workspace(config)

    # Cluster workers should be in a security group that permits traffic within
    # the group, and also SSH access from outside.
    config = _configure_security_group_from_workspace(config)

    # Provide a helpful message for missing AMI.
    config = _configure_image(config)

    config = _configure_prefer_spot_node(config)
    return config


def _configure_from_launch_template(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges any launch template data referenced by the node config of all
    available node type's into their parent node config. Any parameters
    specified in node config override the same parameters in the launch
    template, in compliance with the behavior of the create_instances
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
        node_config = node_type["node_config"]
        if ("LaunchTemplateId" in node_config
                or "LaunchTemplateName" in node_config):
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

    node_config = node_type["node_config"]
    node_type["node_config"] = \
        _configure_node_config_from_launch_template(config, node_config)
    return node_type


def _configure_node_config_from_launch_template(
        config: Dict[str, Any], node_config: Dict[str, Any]) -> Dict[str, Any]:
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
    here, since the AliyunNodeProvider's ecs.create_instances call will do this
    for us after it fetches the referenced launch template data.

    Args:
        config (Dict[str, Any]): config to bootstrap
        node_config (Dict[str, Any]): node config to bootstrap
    Returns:
        node_config (Dict[str, Any]): The input node config merged with all launch
        template data. If no launch template data is found, then the node
        config is returned unchanged.
    Raises:
        ValueError: If no launch template is found for the given launch
        template [name|id] and version, or more than one launch template is
        found.
    """
    # create a copy of the input config to modify
    node_config = copy.deepcopy(node_config)

    ecs_client = EcsClient(config["provider"])
    query_params = {
        "LaunchTemplateId": node_config.get("LaunchTemplateId"),
        "LaunchTemplateName": node_config.get("LaunchTemplateName")
    }
    if "LaunchTemplateVersion" in node_config:
        query_params["LaunchTemplateVersion"] = [node_config["LaunchTemplateVersion"]]
    else:
        query_params["DefaultVersion"] = True

    templates = ecs_client.describe_launch_template_versions(query_params)

    if templates is None or len(templates) != 1:
        raise ValueError(f"Expected to find 1 launch template but found "
                         f"{len(templates)}")

    lt_data = templates[0].launch_template_data.to_map()
    # override launch template parameters with explicit node config parameters
    lt_data.update(node_config)
    # copy all new launch template parameters back to node config
    node_config.update(lt_data)

    return node_config


def _configure_cloud_storage_from_workspace(config):
    use_managed_cloud_storage = is_use_managed_cloud_storage(config)
    if use_managed_cloud_storage:
        _configure_managed_cloud_storage_from_workspace(config, config["provider"])

    return config


def _configure_managed_cloud_storage_from_workspace(config, cloud_provider):
    workspace_name = config["workspace_name"]
    oss_bucket = get_managed_oss_bucket(cloud_provider, workspace_name)
    if oss_bucket is None:
        cli_logger.abort("No managed OSS bucket was found. If you want to use managed OSS bucket, "
                         "you should set managed_cloud_storage equal to True when you creating workspace.")

    cloud_storage = get_aliyun_oss_storage_config_for_update(config["provider"])
    cloud_storage[ALIYUN_OSS_BUCKET] = oss_bucket.name


def _key_assert_msg(node_type: str) -> str:
    return ("`KeyPairName` missing from the `node_config` of"
            f" node type `{node_type}`.")


def _key_pair(i, region, key_name):
    """
    If key_name is not None, key_pair will be named after key_name.
    Returns the ith default (aws_key_pair_name, key_pair_path).
    """
    if i == 0:
        key_pair_name = ("{}_aliyun_{}".format(ALIYUN_RESOURCE_NAME_PREFIX, region)
                         if key_name is None else key_name)
        return (key_pair_name,
                os.path.expanduser("~/.ssh/{}.pem".format(key_pair_name)))

    key_pair_name = ("{}_aliyun_{}_{}".format(ALIYUN_RESOURCE_NAME_PREFIX, region, i)
                     if key_name is None else key_name + "_key-{}".format(i))
    return (key_pair_name,
            os.path.expanduser("~/.ssh/{}.pem".format(key_pair_name)))


def _configure_key_pair(config):
    node_types = config["available_node_types"]

    if "ssh_private_key" in config["auth"]:
        # If the key is not configured via the cloudinit
        # UserData, it should be configured via KeyName or
        # else we will risk starting a node that we cannot
        # SSH into:
        for node_type in node_types:
            node_config = node_types[node_type]["node_config"]
            if "UserData" not in node_config:
                cli_logger.doassert("KeyPairName" in node_config,
                                    _key_assert_msg(node_type))
                assert "KeyPairName" in node_config

        return config

    ecs_client = EcsClient(config["provider"])

    # Writing the new ssh key to the filesystem fails if the ~/.ssh
    # directory doesn't already exist.
    os.makedirs(os.path.expanduser("~/.ssh"), exist_ok=True)

    # Try a few times to get or create a good key pair.
    MAX_NUM_KEYS = 30
    for i in range(MAX_NUM_KEYS):
        key_name = config["provider"].get("key_pair", {}).get("key_name")
        key_name, key_path = _key_pair(
            i, config["provider"]["region"], key_name)
        key = ecs_client.describe_key_pair(key_name)
        # Found a good key.
        if key and os.path.exists(key_path):
            break

        # We can safely create a new key.
        if not key and not os.path.exists(key_path):
            cli_logger.verbose(
                "Creating new key pair {} for use as the default.",
                cf.bold(key_name))
            key = ecs_client.create_key_pair(key_name)

            # We need to make sure to _create_ the file with the right
            # permissions. In order to do that we need to change the default
            # os.open behavior to include the mode we want.
            with open(key_path, "w", opener=partial(os.open, mode=0o600)) as f:
                f.write(key.private_key_body)
            break

    if not key:
        cli_logger.abort(
            "No matching local key file for any of the key pairs in this "
            "account with ids from 0..{}. "
            "Consider deleting some unused keys pairs from your account.",
            key_name)

    cli_logger.doassert(
        os.path.exists(key_path), "Private key file " + cf.bold("{}") +
        " not found for " + cf.bold("{}"), key_path, key_name)
    assert os.path.exists(key_path), \
        "Private key file {} not found for {}".format(key_path, key_name)

    config["auth"]["ssh_private_key"] = key_path
    for node_type in node_types.values():
        node_config = node_type["node_config"]
        node_config["KeyPairName"] = key_name

    return config


def _configure_ram_role_from_workspace(config):
    config = copy.deepcopy(config)
    _configure_ram_role_for_head(config)

    worker_role_for_cloud_storage = is_worker_role_for_cloud_storage(config)
    if worker_role_for_cloud_storage:
        _configure_ram_role_for_worker(config)

    return config


def _configure_ram_role_for_head(config):
    head_node_type = config["head_node_type"]
    head_node_config = config["available_node_types"][head_node_type][
        "node_config"]
    if "RamRoleName" in head_node_config:
        return

    head_instance_ram_role_name = _get_head_instance_role_name(
        config["workspace_name"])
    if not head_instance_ram_role_name:
        raise RuntimeError("Head instance ram role: {} not found!".format(head_instance_ram_role_name))

    # Add IAM role to "head_node" field so that it is applied only to
    # the head node -- not to workers with the same node type as the head.
    config["head_node"]["RamRoleName"] = head_instance_ram_role_name


def _configure_ram_role_for_worker(config):
    worker_instance_ram_role_name = _get_worker_instance_role_name(
        config["workspace_name"])
    if not worker_instance_ram_role_name:
        raise RuntimeError("Workspace worker instance ram role: {} not found!".format(
            worker_instance_ram_role_name))

    for key, node_type in config["available_node_types"].items():
        node_config = node_type["node_config"]
        if key == config["head_node_type"]:
            continue

        if "RamRoleName" in node_config:
            continue

        node_config["RamRoleName"] = worker_instance_ram_role_name


def _configure_security_group_from_workspace(config):
    vpc_client = VpcClient(config["provider"])
    vpc_id = get_workspace_vpc_id(config, vpc_client)

    ecs_cli = EcsClient(config["provider"])
    security_group = get_workspace_security_group(config, ecs_cli, vpc_id)

    for node_type_key in config["available_node_types"].keys():
        node_config = config["available_node_types"][node_type_key][
            "node_config"]
        node_config["SecurityGroupId"] = security_group.security_group_id

    return config


def _configure_vswitch_from_workspace(config):
    workspace_name = config["workspace_name"]
    use_internal_ips = is_use_internal_ip(config)

    vpc_cli = VpcClient(config["provider"])
    vpc_id = get_workspace_vpc_id(workspace_name, vpc_cli)

    public_vswitches = get_workspace_public_vswitches(workspace_name, vpc_id, vpc_cli)
    private_vswitches = get_workspace_private_vswitches(workspace_name, vpc_id, vpc_cli)

    public_vswitch_ids = [public_vswitch.v_switch_id for public_vswitch in public_vswitches]
    private_vswitch_ids = [private_vswitch.v_switch_id for private_vswitch in private_vswitches]

    # We need to make sure the first private vswitch is the same availability zone with the first public vswitch
    if not use_internal_ips and len(public_vswitch_ids) > 0:
        availability_zone = public_vswitches[0].zone_id
        for private_vswitch in private_vswitches:
            if availability_zone == private_vswitch.zone_id:
                private_vswitch_ids.remove(private_vswitch.v_switch_id)
                private_vswitch_ids.insert(0, private_vswitch.v_switch_id)
                break

    for key, node_type in config["available_node_types"].items():
        node_config = node_type["node_config"]
        if key == config["head_node_type"]:
            if use_internal_ips:
                node_config["v_switch_id"] = private_vswitch_ids[0]
            else:
                node_config["v_switch_id"] = public_vswitch_ids[0]
        else:
            node_config["v_switch_id"] = private_vswitch_ids[0]

    return config


def get_latest_image_id(config: Dict[str, Any]):
    try:
        ecs_client = EcsClient(config["provider"])
        images = ecs_client.describe_images(ALIYUN_DEFAULT_IMAGE_FAMILY)
        if images is not None and len(images) > 0:
            images.sort(key=lambda item: item.creation_time, reverse=True)
            image_id = images[0].image_id
            return image_id
        else:
            return None
    except Exception as e:
        cli_logger.warning(
            "Error when getting latest image information.", str(e))
        return None


def _get_default_image(config, default_image):
    if default_image is not None:
        return default_image

    default_image = get_latest_image_id(config)
    if not default_image:
        region = config["provider"]["region"]
        cli_logger.warning(
            "Can not get latest image information in this region: {}. Will use default image id".format(region))
        default_image = ALIYUN_DEFAULT_IMAGE_BY_REGION.get(region)
        if not default_image:
            return ALIYUN_DEFAULT_IMAGE_ID
    return default_image


def _configure_image(config):
    """Provide helpful message for missing ImageId for node configuration."""
    default_image = None
    for key, node_type in config["available_node_types"].items():
        node_config = node_type["node_config"]
        image_id = node_config.get("ImageId")
        if not image_id:
            # Only set to default image if not specified by the user
            default_image = _get_default_image(config, default_image)
            node_config["ImageId"] = default_image

    return config


def _configure_spot_for_node_type(node_type_config,
                                  prefer_spot_node):
    # To be improved if scheduling has other configurations
    # SpotStrategy: SpotAsPriceGo
    # SpotStrategy: NoSpot
    node_config = node_type_config["node_config"]
    if prefer_spot_node:
        # Add spot instruction
        node_config["SpotStrategy"] = "SpotAsPriceGo"
    else:
        # Remove spot instruction
        node_config.pop("SpotStrategy", None)


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


def verify_oss_storage(provider_config: Dict[str, Any]):
    oss_storage = get_aliyun_oss_storage_config(provider_config)
    if oss_storage is None:
        return

    credentials_config = None
    oss_access_key_id = oss_storage.get("oss.access.key.id")
    oss_access_key_secret = oss_storage.get("oss.access.key.secret")
    if oss_access_key_id and oss_access_key_secret:
        credentials_config = {
            "aliyun_access_key_id": oss_access_key_id,
            "aliyun_access_key_secret": oss_access_key_secret
        }

    oss_client = OssClient(provider_config, credentials_config)

    try:
        oss_client.list_objects(oss_storage[ALIYUN_OSS_BUCKET])
    except Exception as e:
        raise StorageTestingError("Error happens when verifying OSS storage configurations. "
                                  "If you want to go without passing the verification, "
                                  "set 'verify_cloud_storage' to False under provider config. "
                                  "Error: {}.".format(e.message)) from None


def post_prepare_aliyun(config: Dict[str, Any]) -> Dict[str, Any]:
    try:
        config = fill_available_node_types_resources(config)
    except Exception as exc:
        cli_logger.warning(
            "Failed to detect node resources. Make sure you have properly configured the Alibaba Cloud credentials: {}.",
            str(exc))
        raise
    return config


def list_ecs_instances(provider_config) -> List[Dict[str, Any]]:
    """Get all instance-types/resources available.
    Args:
        provider_config: the provider config of the Alibaba Cloud.
    Returns:
        final_instance_types: a list of instances.

    """
    final_instance_types = []
    ecs_client = EcsClient(provider_config)
    instance_types_body = ecs_client.describe_instance_types()
    if (instance_types_body.instance_types is not None
            and instance_types_body.instance_types.instance_type is not None):
        final_instance_types.extend(
            copy.deepcopy(instance_types_body.instance_types.instance_type))
    while instance_types_body.next_token is not None:
        instance_types_body = ecs_client.describe_instance_types(
            next_token=instance_types_body.next_token)
        if (instance_types_body.instance_types is not None
                and instance_types_body.instance_types.instance_type is not None):
            final_instance_types.extend(
                copy.deepcopy(instance_types_body.instance_types.instance_type))
    return final_instance_types


def fill_available_node_types_resources(
        cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    """Fills out missing "resources" field for available_node_types."""
    if "available_node_types" not in cluster_config:
        return cluster_config
    cluster_config = copy.deepcopy(cluster_config)

    # Get instance information from cloud provider
    instances_list = list_ecs_instances(cluster_config["provider"])
    instances_dict = {
        instance.instance_type_id: instance
        for instance in instances_list
    }

    # Update the instance information to node type
    available_node_types = cluster_config["available_node_types"]
    for node_type in available_node_types:
        instance_type = available_node_types[node_type]["node_config"][
            "InstanceType"]
        if instance_type in instances_dict:
            cpus = instances_dict[instance_type].cpu_core_count
            detected_resources = {"CPU": cpus}

            # memory_size is in GB float
            memory_total = int(instances_dict[instance_type].memory_size * 1024)
            memory_total_in_bytes = int(memory_total) * 1024 * 1024
            detected_resources["memory"] = memory_total_in_bytes

            gpuamount = instances_dict[instance_type].gpuamount
            if gpuamount:
                gpu_name = instances_dict[instance_type].gpuspec
                detected_resources.update({
                    "GPU": gpuamount,
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
                             " is not available.")
    return cluster_config


def with_aliyun_environment_variables(provider_config, node_type_config: Dict[str, Any], node_id: str):
    config_dict = {}
    export_aliyun_oss_storage_config(provider_config, config_dict)

    if node_type_config is not None:
        node_config = node_type_config.get("node_config")
        ram_role_name = node_config.get("RamRoleName")
        if ram_role_name:
            config_dict["ALIYUN_ECS_RAM_ROLE_NAME"] = ram_role_name

    return config_dict


def _client(config):
    return AcsClient(
        access_key=config["provider"].get("aliyun_credentials",{}).get("aliyun_access_key_id"),
        access_key_secret=config["provider"].get("aliyun_credentials", {}).get("aliyun_secret_key_secret"),
        region_id=config["provider"]["region"],
        max_retries=1,
    )


def _get_or_create_security_group(config):
    cli = _client(config)
    security_groups = cli.describe_security_groups(vpc_id=config["provider"]["vpc_id"])
    if security_groups is not None and len(security_groups) > 0:
        config["provider"]["security_group_id"] = security_groups[0]["SecurityGroupId"]
        return config

    security_group_id = cli.create_security_group(vpc_id=config["provider"]["vpc_id"])

    for rule in config["provider"].get("security_group_rule", {}):
        cli.authorize_security_group(
            security_group_id=security_group_id,
            port_range=rule["port_range"],
            source_cidr_ip=rule["source_cidr_ip"],
            ip_protocol=rule["ip_protocol"],
        )
    config["provider"]["security_group_id"] = security_group_id
    return


def _get_or_create_vpc(config):
    cli = _client(config)
    vpcs = cli.describe_vpcs()
    if vpcs is not None and len(vpcs) > 0:
        config["provider"]["vpc_id"] = vpcs[0].vpc_id
        return

    vpc_id = cli.create_vpc()
    if vpc_id is not None:
        config["provider"]["vpc_id"] = vpc_id


def _get_or_create_vswitch(config):
    cli = _client(config)
    vswitches = cli.describe_v_switches(vpc_id=config["provider"]["vpc_id"])
    if vswitches is not None and len(vswitches) > 0:
        config["provider"]["v_switch_id"] = vswitches[0].v_switch_id
        return

    v_switch_id = cli.create_v_switch(
        vpc_id=config["provider"]["vpc_id"],
        zone_id=config["provider"]["zone_id"],
        cidr_block=config["provider"]["cidr_block"],
    )

    if v_switch_id is not None:
        config["provider"]["v_switch_id"] = v_switch_id


def _get_or_import_key_pair(config):
    cli = _client(config)
    key_name = config["provider"].get("key_name", "ray")
    key_path = os.path.expanduser("~/.ssh/{}".format(key_name))
    keypairs = cli.describe_key_pairs(key_pair_name=key_name)

    if keypairs is not None and len(keypairs) > 0:
        if "ssh_private_key" not in config["auth"]:
            logger.info(
                "{} keypair exists, use {} as local ssh key".format(key_name, key_path)
            )
            config["auth"]["ssh_private_key"] = key_path
    else:
        if "ssh_private_key" not in config["auth"]:
            # create new keypair
            resp = cli.create_key_pair(key_pair_name=key_name)
            if resp is not None:
                with open(key_path, "w+") as f:
                    f.write(resp.get("PrivateKeyBody"))
                os.chmod(key_path, stat.S_IRUSR)
                config["auth"]["ssh_private_key"] = key_path
        else:
            public_key_file = config["auth"]["ssh_private_key"] + ".pub"
            # create new keypair, from local file
            with open(public_key_file) as f:
                public_key = f.readline().strip("\n")
                cli.import_key_pair(key_pair_name=key_name, public_key_body=public_key)
                return


def get_workspace_vpc_peering_name(workspace_name):
    return ALIYUN_WORKSPACE_VPC_PEERING_NAME.format(workspace_name)


def _create_workspace_vpc_peer_connection(config, vpc_peer_cli):
    provider_config = config["provider"]
    current_region_id = get_current_instance_region()
    current_vpc_peer_cli = VpcPeerClient(provider_config, current_region_id)
    
    owner_account_id = get_current_instance_owner_account_id
    workspace_name = config["workspace_name"]
    vpc_peer_name = get_workspace_vpc_peering_name(workspace_name)
    region = provider_config["region"]
    current_vpc = get_current_vpc(config)
    vpc_cli = VpcClient(provider_config)
    workspace_vpc = get_workspace_vpc(workspace_name, vpc_cli)
    cli_logger.print("Creating VPC peering connection.")

    instance_id = current_vpc_peer_cli.create_vpc_peer_connection(
        region_id=current_region_id,
        vpc_id=current_vpc.vpc_id,
        accepted_ali_uid=owner_account_id,
        accepted_vpc_id=workspace_vpc.vpc_id,
        accepted_region_id=region,
        name=vpc_peer_name)
    cli_logger.print("Successfully created VPC peering connection: {}.", vpc_peer_name)
    return instance_id


def get_vpc_route_tables(vpc, vpc_cli):
    route_tables = vpc_cli.describe_route_tables(vpc.vpc_id)
    return route_tables


def get_workspace_vpc_peer_connection(config):
    workspace_name = config["workspace_name"]
    vpc_peer_name = get_workspace_vpc_peering_name(workspace_name)
    current_region_id = get_current_instance_region()
    current_vpc_peer_cli =VpcPeerClient(config["provider"], current_region_id)
    current_vpc_id = get_current_vpc_id(config)

    vpc_peer_connections = current_vpc_peer_cli.describe_vpc_peer_connections(
        vpc_id=current_vpc_id,
        vpc_peer_connection_name=vpc_peer_name
    )
    return None if len(vpc_peer_connections) == 0 else vpc_peer_connections[0]


def get_workspace_vpc_peer_connection_route_entry_name(workspace_name):
    return ALIYUN_WORKSPACE_VPC_PEERING_ROUTE_ENTRY_NAME.format(workspace_name)

def _update_route_tables_for_workspace_vpc_peer_connection(config, vpc_peer_cli):
    provider_config = config["provider"]
    workspace_name = config["workspace_name"]
    current_region_id = get_current_instance_region()
    current_vpc_peer_cli = VpcPeerClient(provider_config, current_region_id)
    current_vpc_cli = VpcClient(provider_config, current_region_id)
    current_vpc = get_current_vpc(config)
    vpc_cli = VpcClient(provider_config)
    workspace_vpc = get_workspace_vpc(workspace_name, vpc_cli)

    current_vpc_route_tables = get_vpc_route_tables(current_vpc, current_vpc_cli)
    workspace_vpc_route_tables = get_vpc_route_tables(workspace_vpc, vpc_cli)

    vpc_peer_connection = get_workspace_vpc_peer_connection(config)
    if vpc_peer_connection is None:
        cli_logger.abort(
            "No vpc peer connection found for workspace: {}.".format(workspace_name))

    for current_vpc_route_table in current_vpc_route_tables:
        current_vpc_cli.create_route_entry(
            route_table_id=current_vpc_route_table.route_table_id,
            cidr_block=workspace_vpc.cidr_block,
            next_hop_id=vpc_peer_connection.instance_id,
            next_hop_type="VpcPeer",
            name=get_workspace_vpc_peer_connection_route_entry_name(workspace_name))
        cli_logger.print(
            "Successfully add route destination to current VPC route table {} with workspace VPC CIDR block.".format(
                current_vpc_route_table.route_table_id))

    for workspace_vpc_route_table in workspace_vpc_route_tables:
        vpc_cli.create_route_entry(
            route_table_id=workspace_vpc_route_table.route_table_id,
            cidr_block=current_vpc.cidr_block,
            next_hop_id=vpc_peer_connection.instance_id,
            next_hop_type="VpcPeer",
            name="cloudtik-{}-vpc-peering-connection-route-entry".format(workspace_name))
        cli_logger.print(
            "Successfully add route destination to current VPC route table {} with workspace VPC CIDR block.".format(
                workspace_vpc_route_table.route_table_id))


def _create_and_configure_vpc_peer_connection(config, vpc_peer_cli):
    current_step = 1
    total_steps = 2

    with cli_logger.group(
            "Creating VPC peer connection",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_workspace_vpc_peer_connection(config, vpc_peer_cli)

    with cli_logger.group(
            "Update route tables for the VPC peer connection",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _update_route_tables_for_workspace_vpc_peer_connection(config, vpc_peer_cli)


def  _create_network_resources(config, current_step, total_steps):
    provider_config = config["provider"]
    ecs_cli = EcsClient(provider_config)
    vpc_cli= VpcClient(provider_config)
    vpc_peer_cli= VpcPeerClient(provider_config)

    # create VPC
    with cli_logger.group(
            "Creating VPC",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        vpc_id = _configure_vpc(config, vpc_cli)

    # create vswitches
    with cli_logger.group(
            "Creating vswitches",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_and_configure_vswitches(config, vpc_cli)

    # create NAT gateway for public subnets
    with cli_logger.group(
            "Creating Internet gateway",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_and_configure_nat_gateway(config, vpc_cli)

    with cli_logger.group(
            "Creating security group",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1

        _upsert_security_group(config, vpc_id, ecs_cli)

    if is_use_peering_vpc(config):
        with cli_logger.group(
                "Creating VPC peering connection",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _create_and_configure_vpc_peer_connection(config, vpc_peer_cli)

    return current_step


def get_managed_oss_bucket(provider_config, workspace_name):
    oss_cli = OssClient(provider_config)
    region = provider_config["region"]
    bucket_name_prefix = "cloudtik-{workspace_name}-{region}-".format(
        workspace_name=workspace_name,
        region=region
    )
    cli_logger.verbose("Getting OSS bucket with prefix: {}.".format(bucket_name_prefix))
    for bucket in oss_cli.list_buckets():
        if bucket_name_prefix in bucket.name:
            cli_logger.verbose("Successfully get the OSS bucket: {}.".format(bucket.name))
            return bucket
    cli_logger.verbose_error("Failed to get the OSS bucket for workspace.")
    return None


def _delete_workspace_cloud_storage(config, workspace_name):
    _delete_managed_cloud_storage(config["provider"], workspace_name)


def _delete_managed_cloud_storage(cloud_provider, workspace_name):
    bucket = get_managed_oss_bucket(cloud_provider, workspace_name)
    if bucket is None:
        cli_logger.warning("No OSS bucket with the name found.")
        return

    try:
        cli_logger.print("Deleting OSS bucket: {}...".format(bucket.name))
        oss_cli = OssClient(cloud_provider)
        objects = oss_cli.list_objects(bucket.name)
        # Delete all objects before deleting the bucket
        for object in objects:
            oss_cli.delete_object(bucket.name, object.key)
        # Delete the bucket
        oss_cli.delete_bucket(bucket.name)
        cli_logger.print("Successfully deleted OSS bucket: {}.".format(bucket.name))
    except Exception as e:
        cli_logger.error("Failed to delete OSS bucket. {}", str(e))
        raise e
    return


def _create_workspace_cloud_storage(config, workspace_name):
    _create_managed_cloud_storage(config["provider"], workspace_name)


def _create_managed_cloud_storage(cloud_provider, workspace_name):
    # If the managed cloud storage for the workspace already exists
    # Skip the creation step
    oss_cli = OssClient(cloud_provider)
    bucket = get_managed_oss_bucket(cloud_provider, workspace_name)
    if bucket is not None:
        cli_logger.print("OSS bucket for the workspace already exists. Skip creation.")
        return

    region = cloud_provider["region"]
    suffix = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))
    bucket_name = "cloudtik-{workspace_name}-{region}-{suffix}".format(
        workspace_name=workspace_name,
        region=region,
        suffix=suffix
    )

    cli_logger.print("Creating OSS bucket for the workspace: {}...".format(workspace_name))
    oss_cli.put_bucket(bucket_name)
    cli_logger.print("Successfully created OSS bucket: {}.".format(bucket_name))


def _create_workspace(config):
    workspace_name = config["workspace_name"]
    managed_cloud_storage = is_managed_cloud_storage(config)
    use_peering_vpc = is_use_peering_vpc(config)

    current_step = 1
    total_steps = ALIYUN_WORKSPACE_NUM_CREATION_STEPS
    if managed_cloud_storage:
        total_steps += 1
    if use_peering_vpc:
        total_steps += 1

    try:
        with cli_logger.group("Creating workspace: {}", workspace_name):
            current_step = _create_network_resources(config, current_step, total_steps)

            with cli_logger.group(
                    "Creating instance role",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _create_workspace_instance_role(config)

            if managed_cloud_storage:
                with cli_logger.group(
                        "Creating OSS  from cloudtik.providers._private.aliyun.utils import AcsClientbucket",
                        _numbered=("[]", current_step, total_steps)):
                    current_step += 1
                    _create_workspace_cloud_storage(config, workspace_name)

    except Exception as e:
        cli_logger.error("Failed to create workspace with the name {}. "
                         "You need to delete and try create again. {}", workspace_name, str(e))
        raise e

    cli_logger.print(
        "Successfully created workspace: {}.",
        cf.bold(workspace_name))

    return config


def _configure_vpc(config, vpc_cli):
    workspace_name = config["workspace_name"]
    use_working_vpc = is_use_working_vpc(config)
    if use_working_vpc:
        # No need to create new vpc
        vpc_name = _get_workspace_vpc_name(workspace_name)
        vpc_id = get_current_vpc_id(config)
        vpc_cli.tag_resource(
            resource_id=vpc_id,
            resource_type="VPC",
            tags=[
                {'Key': 'Name', 'Value': vpc_name},
                {'Key': ALIYUN_WORKSPACE_VERSION_TAG_NAME, 'Value': ALIYUN_WORKSPACE_VERSION_CURRENT}
            ]
        )
        cli_logger.print("Using the existing VPC: {} for workspace. Skip creation.".format(vpc_id))
    else:
        # Need to create a new vpc
        if get_workspace_vpc_id(config, vpc_cli) is None:
            vpc_id = _create_vpc(config, vpc_cli)
        else:
            raise RuntimeError("There is a same name VPC for workspace: {}, "
                               "if you want to create a new workspace with the same name, "
                               "you need to execute workspace delete first!".format(workspace_name))
    return vpc_id


def get_current_instance_region():
    try:
        output = subprocess.Popen("curl http://100.100.100.200/latest/meta-data/region-id", shell=True)
        region_id = output.stdout.readline().decode()
        return region_id
    except Exception as e:
        cli_logger.abort("Failed to get instance region: {}. "
                         "Please make sure your current machine is an Aliyun virtual machine", str(e))
        return None


def get_current_instance_owner_account_id():
    try:
        output = subprocess.Popen("curl http://100.100.100.200/latest/meta-data/owner-account-id", shell=True)
        owner_account_id = output.stdout.readline().decode()
        return owner_account_id
    except Exception as e:
        cli_logger.abort("Failed to get instance owner-account-id: {}. "
                         "Please make sure your current machine is an Aliyun virtual machine", str(e))
        return None


def get_current_instance(ecs_cli):
    ip_address = get_node_ip_address(address="8.8.8.8:53")
    for instance in ecs_cli.describe_instances():
        for network_interface in instance.network_interfaces.network_interface:
            if network_interface.primary_ip_address == ip_address:
                return instance

    raise RuntimeError("Failed to get the instance metadata for the current machine. "
                           "Please make sure your current machine is an Aliyun virtual machine.")


def get_current_vpc(config):
    current_region_id = get_current_instance_region()
    current_vpc_cli = VpcClient(config["provider_config"], current_region_id)
    current_vpc_id = get_current_vpc_id(config)
    current_vpc = current_vpc_cli.describe_vpcs(vpc_id=current_vpc_id)[0]
    return current_vpc


def get_current_vpc_id(config):
    current_region_id = get_current_instance_region()
    current_ecs_client = EcsClient(config["provider_config"], current_region_id)
    ip_address = get_node_ip_address(address="8.8.8.8:53")
    vpc_id = None
    for instance in current_ecs_client.describe_instances():
        for network_interface in instance.network_interfaces.network_interface:
            if network_interface.primary_ip_address == ip_address:
                vpc_id = instance.vpc_attributes.vpc_id

    if vpc_id is None:
        raise RuntimeError("Failed to get the VPC for the current machine. "
                           "Please make sure your current machine is an Aliyun virtual machine.")
    return vpc_id


def get_existing_routes_cidr_block(current_vpc_cli, route_tables):
    existing_routes_cidr_block = set()
    for route_table in route_tables:
        for route_entry in current_vpc_cli.describe_route_entry_list(route_table.route_table_id):
            if route_entry.destination_cidr_block != '0.0.0.0/0':
                existing_routes_cidr_block.add(route_entry.destination_cidr_block)

    return existing_routes_cidr_block


def _configure_peering_vpc_cidr_block(config, current_vpc):
    current_vpc_cidr_block = current_vpc.cidr_block
    current_region_id = get_current_instance_region()
    current_vpc_cli = VpcClient(config["provider"], current_region_id)
    current_vpc_route_tables = get_vpc_route_tables(current_vpc, current_vpc_cli)
    existing_routes_cidr_block = get_existing_routes_cidr_block(current_vpc_cli, current_vpc_route_tables)
    existing_routes_cidr_block.add(current_vpc_cidr_block)

    ip = current_vpc_cidr_block.split("/")[0].split(".")
    for  i in range(0, 256):
        tmp_cidr_block = ip[0] + "." + str(i) + ".0.0/16"

        if check_cidr_conflict(tmp_cidr_block, existing_routes_cidr_block):
            cli_logger.print("Successfully found cidr block for peering VPC.")
            return tmp_cidr_block

    cli_logger.abort("Failed to find non-conflicted cidr block for peering VPC.")


def _create_vpc(config, vpc_cli):
    workspace_name = config["workspace_name"]
    vpc_name = _get_workspace_vpc_name(workspace_name)

    cli_logger.print("Creating workspace VPC: {}...", vpc_name)
    # create vpc
    cidr_block = '10.0.0.0/16'
    if is_use_peering_vpc(config):
        current_vpc = get_current_vpc(config)
        cidr_block = _configure_peering_vpc_cidr_block(config, current_vpc)

    vpc_id = vpc_cli.create_vpc(vpc_name, cidr_block)
    if check_resource_status(MAX_POLLS, POLL_INTERVAL, vpc_cli.describe_vpc_attribute, "Available", vpc_id):
        cli_logger.print("Successfully created workspace VPC: {}.", vpc_name)
    else:
        cli_logger.abort("Failed to create workspace VPC. {}", vpc_name)
    return vpc_id


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
    vpc_cli.delete_vpc(vpc_id)
    cli_logger.print("Successfully deleted the VPC: {}.".format(vpc_name))


def get_workspace_vpc_id(config, vpc_cli):
    return _get_workspace_vpc_id(config["workspace_name"], vpc_cli)


def get_workspace_vpc(config, vpc_cli):
    return _get_workspace_vpc(config["workspace_name"], vpc_cli)


def _get_workspace_vpc(workspace_name, vpc_cli):
    vpc_name = _get_workspace_vpc_name(workspace_name)
    cli_logger.verbose("Getting the VPC for workspace: {}...".format(vpc_name))
    vpcs = vpc_cli.describe_vpcs(vpc_name=vpc_name)
    if len(vpcs) == 0:
        cli_logger.verbose("The VPC for workspace is not found: {}.".format(vpc_name))
        return None
    else:
        cli_logger.verbose_error("Successfully get the VPC Id of {} for workspace.".format(vpc_name))
        return vpcs[0]


def _get_workspace_vpc_id(workspace_name, vpc_cli):
    vpc = _get_workspace_vpc(workspace_name, vpc_cli)
    return None if vpc is None else vpc.vpc_id


def _get_workspace_vpc_name(workspace_name):
    return ALIYUN_WORKSPACE_VPC_NAME.format(workspace_name)


def _configure_vswitches_cidr(vpc, vpc_cli, vswitches_count):
    cidr_list = []
    vswitches = vpc_cli.describe_vswitches(vpc.vpc_id)

    vpc_cidr = vpc.cidr_block
    ip = vpc_cidr.split("/")[0].split(".")

    if len(vswitches) == 0:
        for i in range(0, ALIYUN_VPC_SWITCHES_COUNT):
            cidr_list.append(ip[0] + "." + ip[1] + "." + str(i) + ".0/24")
    else:
        cidr_blocks = [vswitch.cidr_block for vswitch in vswitches]
        for i in range(0, 256):
            tmp_cidr_block = ip[0] + "." + ip[1] + "." + str(i) + ".0/24"

            if check_cidr_conflict(tmp_cidr_block, cidr_blocks):
                cidr_list.append(tmp_cidr_block)

            if len(cidr_list) == vswitches_count:
                break

    return cidr_list


def _create_vswitch_for_nat_gateway(config, vpc_cli):
    workspace_name = config["workspace_name"]
    vpc = get_workspace_vpc(config, vpc_cli)
    vpc_id = vpc.vpc_id
    cidr_list = _configure_vswitches_cidr(vpc, vpc_cli, 1)
    availability_zones_id = [zone.zone_id for zone in vpc_cli.list_enhanced_nat_gateway_available_zones()]
    vswitch_name = ALIYUN_WORKSPACE_NAT_VSWITCH_NAME.format(workspace_name)
    cli_logger.print("Creating vswitch for NAT gateway: {} with CIDR: {}...".format(vswitch_name, cidr_list[0]))

    vswitch_id = vpc_cli.create_vswitch(vpc_id, availability_zones_id[0], cidr_list[0], vswitch_name)

    if check_resource_status(MAX_POLLS, POLL_INTERVAL, vpc_cli.describe_vswitch_attributes, "Available", vswitch_id):
        cli_logger.print("Successfully created vswitch: {}.".format(vswitch_name))
    else:
        cli_logger.abort("Failed to create vswitch: {}.".format(vswitch_name))
    return vswitch_id


def _create_and_configure_vswitches(config, vpc_cli):
    workspace_name = config["workspace_name"]
    vpc = get_workspace_vpc(config, vpc_cli)
    vpc_id = vpc.vpc_id

    vswitches = []
    cidr_list = _configure_vswitches_cidr(vpc, vpc_cli, ALIYUN_VPC_SWITCHES_COUNT)
    cidr_len = len(cidr_list)

    availability_zones_id = [zone.zone_id for zone in vpc_cli.describe_zones()]
    default_availability_zone_id = availability_zones_id[0]
    availability_zones_id = set(availability_zones_id)
    used_availability_zones_id = set()
    last_availability_zone_id = None

    for i in range(0, cidr_len):
        cidr_block = cidr_list[i]
        vswitch_type = "public" if i == 0 else "private"
        with cli_logger.group(
                "Creating {} vswitch", vswitch_type,
                _numbered=("()", i + 1, cidr_len)):
            try:
                cli_logger.print("Creating vswitch for VPC: {} with CIDR: {}...".format(vpc_id, cidr_block))
                if i == 0:
                    vswitch_name = ALIYUN_WORKSPACE_PUBLIC_VSWITCH_NAME.format(workspace_name)
                    vswitch_id = vpc_cli.create_vswitch(vpc_id, default_availability_zone_id, cidr_block, vswitch_name)
                else:
                    if last_availability_zone_id is None:
                        vswitch_name = ALIYUN_WORKSPACE_PRIVATE_VSWITCH_NAME.format(workspace_name)
                        last_availability_zone_id = default_availability_zone_id
                        vswitch_id = vpc_cli.create_vswitch(vpc_id, default_availability_zone_id, cidr_block, vswitch_name)
                        last_availability_zone_id = _next_availability_zone(
                            availability_zones_id, used_availability_zones_id, last_availability_zone_id)
                if check_resource_status(MAX_POLLS, POLL_INTERVAL, vpc_cli.describe_vswitch_attributes, "Available", vswitch_id):
                    cli_logger.print("Successfully created vswitch: {}.".format(vswitch_name))
                else:
                    cli_logger.abort("Failed to create vswitch: {}.".format(vswitch_name))
            except Exception as e:
                cli_logger.error("Failed to create {} vswitch. {}", vswitch_type, str(e))
                raise e
            vswitches.append(vswitch_id)

    assert len(vswitches) == ALIYUN_VPC_SWITCHES_COUNT, "We must create {} vswitches for VPC: {}!".format(
        ALIYUN_VPC_SWITCHES_COUNT, vpc_id)
    return vswitches


def _delete_private_vswitches(workspace_name, vpc_id, vpc_cli):
    _delete_vswitches(workspace_name, vpc_id, vpc_cli, ALIYUN_WORKSPACE_PRIVATE_VSWITCH_NAME)


def _delete_public_vswitches(workspace_name, vpc_id, vpc_cli):
    _delete_vswitches(workspace_name, vpc_id, vpc_cli, ALIYUN_WORKSPACE_PUBLIC_VSWITCH_NAME)


def _delete_nat_vswitches(workspace_name, vpc_id, vpc_cli):
    _delete_vswitches(workspace_name, vpc_id, vpc_cli, ALIYUN_WORKSPACE_NAT_VSWITCH_NAME)


def get_workspace_private_vswitches(workspace_name, vpc_id, vpc_cli):
    return _get_workspace_vswitches(workspace_name, vpc_id, vpc_cli, ALIYUN_WORKSPACE_PRIVATE_VSWITCH_NAME)


def get_workspace_public_vswitches(workspace_name, vpc_id, vpc_cli):
    return _get_workspace_vswitches(workspace_name, vpc_id, vpc_cli, ALIYUN_WORKSPACE_PUBLIC_VSWITCH_NAME)


def get_workspace_nat_vswitches(workspace_name, vpc_id, vpc_cli):
    return _get_workspace_vswitches(workspace_name, vpc_id, vpc_cli, ALIYUN_WORKSPACE_NAT_VSWITCH_NAME)


def _get_workspace_vswitches(workspace_name, vpc_id, vpc_cli, name_pattern):
    vswitches = [vswitch for vswitch in vpc_cli.describe_vswitches(vpc_id)
               if vswitch.v_switch_name.startswith(name_pattern.format(workspace_name))]
    return vswitches


def _delete_vswitches(workspace_name, vpc_id, vpc_cli, name_pattern):
    """ Delete custom vswitches """
    vswitches = _get_workspace_vswitches(workspace_name, vpc_id, vpc_cli, name_pattern)

    if len(vswitches) == 0:
        cli_logger.print("No vswitches for workspace were found under this VPC: {}...".format(vpc_id))
        return

    for vswitch in vswitches:
        vswitch_id = vswitch.v_switch_id
        cli_logger.print("Deleting vswitch: {}...".format(vswitch_id))
        vpc_cli.delete_vswitch(vswitch_id)
        if check_resource_status(MAX_POLLS, POLL_INTERVAL, vpc_cli.describe_vswitch_attributes, "", vswitch_id):
            cli_logger.print("Successfully deleted vswitch: {}.".format(vswitch_id))
        else:
            cli_logger.abort("Failed to delete vswitch: {}.".format(vswitch_id))


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


def _create_workspace_security_group(config, vpc_id, ecs_cli):
    security_group_name = ALIYUN_WORKSPACE_SECURITY_GROUP_NAME.format(config["workspace_name"])
    cli_logger.print("Creating security group for VPC: {} ...".format(vpc_id))
    security_group_id = ecs_cli.create_security_group(vpc_id, security_group_name)
    cli_logger.print("Successfully created security group: {}.".format(security_group_name))
    return security_group_id


def _add_security_group_rules(config, security_group_id, ecs_cli):
    cli_logger.print("Updating rules for security group: {}...".format(security_group_id))
    _update_inbound_rules(security_group_id, config, ecs_cli)
    cli_logger.print("Successfully updated rules for security group.")


def _update_inbound_rules(target_security_group_id, config, ecs_cli):
    extended_rules = config["provider"] \
        .get("security_group_rule", [])
    new_permissions = _create_default_inbound_rules(config, extended_rules)
    security_group_attribute = ecs_cli.describe_security_group_attribute(target_security_group_id)
    old_permissions = security_group_attribute.permissions.permission
    
    # revoke old permissions
    for old_permission in old_permissions:
        ecs_cli.revoke_security_group(
            ip_protocol=old_permission.ip_protocol,
            port_range=old_permission.port_range,
            security_group_id=old_permission.security_group_rule_id,
            source_cidr_ip=old_permission.source_cidr_ip)
    # revoke old permissions
    for new_permission in new_permissions:
        ecs_cli.authorize_security_group(
            ip_protocol=new_permission.get("IpProtocol"),
            port_range=new_permission.get("PortRange"),
            security_group_id=target_security_group_id,
            source_cidr_ip=new_permission.get("SourceCidrIp"))
    

def _create_allow_working_node_inbound_rules(config):
    allow_ssh_only = is_peering_firewall_allow_ssh_only(config)
    vpc = get_current_vpc(config)
    working_vpc_cidr = vpc.cidr_block
    return [{
        "PortRange": "22/22" if allow_ssh_only else "-1/-1",
        "SourceCidrIp": working_vpc_cidr,
        "IpProtocol": "TCP" if allow_ssh_only else "All"
    }]


def _create_default_inbound_rules(config, extended_rules=None):
    if extended_rules is None:
        extended_rules = []
    intra_cluster_rules = _create_default_intra_cluster_inbound_rules(config)

    # TODO: support VPC peering
    if is_use_peering_vpc(config) and is_peering_firewall_allow_working_subnet(config):
        extended_rules += _create_allow_working_node_inbound_rules(config)

    merged_rules = itertools.chain(
        intra_cluster_rules,
        extended_rules,
    )
    return list(merged_rules)


def get_workspace_security_group(config, ecs_cli, vpc_id):
    return _get_security_group(config, vpc_id, ecs_cli, get_workspace_security_group_name(config["workspace_name"]))


def _get_security_group(config, vpc_id, ecs_cli, group_name):
    security_group = _get_security_groups(config, ecs_cli, [vpc_id], [group_name])
    return None if not security_group else security_group[0]


def _get_security_groups(config, ecs_cli, vpc_ids, group_names):
    unique_vpc_ids = list(set(vpc_ids))
    unique_group_names = set(group_names)
    filtered_groups = []
    for vpc_id in unique_vpc_ids:
        security_groups = [security_group for security_group in ecs_cli.describe_security_groups(vpc_id=vpc_id)
                           if security_group.security_group_name in unique_group_names]
        filtered_groups.extend(security_groups)
    return filtered_groups


def get_workspace_security_group_name(workspace_name):
    return ALIYUN_WORKSPACE_SECURITY_GROUP_NAME.format(workspace_name)


def _update_security_group(config, ecs_cli, vpc_cli):
    vpc_id = get_workspace_vpc_id(config, vpc_cli)
    security_group = get_workspace_security_group(config, ecs_cli, vpc_id)
    _add_security_group_rules(config, security_group.security_group_id, ecs_cli)
    return security_group


def _upsert_security_group(config, vpc_id, ecs_cli):
    current_step = 1
    total_steps = 2

    with cli_logger.group(
            "Creating security group for VPC",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        security_group_id = _create_workspace_security_group(config, vpc_id, ecs_cli)

    with cli_logger.group(
            "Configuring rules for security group",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _add_security_group_rules(config, security_group_id, ecs_cli)

    return security_group_id


def _delete_security_group(config, vpc_id, ecs_cli):
    """ Delete any security-groups """
    workspace_security_group = get_workspace_security_group(config, ecs_cli, vpc_id)
    if workspace_security_group is None:
        cli_logger.print("No security groups for workspace were found under this VPC: {}...".format(vpc_id))
        return
    security_group_id = workspace_security_group.security_group_id
    cli_logger.print("Deleting security group: {}...".format(security_group_id))
    ecs_cli.delete_security_group(security_group_id)
    cli_logger.print("Successfully deleted security group: {}.".format(security_group_id))


def _create_default_intra_cluster_inbound_rules(config):
    vpc_cli = VpcClient(config["provider"])
    vpc = get_workspace_vpc(config, vpc_cli)
    vpc_cidr = vpc.cidr_block
    return [{
        "PortRange": "-1/-1",
        "SourceCidrIp": vpc_cidr,
        "IpProtocol": "All"
    }]


def _delete_nat_gateway(config, vpc_cli):
    current_step = 1
    total_steps = 4

    with cli_logger.group(
            "Deleting SNAT Entries",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_snat_entries(config, vpc_cli)

    with cli_logger.group(
            "Dissociating elastic ip",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _dissociate_elastic_ip(config, vpc_cli)

    with cli_logger.group(
            "Deleting NAT Gateway",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_nat_gateway_resource(config, vpc_cli)

    with cli_logger.group(
            "Releasing Elastic IP",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        release_elastic_ip(config, vpc_cli)


def _create_and_configure_nat_gateway(config, vpc_cli):
    current_step = 1
    total_steps = 5

    with cli_logger.group(
            "Creating Elastic IP",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_elastic_ip(config, vpc_cli)

    with cli_logger.group(
            "Creating VSwitch for NAT Gateway",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_vswitch_for_nat_gateway(config, vpc_cli)

    with cli_logger.group(
            "Creating NAT Gateway",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_nat_gateway(config, vpc_cli)

    with cli_logger.group(
            "Associate NAT Gateway with EIP",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _associate_nat_gateway_with_elastic_ip(config, vpc_cli)

    with cli_logger.group(
            "Creating SNAT Entry",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_snat_entries(config, vpc_cli)


def get_workspace_snat_entry_name(workspace_name):
    return ALIYUN_WORKSPACE_SNAT_ENTRY_NAME.format(workspace_name)


def check_snat_entry_status(time_default_out, default_time, check_status, vpc_cli, snat_table_id, snat_entry_id):
    for i in range(time_default_out):
        time.sleep(default_time)
        try:
            snat_entry = vpc_cli.describe_snat_entries(snat_table_id=snat_table_id, snat_entry_id=snat_entry_id)
            status = "" if len(snat_entry) == 0 else snat_entry[0].to_map().get("Status", "")
            if status == check_status:
                return True
        except UnretryableException as e:
            cli_logger.error("Failed to get attributes of resource. {}".format(format_exception_message(str(e))))
            continue
    return False


def _create_snat_entries(config, vpc_cli):
    workspace_name = config["workspace_name"]
    snat_entry_name = get_workspace_snat_entry_name(workspace_name)
    vpc_id = get_workspace_vpc_id(config, vpc_cli)
    private_vswitches = get_workspace_private_vswitches(workspace_name, vpc_id, vpc_cli)
    nat_gateway = get_workspace_nat_gateway(config, vpc_cli)
    snat_table_id = nat_gateway.snat_table_ids.snat_table_id[0]
    elastic_ip = get_workspace_elastic_ip(config, vpc_cli)
    snat_ip = elastic_ip.ip_address
    cli_logger.print("Creating SNAT Entries: {}...".format(snat_entry_name))
    for private_vswitch in private_vswitches:
        snat_entry_id = vpc_cli.create_snat_entry(snat_table_id, private_vswitch.v_switch_id, snat_ip, snat_entry_name)
        if check_snat_entry_status(MAX_POLLS, POLL_INTERVAL, "Available", vpc_cli, snat_table_id, snat_entry_id):
            cli_logger.print("Successfully created SNAT Entry: {}.".format(snat_entry_id))
        else:
            cli_logger.abort("Failed to create SNAT Entry: {}.".format(snat_entry_id))

    cli_logger.print("Successfully created SNAT Entries: {}.".format(snat_entry_name))


def _delete_snat_entries(config, vpc_cli):
    workspace_name = config["workspace_name"]
    snat_entry_name = get_workspace_snat_entry_name(workspace_name)
    nat_gateway = get_workspace_nat_gateway(config, vpc_cli)
    if nat_gateway is None:
        cli_logger.print("Nat gateway does not exist and no need to delete SNAT Entries.")
        return
    snat_table_id = nat_gateway.snat_table_ids.snat_table_id[0]
    cli_logger.print("Deleting SNAT Entries: {}...".format(snat_entry_name))
    for snat_table_entry in vpc_cli.describe_snat_entries(snat_table_id=snat_table_id):
        vpc_cli.delete_snat_entry(snat_table_id, snat_table_entry.snat_entry_id)
        if check_snat_entry_status(MAX_POLLS, POLL_INTERVAL, "", vpc_cli, snat_table_id, snat_table_entry.snat_entry_id):
            cli_logger.print("Successfully deleted SNAT Entry: {}.".format(snat_table_entry.snat_entry_id))
        else:
            cli_logger.abort("Failed to delete SNAT Entry: {}.".format(snat_table_entry.snat_entry_id))
    cli_logger.print("Successfully deleted SNAT Entries: {}.".format(snat_entry_name))


def _delete_nat_gateway_resource(config, vpc_cli):
    """ Delete custom nat_gateway """
    nat_gateway = get_workspace_nat_gateway(config, vpc_cli)
    if nat_gateway is None:
        cli_logger.print("No Nat Gateway for workspace were found...")
        return
    nat_gateway_id = nat_gateway.nat_gateway_id
    nat_gateway_name = nat_gateway.name
    cli_logger.print("Deleting Nat Gateway: {}...".format(nat_gateway_name))
    vpc_cli.delete_nat_gateway(nat_gateway_id)
    if check_resource_status(MAX_POLLS_NAT, POLL_INTERVAL, vpc_cli.get_nat_gateway_attribute, "", nat_gateway_id):
        cli_logger.print("Successfully deleted Nat Gateway: {}.".format(nat_gateway_name))
    else:
        cli_logger.abort("Failed to delete Nat Gateway: {}.".format(nat_gateway_name))


def get_workspace_nat_gateway(config, vpc_cli):
    return _get_workspace_nat_gateway(config, vpc_cli)


def _get_workspace_nat_gateway(config, vpc_cli):
    workspace_name = config["workspace_name"]
    nat_gateway_name = get_workspace_nat_gateway_name(workspace_name)
    vpc_id = get_workspace_vpc_id(config, vpc_cli)
    cli_logger.verbose("Getting the Nat Gateway for workspace: {}...".format(nat_gateway_name))
    nat_gateways = vpc_cli.describe_nat_gateways(vpc_id, nat_gateway_name)
    if len(nat_gateways) == 0:
        cli_logger.verbose("The Nat Gateway for workspace is not found: {}.".format(nat_gateway_name))
        return None
    else:
        cli_logger.verbose_error("Successfully get the Nat Gateway: {} for workspace.".format(nat_gateway_name))
        return nat_gateways[0]


def get_workspace_nat_gateway_name(workspace_name):
    return ALIYUN_WORKSPACE_NAT_GATEWAY_NAME.format(workspace_name)


def _create_nat_gateway(config, vpc_cli):
    workspace_name = config["workspace_name"]
    vpc_id =  get_workspace_vpc_id(config, vpc_cli)
    nat_gateway_name = get_workspace_nat_gateway_name(workspace_name)
    nat_switch = get_workspace_nat_vswitches(workspace_name, vpc_id, vpc_cli)[0]
    nat_switch_id = nat_switch.v_switch_id
    cli_logger.print("Creating nat-gateway: {}...".format(nat_gateway_name))
    nat_gateway_id = vpc_cli.create_nat_gateway(vpc_id, nat_switch_id, nat_gateway_name)
    if check_resource_status(MAX_POLLS_NAT, POLL_INTERVAL, vpc_cli.get_nat_gateway_attribute, "Available", nat_gateway_id):
        cli_logger.print("Successfully created Nat Gateway: {}.".format(nat_gateway_name))
    else:
        cli_logger.abort("Failed to create Nat Gateway: {}.".format(nat_gateway_name))
    return nat_gateway_id


def _associate_nat_gateway_with_elastic_ip(config, vpc_cli):
    elastic_ip = get_workspace_elastic_ip(config, vpc_cli)
    eip_allocation_id = elastic_ip.allocation_id
    nat_gateway = get_workspace_nat_gateway(config, vpc_cli)
    instance_id = nat_gateway.nat_gateway_id
    cli_logger.print("Associating NAT gateway with Elastic IP...")
    vpc_cli.associate_eip_address(eip_allocation_id, instance_id, "Nat")
    if check_resource_status(MAX_POLLS_NAT, POLL_INTERVAL, vpc_cli.describe_eip_addresses, "InUse", eip_allocation_id):
        cli_logger.print("Successfully associated NAT gateway with Elastic IP.")
    else:
        cli_logger.abort("Faild to associate NAT gateway with Elastic IP.")


def get_workspace_elastic_ip_name(workspace_name):
    return ALIYUN_WORKSPACE_EIP_NAME.format(workspace_name)


def _create_elastic_ip(config, vpc_cli):
    eip_name = get_workspace_elastic_ip_name(config["workspace_name"])
    allocation_id = vpc_cli.allocate_eip_address(eip_name)
    if check_resource_status(MAX_POLLS_NAT, POLL_INTERVAL, vpc_cli.describe_eip_addresses, "Available", allocation_id):
        cli_logger.print("Successfully allocate Elastic IP:{}.".format(eip_name))
    else:
        cli_logger.print("Faild to allocate Elastic IP:{}.".format(eip_name))


def get_workspace_elastic_ip(config, vpc_cli):
    return _get_workspace_elastic_ip(config, vpc_cli)


def _get_workspace_elastic_ip(config, vpc_cli):
    workspace_name = config["workspace_name"]
    elastic_ip_name = get_workspace_elastic_ip_name(workspace_name)
    cli_logger.verbose("Getting the Elastic IP for workspace: {}...".format(elastic_ip_name))
    eip_addresses = vpc_cli.describe_eip_addresses(eip_name=elastic_ip_name)
    if len(eip_addresses) == 0:
        cli_logger.verbose("The Elastic IP for workspace is not found: {}.".format(elastic_ip_name))
        return None
    else:
        cli_logger.verbose_error("Successfully get the Elastic IP: {} for workspace.".format(elastic_ip_name))
        return eip_addresses[0]


def release_elastic_ip(config, vpc_cli):
    """ Release Elastic IP """
    elastic_ip = get_workspace_elastic_ip(config, vpc_cli)
    if elastic_ip is None:
        cli_logger.print("No Elastic IP for workspace were found...")
        return
    allocation_id = elastic_ip.allocation_id
    elastic_ip_name = elastic_ip.name
    cli_logger.print("Releasing Elastic IP: {}...".format(elastic_ip_name))
    vpc_cli.release_eip_address(allocation_id)
    cli_logger.print("Successfully to release Elastic IP:{}.".format(elastic_ip_name))


def _dissociate_elastic_ip(config, vpc_cli):
    elastic_ip = get_workspace_elastic_ip(config, vpc_cli)
    if elastic_ip is None:
        return
    eip_allocation_id = elastic_ip.allocation_id
    instance_id = elastic_ip.instance_id
    if instance_id == "":
        cli_logger.print("No instance associated with this EIP.")
        return
    elastic_ip_name = elastic_ip.name
    cli_logger.print("Dissociating Elastic IP: {}...".format(elastic_ip_name))
    vpc_cli.unassociate_eip_address(eip_allocation_id, instance_id, "Nat")

    if check_resource_status(MAX_POLLS_NAT, POLL_INTERVAL, vpc_cli.describe_eip_addresses, "Available", eip_allocation_id):
        cli_logger.print("Successfully dissociated Elastic IP:{}.".format(elastic_ip_name))
    else:
        cli_logger.print("Faild to dissociate Elastic IP:{}.".format(elastic_ip_name))


def _get_instance_role(ram_cli, role_name):
    return ram_cli.get_role(role_name)


def _get_head_instance_role_name(workspace_name):
    return "cloudtik-{}-head-role".format(workspace_name)


def _get_worker_instance_role_name(workspace_name):
    return "cloudtik-{}-worker-role".format(workspace_name)


def _get_head_instance_role(config, ram_cli):
    workspace_name = config["workspace_name"]
    head_instance_role_name = _get_head_instance_role_name(workspace_name)
    return _get_instance_role(ram_cli, head_instance_role_name)


def _get_worker_instance_role(config, ram_cli):
    workspace_name = config["workspace_name"]
    worker_instance_role_name = _get_worker_instance_role_name(workspace_name)
    return _get_instance_role(ram_cli, worker_instance_role_name)


def _delete_instance_role(config, ram_cli, instance_role_name):
    cli_logger.print("Deleting instance role: {}...".format(instance_role_name))
    role = _get_instance_role(ram_cli, instance_role_name)

    if role is None:
        cli_logger.warning("No instance role was found.")
        return

    policies = ram_cli.list_policy_for_role(instance_role_name)

    # Detach all policies from instance role
    for policy in policies:
        policy_type = policy.policy_type
        policy_name = policy.policy_name
        ram_cli.detach_policy_from_role(instance_role_name, policy_type, policy_name)

    # Delete the specified instance role. The instance role must not have an associated policies.
        ram_cli.delete_role(instance_role_name)

    cli_logger.print("Successfully deleted instance role.")


def _delete_instance_role_for_head(config, ram_cli):
    head_instance_role_name = _get_head_instance_role_name(config["workspace_name"])
    _delete_instance_role(config, ram_cli, head_instance_role_name)


def _delete_instance_role_for_worker(config, ram_cli):
    worker_instance_role_name = _get_worker_instance_role_name(config["workspace_name"])
    _delete_instance_role(config, ram_cli, worker_instance_role_name)


def _delete_workspace_instance_role(config):
    current_step = 1
    total_steps = 2
    ram_cli = RamClient(config["provider"])

    with cli_logger.group(
            "Deleting instance role for head",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_instance_role_for_head(config, ram_cli)

    with cli_logger.group(
            "Deleting instance role for worker",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_instance_role_for_worker(config, ram_cli)


def _create_or_update_instance_role(config, ram_cli, instance_role_name, is_head=True):
    role = _get_instance_role(ram_cli, instance_role_name)

    if role is None:
        cli_logger.verbose(
            "Creating new RAM instance role {} for use as the default.",
            cf.bold(instance_role_name))
        assume_role_policy_document = '''{
            "Statement": [
                {
                    "Action": "sts:AssumeRole",
                    "Effect": "Allow",
                    "Principal": {
                        "Service": [
                            "ecs.aliyuncs.com"
                        ]
                    }
                }
            ],
            "Version": "1"
        }'''
        ram_cli.create_role(instance_role_name, assume_role_policy_document)
        role = _get_instance_role(ram_cli, instance_role_name)
        assert role is not None, "Failed to create role"

        attach_policies = HEAD_ROLE_ATTACH_POLICIES if is_head else WORKER_ROLE_ATTACH_POLICIES
        for policy in attach_policies:
            ram_cli.attach_policy_to_role(instance_role_name, "System", policy)


def _create_instance_role_for_head(config, ram_cli):
    head_instance_role_name = _get_head_instance_role_name(config["workspace_name"])
    cli_logger.print("Creating head instance role: {}...".format(head_instance_role_name))
    _create_or_update_instance_role(config, ram_cli, head_instance_role_name)
    cli_logger.print("Successfully created and configured head instance role.")


def _create_instance_role_for_worker(config, ram_cli):
    worker_instance_role_name = _get_worker_instance_role_name(config["workspace_name"])
    cli_logger.print("Creating worker instance role: {}...".format(worker_instance_role_name))
    _create_or_update_instance_role(config, ram_cli, worker_instance_role_name, is_head=False)
    cli_logger.print("Successfully created and configured worker instance role.")


def _create_workspace_instance_role(config):
    current_step = 1
    total_steps = 2
    ram_cli = RamClient(config["provider"])
    
    with cli_logger.group(
            "Creating instance role for head",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_instance_role_for_head(config, ram_cli)

    with cli_logger.group(
            "Creating instance role for worker",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_instance_role_for_worker(config, ram_cli)


def create_aliyun_workspace(config):
    # create a copy of the input config to modify
    config = copy.deepcopy(config)

    # create workspace
    config = _create_workspace(config)

    return config


def delete_aliyun_workspace(config, delete_managed_storage: bool = False):
    workspace_name = config["workspace_name"]
    use_peering_vpc = is_use_peering_vpc(config)
    managed_cloud_storage = is_managed_cloud_storage(config)
    vpc_cli = VpcClient(config["provider"])
    vpc_id = _get_workspace_vpc_id(workspace_name, vpc_cli)

    current_step = 1
    total_steps = ALIYUN_WORKSPACE_NUM_DELETION_STEPS
    if vpc_id is None:
        total_steps = 1
    else:
        if use_peering_vpc:
            total_steps += 1
    if managed_cloud_storage and delete_managed_storage:
        total_steps += 1

    try:

        with cli_logger.group("Deleting workspace: {}", workspace_name):
            # Delete in a reverse way of creating
            if managed_cloud_storage and delete_managed_storage:
                with cli_logger.group(
                        "Deleting OSS bucket",
                        _numbered=("[]", current_step, total_steps)):
                    current_step += 1
                    _delete_workspace_cloud_storage(config, workspace_name)

            with cli_logger.group(
                    "Deleting instance role",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _delete_workspace_instance_role(config)

            if vpc_id:
                _delete_network_resources(config, workspace_name, vpc_id,
                                          current_step, total_steps)

    except Exception as e:
        cli_logger.error(
            "Failed to delete workspace {}. {}", workspace_name, str(e))
        raise e

    cli_logger.print(
            "Successfully deleted workspace: {}.",
            cf.bold(workspace_name))
    return None


def _delete_routes_for_workspace_vpc_peering_connection(config, vpc_peer_cli):
    workspace_name = config["workspace_name"]
    provider_config = config["provider"]
    current_region_id = get_current_instance_region()
    current_vpc_cli = VpcClient(provider_config, current_region_id)
    current_vpc = get_current_vpc(config)
    vpc_cli = VpcClient(provider_config)
    workspace_vpc = get_workspace_vpc(config, vpc_cli)

    current_vpc_route_tables = get_vpc_route_tables(current_vpc, current_vpc_cli)
    workspace_vpc_route_tables = get_vpc_route_tables(workspace_vpc, vpc_cli)

    vpc_peer_connection = get_workspace_vpc_peer_connection(config)
    if vpc_peer_connection is None:
        cli_logger.print("No VPC peering connection was found in workspace. Skip delete "
                         "routes for workspace vpc peering connection.")
        return
    route_entry_name = get_workspace_vpc_peer_connection_route_entry_name(workspace_name)
    for current_vpc_route_table in current_vpc_route_tables:
        for route_entry in current_vpc_cli.describe_route_entry_list(
                route_table_id=current_vpc_route_table.route_table_id,
                cidr_block=workspace_vpc.vpc_id,
                entry_name=route_entry_name):
            current_vpc_cli.delete_route_entry(route_entry.route_entry_id)
            cli_logger.print(
                "Successfully delete the route entry about VPC peering connection for current VPC route table {}.".format(
                    current_vpc_route_table.route_table_id))

    for workspace_vpc_route_table in workspace_vpc_route_tables:
        for route_entry in vpc_cli.describe_route_entry_list(
                route_table_id=workspace_vpc_route_table.route_table_id,
                cidr_block=current_vpc.vpc_id,
                entry_name=route_entry_name):
            vpc_cli.delete_route_entry(route_entry.route_entry_id)
            cli_logger.print(
                "Successfully delete the route about VPC peering connection for workspace VPC route table {}.".format(
                    workspace_vpc_route_table.route_table_id))


def _delete_workspace_vpc_peering_connection(config, vpc_peer_cli):
    vpc_peering_connection = get_workspace_vpc_peer_connection(config)
    if vpc_peering_connection is None:
        cli_logger.print("No VPC peering connection was found in workspace.")
        return
    vpc_peering_connection_id = vpc_peering_connection.instance_id
    current_region_id = get_current_instance_region()
    current_vpc_peer_cli = VpcPeerClient(config["provider"], current_region_id)
    current_vpc_peer_cli.delete_vpc_peer_connection(vpc_peering_connection_id)
    cli_logger.print("Successfully deleted VPC peering connection for: {}.".format(vpc_peering_connection_id))


def _delete_workspace_vpc_peer_connection_and_routes(config, vpc_peer_cli):
    current_step = 1
    total_steps = 2

    with cli_logger.group(
            "Deleting routes for VPC peering connection",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_routes_for_workspace_vpc_peering_connection(config, vpc_peer_cli)

    with cli_logger.group(
            "Deleting VPC peering connection",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_workspace_vpc_peering_connection(config, vpc_peer_cli)


def _delete_network_resources(config, workspace_name, vpc_id,
                              current_step, total_steps):
    use_working_vpc = is_use_working_vpc(config)
    use_peering_vpc = is_use_peering_vpc(config)
    provider_config = config["provider"]
    ecs_cli = EcsClient(provider_config)
    vpc_cli = VpcClient(provider_config)
    vpc_peer_cli = VpcPeerClient(provider_config)

    """
         Do the work - order of operation:
         Delete vpc peer connection
         Delete private vswitches
         Delete nat-gateway for private vswitches
         Delete vswitches for nat-gateway
         Delete public vswitches
         Delete security group
         Delete vpc
    """

    # delete vpc peering connection
    if use_peering_vpc:
        with cli_logger.group(
                "Deleting VPC peer connection",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _delete_workspace_vpc_peer_connection_and_routes(config, vpc_peer_cli)

    # delete nat-gateway
    with cli_logger.group(
            "Deleting NAT gateway",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_nat_gateway(config, vpc_cli)

    # delete private vswitches
    with cli_logger.group(
            "Deleting private vswitches",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_private_vswitches(workspace_name, vpc_id, vpc_cli)

    # delete nat-gateway
    with cli_logger.group(
            "Deleting nat vswitches",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_nat_vswitches(workspace_name, vpc_id, vpc_cli)

    # delete public vswitches
    with cli_logger.group(
            "Deleting public vswitches",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_public_vswitches(workspace_name, vpc_id, vpc_cli)

    # delete security group
    with cli_logger.group(
            "Deleting security group",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_security_group(config, vpc_id, ecs_cli)

    # delete vpc
    with cli_logger.group(
            "Deleting VPC",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        if not use_working_vpc:
            _delete_vpc(config, vpc_cli)
        else:
            # deleting the tags we created on working vpc
            _delete_vpc_tags(vpc_cli, vpc_id)


def _delete_vpc_tags(vpc_cli, vpc_id):
    vpc = vpc_cli.describe_vpcs(vpc_id=vpc_id)[0]
    cli_logger.print("Deleting VPC tags: {}...".format(vpc.id))
    vpc_cli.untag_resource(
        resource_id=vpc_id,
        resource_type="VPC",
        tag_keys=['Name', ALIYUN_WORKSPACE_VERSION_TAG_NAME]
    )
    cli_logger.print("Successfully deleted VPC tags: {}.".format(vpc.id))


def check_aliyun_workspace_integrity(config):
    existence = check_aliyun_workspace_existence(config)
    return True if existence == Existence.COMPLETED else False


def update_aliyun_workspace_firewalls(config):
    workspace_name = config["workspace_name"]
    ecs_cli = EcsClient(config["provider"])
    vpc_cli = VpcClient(config["provider"])
    vpc_id = get_workspace_vpc_id(config, vpc_cli)
    if vpc_id is None:
        cli_logger.print("The workspace: {} doesn't exist!".format(config["workspace_name"]))
        return

    current_step = 1
    total_steps = 1

    try:
        with cli_logger.group(
                "Updating workspace firewalls",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _update_security_group(config, ecs_cli, vpc_cli)

    except Exception as e:
        cli_logger.error(
            "Failed to update the firewalls of workspace {}. {}", workspace_name, str(e))
        raise e

    cli_logger.print(
        "Successfully updated the firewalls of workspace: {}.",
        cf.bold(workspace_name))
    return None


def _configure_allowed_ssh_sources(config):
    provider_config = config["provider"]
    if "allowed_ssh_sources" not in provider_config:
        return

    allowed_ssh_sources = provider_config["allowed_ssh_sources"]
    if len(allowed_ssh_sources) == 0:
        return

    if "security_group_rule" not in provider_config:
        provider_config["security_group_rule"] = []
    security_group_rule_config = provider_config["security_group_rule"]

    for allowed_ssh_source in allowed_ssh_sources:
        permission = {
            "IpProtocol": "TCP",
            "PortRange": "22/22",
            "SourceCidrIp": allowed_ssh_source
        }
        security_group_rule_config.append(permission)


def get_workspace_oss_bucket(config, workspace_name):
    return get_managed_oss_bucket(config["provider"], workspace_name)


def _get_workspace_head_nodes(provider_config, workspace_name):
    vpc_client = VpcClient(provider_config)
    vpc_id = _get_workspace_vpc_id(workspace_name, vpc_client)
    if vpc_id is None:
        raise RuntimeError(
            "Failed to get the VPC. The workspace {} doesn't exist or is in the wrong state.".format(
                workspace_name
            ))

    # List the nodes filtering the vpc_id and running state and head tag
    ecs_client = EcsClient(provider_config)
    tag_filters = [
        {
            "Key": CLOUDTIK_TAG_NODE_KIND,
            "Value": NODE_KIND_HEAD
        }
    ]
    nodes = list(ecs_client.describe_instances(
        tags=tag_filters, vpc_id=vpc_id, status="Running"))
    return nodes


def get_workspace_head_nodes(config):
    return _get_workspace_head_nodes(
        config["provider"], config["workspace_name"])


def get_cluster_name_from_head(head_node) -> Optional[str]:
    if (head_node.tags is not None
            and head_node.tags.tag is not None):
        for tag in head_node.tags.tag:
            tag_key = tag.tag_key
            if tag_key == CLOUDTIK_TAG_CLUSTER_NAME:
                return tag.tag_value
    return None


def list_aliyun_clusters(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    head_nodes = get_workspace_head_nodes(config)
    clusters = {}
    for head_node in head_nodes:
        cluster_name = get_cluster_name_from_head(head_node)
        if cluster_name:
            clusters[cluster_name] = _get_node_info(head_node)
    return clusters


def bootstrap_aliyun_workspace(config):
    # create a copy of the input config to modify
    config = copy.deepcopy(config)
    _configure_allowed_ssh_sources(config)
    return config


def check_aliyun_workspace_existence(config):
    workspace_name = config["workspace_name"]
    managed_cloud_storage = is_managed_cloud_storage(config)
    use_peering_vpc = is_use_peering_vpc(config)
    provider_config = config["provider"]
    ecs_cli = EcsClient(provider_config)
    ram_cli = RamClient(provider_config)
    vpc_cli = VpcClient(provider_config)

    existing_resources = 0
    target_resources = ALIYUN_WORKSPACE_TARGET_RESOURCES
    if managed_cloud_storage:
        target_resources += 1
    if use_peering_vpc:
        target_resources += 1

    """
         Do the work - order of operation:
         Check VPC
         Check private vswitches
         Check public vswitches
         Check vswitches for nat-gateways
         Check nat-gateways
         Check security-group
         Check VPC peering if needed
         Instance roles
         Check OSS bucket
    """
    skipped_resources = 0
    vpc_id = _get_workspace_vpc_id(workspace_name, vpc_cli)
    if vpc_id is not None:
        existing_resources += 1
        # Network resources that depending on VPC
        if len(get_workspace_private_vswitches(workspace_name, vpc_id, vpc_cli)) >= ALIYUN_VPC_SWITCHES_COUNT - 1:
            existing_resources += 1
        if len(get_workspace_public_vswitches(workspace_name, vpc_id, vpc_cli)) > 0:
            existing_resources += 1
        if len(get_workspace_nat_vswitches(workspace_name, vpc_id, vpc_cli)) > 0:
            existing_resources += 1
        if get_workspace_nat_gateway(config, vpc_cli) is not None:
            existing_resources += 1
        if get_workspace_security_group(config, ecs_cli, vpc_id) is not None:
            existing_resources += 1
        if use_peering_vpc:
            if get_workspace_vpc_peer_connection(config) is not None:
                existing_resources += 1

    if _get_head_instance_role(config, ram_cli) is not None:
        existing_resources += 1

    if _get_worker_instance_role(config, ram_cli) is not None:
        existing_resources += 1

    cloud_storage_existence = False
    if managed_cloud_storage:
        if get_workspace_oss_bucket(config, workspace_name) is not None:
            existing_resources += 1
            cloud_storage_existence = True

    if existing_resources <= skipped_resources:
        return Existence.NOT_EXIST
    elif existing_resources == target_resources:
        return Existence.COMPLETED
    else:
        if existing_resources == skipped_resources + 1 and cloud_storage_existence:
            return Existence.STORAGE_ONLY
        return Existence.IN_COMPLETED


def get_aliyun_workspace_info(config):
    info = {}
    get_aliyun_managed_cloud_storage_info(config, config["provider"], info)
    return info


def get_aliyun_managed_cloud_storage_info(config, cloud_provider, info):
    workspace_name = config["workspace_name"]
    bucket = get_managed_oss_bucket(cloud_provider, workspace_name)
    managed_bucket_name = None if bucket is None else bucket.name

    if managed_bucket_name is not None:
        oss_cloud_storage = {ALIYUN_OSS_BUCKET, managed_bucket_name}
        managed_cloud_storage = {ALIYUN_MANAGED_STORAGE_OSS_BUCKET: managed_bucket_name,
                                 CLOUDTIK_MANAGED_CLOUD_STORAGE_URI: get_aliyun_cloud_storage_uri(oss_cloud_storage)}
        info[CLOUDTIK_MANAGED_CLOUD_STORAGE] = managed_cloud_storage
