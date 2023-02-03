import logging
import copy
import os
import stat
from typing import Any, Dict, Optional

from cloudtik.core._private.cli_logger import cli_logger, cf
from cloudtik.core._private.utils import check_cidr_conflict, get_cluster_uri, is_use_internal_ip, \
    is_managed_cloud_storage, is_use_managed_cloud_storage, is_worker_role_for_cloud_storage, is_use_working_vpc, \
    is_use_peering_vpc, is_peering_firewall_allow_ssh_only, is_peering_firewall_allow_working_subnet
from cloudtik.core.workspace_provider import Existence, CLOUDTIK_MANAGED_CLOUD_STORAGE, \
    CLOUDTIK_MANAGED_CLOUD_STORAGE_URI
from cloudtik.providers._private.aliyun.utils import AcsClient


# instance status
PENDING = "Pending"
RUNNING = "Running"
STARTING = "Starting"
STOPPING = "Stopping"
STOPPED = "Stopped"

logger = logging.getLogger(__name__)

ALIYUN_WORKSPACE_NUM_CREATION_STEPS = 8
ALIYUN_WORKSPACE_NUM_DELETION_STEPS = 9
ALIYUN_WORKSPACE_TARGET_RESOURCES = 10

ALIYUN_RESOURCE_NAME_PREFIX = "cloudtik"
ALIYUN_WORKSPACE_VPC_NAME = ALIYUN_RESOURCE_NAME_PREFIX + "-{}-vpc"

def bootstrap_aliyun(config):
    # print(config["provider"])
    # create vpc
    # _get_or_create_vpc(config)

    # create security group id
    _get_or_create_security_group(config)
    # create vswitch
    _get_or_create_vswitch(config)
    # create key pair
    _get_or_import_key_pair(config)
    # print(config["provider"])
    return config


def _client(config):
    return AcsClient(
        access_key=config["provider"].get("access_key"),
        access_key_secret=config["provider"].get("access_key_secret"),
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
        config["provider"]["vpc_id"] = vpcs[0].get("VpcId")
        return

    vpc_id = cli.create_vpc()
    if vpc_id is not None:
        config["provider"]["vpc_id"] = vpc_id


def _get_or_create_vswitch(config):
    cli = _client(config)
    vswitches = cli.describe_v_switches(vpc_id=config["provider"]["vpc_id"])
    if vswitches is not None and len(vswitches) > 0:
        config["provider"]["v_switch_id"] = vswitches[0].get("VSwitchId")
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


def _create_workspace(config):
    acs_client = _client(config)
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
            current_step = _create_network_resources(config, acs_client,
                                                     current_step, total_steps)

            with cli_logger.group(
                    "Creating instance profile",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _create_workspace_instance_profile(config, workspace_name)

            if managed_cloud_storage:
                with cli_logger.group(
                        "Creating S3 from cloudtik.providers._private.aliyun.utils import AcsClientbucket",
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


def _create_vpc(config, asc_client):
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

    vpc_id = asc_client.create_vpc(vpc_name, cidr_block)
    if vpc_id is None:
        cli_logger.print("Successfully created workspace VPC: {}.", vpc_name)
        return vpc_id
    else:
        cli_logger.abort("Failed to create workspace VPC.")


def _delete_vpc(config, asc_client):
    use_working_vpc = is_use_working_vpc(config)
    if use_working_vpc:
        cli_logger.print("Will not delete the current working VPC.")
        return

    vpc_id = get_workspace_vpc_id(config, asc_client)
    vpc_name = _get_workspace_vpc_name(config["workspace_name"])

    if vpc_id is None:
        cli_logger.print("The VPC: {} doesn't exist.".format(vpc_name))
        return

    """ Delete the VPC """
    cli_logger.print("Deleting the VPC: {}...".format(vpc_name))

    requestId = asc_client.delete_vpc(vpc_id)
    if requestId is None:
        cli_logger.print("Successfully deleted the VPC: {}.".format(vpc_name))
    else:
        cli_logger.abort("Failed to delete the VPC: {}.")


def get_workspace_vpc_id(config, asc_client):
    return _get_workspace_vpc_id(config["workspace_name"], asc_client)


def _get_workspace_vpc_id(workspace_name, asc_client):
    vpc_name = _get_workspace_vpc_name(workspace_name)
    cli_logger.verbose("Getting the VPC Id for workspace: {}...".format(vpc_name))
    vpcs = asc_client.describe_vpcs()
    if vpcs is None:
        cli_logger.verbose("The VPC for workspace is not found: {}.".format(vpc_name))
        return None

    vpc_ids = [vpc.get('VpcId') for vpc in asc_client.describe_vpcs() if vpc.get('VpcName') == vpc_name]
    if len(vpc_ids) == 0:
        cli_logger.verbose("The VPC for workspace is not found: {}.".format(vpc_name))
        return None
    else:
        cli_logger.verbose_error("Successfully get the VPC Id of {} for workspace.".format(vpc_name))
        return vpc_ids[0]


def _get_workspace_vpc_name(workspace_name):
    return ALIYUN_WORKSPACE_VPC_NAME.format(workspace_name)


def create_aliyun_workspace(config):
    # create a copy of the input config to modify
    config = copy.deepcopy(config)

    # create workspace
    config = _create_workspace(config)

    return config


def delete_aliyun_workspace(config, delete_managed_storage: bool = False):
    pass


def check_aliyun_workspace_integrity(config):
    # existence = check_azure_workspace_existence(config)
    # return True if existence == Existence.COMPLETED else False
    pass


def update_aliyun_workspace_firewalls(config):
    pass


def _get_workspace_head_nodes(provider_config, workspace_name):
    pass


def list_aliyun_clusters(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    pass


def bootstrap_aliyun_workspace(config):
    pass


def check_aliyun_workspace_existence(config):
    pass


def get_aliyun_workspace_info(config):
    pass