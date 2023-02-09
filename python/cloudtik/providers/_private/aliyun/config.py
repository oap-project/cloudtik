import logging
import copy
import os
import stat
import itertools
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
ALIYUN_VPC_SWITCHES_COUNT=2

ALIYUN_RESOURCE_NAME_PREFIX = "cloudtik"
ALIYUN_WORKSPACE_VPC_NAME = ALIYUN_RESOURCE_NAME_PREFIX + "-{}-vpc"
ALIYUN_WORKSPACE_SECURITY_GROUP_NAME = ALIYUN_RESOURCE_NAME_PREFIX + "-{}-security-group"
ALIYUN_WORKSPACE_EIP_NAME = ALIYUN_RESOURCE_NAME_PREFIX + "-{}-eip"
ALIYUN_WORKSPACE_NAT_GATEWAY_NAME = ALIYUN_RESOURCE_NAME_PREFIX + "-{}-nat"
ALIYUN_WORKSPACE_SNAT_ENTRY_NAME = ALIYUN_RESOURCE_NAME_PREFIX + "-{}-snat"

HEAD_ROLE_ATTACH_POLICIES = [
    "AliyunECSFullAccess",
    "AliyunOSSFullAccess",
    "AliyunRAMFullAccess"
]

WORKER_ROLE_ATTACH_POLICIES = [
    "AliyunOSSFullAccess",
]

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


def  _create_network_resources(config, acs_client, current_step, total_steps):
    # TODO
    asc_client = _client(config)
    _create_vpc(config, asc_client)
    

    return current_step


def _create_workspace_cloud_storage(config, workspace_name):
    # TODO
    pass


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
                _create_workspace_instance_role(config, workspace_name)

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

    response = asc_client.delete_vpc(vpc_id)
    if response is None:
        cli_logger.print("Successfully deleted the VPC: {}.".format(vpc_name))
    else:
        cli_logger.abort("Failed to delete the VPC: {}.")


def get_workspace_vpc_id(config, asc_client):
    return _get_workspace_vpc_id(config["workspace_name"], asc_client)


def get_workspace_vpc(config, asc_client):
    return _get_workspace_vpc(config["workspace_name"], asc_client)


def _get_workspace_vpc(workspace_name, asc_client):
    vpc_name = _get_workspace_vpc_name(workspace_name)
    cli_logger.verbose("Getting the VPC for workspace: {}...".format(vpc_name))
    all_vpcs = asc_client.describe_vpcs()
    if all_vpcs is None:
        cli_logger.verbose("The VPC for workspace is not found: {}.".format(vpc_name))
        return None

    vpcs = [vpc for vpc in all_vpcs if vpc.get('VpcName') == vpc_name]
    if len(vpcs) == 0:
        cli_logger.verbose("The VPC for workspace is not found: {}.".format(vpc_name))
        return None
    else:
        cli_logger.verbose_error("Successfully get the VPC Id of {} for workspace.".format(vpc_name))
        return vpcs[0]


def _get_workspace_vpc_id(workspace_name, asc_client):
    vpc = _get_workspace_vpc(workspace_name, asc_client)
    return vpc.get('VpcId')



def _get_workspace_vpc_name(workspace_name):
    return ALIYUN_WORKSPACE_VPC_NAME.format(workspace_name)


def _configure_vswitches_cidr(vpc, asc_client):
    cidr_list = []
    vswitches = asc_client.describe_v_switches(vpc.get("VpcId"))

    vpc_cidr = vpc.get("CidrBlock")
    ip = vpc_cidr.split("/")[0].split(".")

    if len(vswitches) == 0:
        for i in range(0, ALIYUN_VPC_SWITCHES_COUNT):
            cidr_list.append(ip[0] + "." + ip[1] + "." + str(i) + ".0/24")
    else:
        cidr_blocks = [vswitch.get("CidrBlock") for vswitch in vswitches]
        for i in range(0, 256):
            tmp_cidr_block = ip[0] + "." + ip[1] + "." + str(i) + ".0/24"

            if check_cidr_conflict(tmp_cidr_block, cidr_blocks):
                cidr_list.append(tmp_cidr_block)

            if len(cidr_list) == ALIYUN_VPC_SWITCHES_COUNT:
                break

    return cidr_list


def _create_and_configure_vswitches(config, asc_client):
    workspace_name = config["workspace_name"]
    vpc = get_workspace_vpc(config, asc_client)
    vpc_id = get_workspace_vpc_id(config, asc_client)

    vswitches = []
    cidr_list = _configure_vswitches_cidr(vpc, asc_client)
    cidr_len = len(cidr_list)

    zones = asc_client.describe_zones()
    if zones is None:
        cli_logger.abort("No available zones found.")
    availability_zones_id = [zone.get("ZoneId") for zone in zones if len(zone['AvailableInstanceTypes']['InstanceTypes']) > 0]
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
                if i == 0:
                    vswitch = _create_vswitch(asc_client, default_availability_zone_id, workspace_name, vpc_id, cidr_block, isPrivate=False)
                else:
                    if last_availability_zone_id is None:
                        last_availability_zone_id = default_availability_zone_id

                        vswitch = _create_vswitch(asc_client, default_availability_zone_id, workspace_name, vpc_id, cidr_block)

                        last_availability_zone_id = _next_availability_zone(
                            availability_zones_id, used_availability_zones_id, last_availability_zone_id)

            except Exception as e:
                cli_logger.error("Failed to create {} vswitch. {}", vswitch_type, str(e))
                raise e
            vswitches.append(vswitch)

    assert len(vswitches) == ALIYUN_VPC_SWITCHES_COUNT, "We must create {} vswitches for VPC: {}!".format(
        ALIYUN_VPC_SWITCHES_COUNT, vpc_id)
    return vswitches


def _create_vswitch(asc_client, zone_id, workspace_name, vpc_id, cidr_block, isPrivate=True):
    vswitch_type = "private" if isPrivate else "public"
    cli_logger.print("Creating {} vswitch for VPC: {} with CIDR: {}...".format(vswitch_type, vpc_id, cidr_block))
    vswitch_name = 'cloudtik-{}-{}-vswitch'.format(workspace_name, vswitch_type)
    vswitch_id = asc_client.create_v_switch(vpc_id, zone_id, cidr_block, vswitch_name)
    if vswitch_id is None:
        cli_logger.abort("Failed to create {} vswitch: {}.".format(vswitch_type, vswitch_name))
    else:
        cli_logger.print("Successfully created {} vswitch: {}.".format(vswitch_type, vswitch_name))
        return vswitch_id


def _delete_private_vswitches(workspace_name, vpc_id, subnet_cli):
    _delete_vswitches(workspace_name, vpc_id, subnet_cli, isPrivate=True)


def _delete_public_vswitches(workspace_name, vpc_id, subnet_cli):
    _delete_vswitches(workspace_name, vpc_id, subnet_cli, isPrivate=False)


def get_workspace_private_vswitches(workspace_name, vpc_id, asc_client):
    return _get_workspace_vswitches(workspace_name, vpc_id, asc_client, "cloudtik-{}-private-vswitch")


def get_workspace_public_vswitches(workspace_name, vpc_id, asc_client):
    return _get_workspace_vswitches(workspace_name, vpc_id, asc_client, "cloudtik-{}-public-vswitch")


def _get_workspace_vswitches(workspace_name, vpc_id, asc_client, name_pattern):
    vswitches = [vswitch for vswitch in asc_client.describe_v_switches(vpc_id)
               if vswitch.get("VSwitchName").startswith(name_pattern.format(workspace_name))]
    return vswitches


def _delete_vswitches(workspace_name, vpc_id, asc_client, isPrivate=True):
    vswitch_type = "private" if isPrivate else "public"
    """ Delete custom vswitches """
    vswitches =  get_workspace_private_vswitches(workspace_name, vpc_id, asc_client) \
        if isPrivate else get_workspace_public_vswitches(workspace_name, vpc_id, asc_client)

    if len(vswitches) == 0:
        cli_logger.print("No vswitches for workspace were found under this VPC: {}...".format(vpc_id))
        return

    for vswitch in vswitches:
        vswitch_id = vswitch.get("VSwitchId")
        cli_logger.print("Deleting {} vswitch: {}...".format(vswitch_type, vswitch_id))
        response = asc_client.delete_v_switch(vswitch_id)
        if response is None:
            cli_logger.abort("Failed to delete {} vswitch.".format(vswitch_type))
        else:
            cli_logger.print("Successfully deleted {} vswitch: {}.".format(vswitch_type, vswitch_id))


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


def _create_workspace_security_group(config, vpc_id, asc_client):
    security_group_name = ALIYUN_WORKSPACE_SECURITY_GROUP_NAME.format(config["workspace_name"])
    cli_logger.print("Creating security group for VPC: {} ...".format(vpc_id))
    security_group_id = asc_client.create_security_group(vpc_id, security_group_name)
    if security_group_id is None:
        cli_logger.abort("Failed to create security group for VPC: {}...".format(security_group_name))
    else:
        cli_logger.print("Successfully created security group: {}.".format(security_group_name))
        return security_group_id

    # security_group_id = cli.create_security_group(vpc_id=config["provider"]["vpc_id"])
    #
    # for rule in config["provider"].get("security_group_rule", {}):
    #     cli.authorize_security_group(
    #         security_group_id=security_group_id,
    #         port_range=rule["port_range"],
    #         source_cidr_ip=rule["source_cidr_ip"],
    #         ip_protocol=rule["ip_protocol"],
    #     )


def _add_security_group_rules(config, security_group_id, asc_client):
    cli_logger.print("Updating rules for security group: {}...".format(security_group_id))
    _update_inbound_rules(security_group_id, config, asc_client)
    cli_logger.print("Successfully updated rules for security group.")


def _update_inbound_rules(target_security_group_id, config, asc_client):
    extended_rules = config["provider"] \
        .get("security_group_rule", [])
    new_permissions = _create_default_inbound_rules(config, asc_client, extended_rules)
    security_group_attribute = asc_client.describe_security_group_attribute(target_security_group_id)
    old_permissions = security_group_attribute.get('Permissions').get('Permission')
    
    # revoke old permissions
    for old_permission in old_permissions:
        asc_client.revoke_security_group(
            ip_protocol=old_permission.get("IpProtocol"),
            port_range=old_permission.get("PortRange"),
            security_group_id=old_permission.get("SecurityGroupRuleId"),
            source_cidr_ip=old_permission.get("SourceCidrIp"))
    # revoke old permissions
    for new_permission in new_permissions:
        asc_client.authorize_security_group(
            ip_protocol=new_permission.get("IpProtocol"),
            port_range=new_permission.get("PortRange"),
            security_group_id=target_security_group_id,
            source_cidr_ip=new_permission.get("SourceCidrIp"))
    

def _create_default_inbound_rules(config, asc_client, extended_rules=None):
    if extended_rules is None:
        extended_rules = []
    intra_cluster_rules = _create_default_intra_cluster_inbound_rules(asc_client, config)
    # ssh_rules = _create_default_ssh_inbound_rules(asc_client, config)

    # TODO: support VPC peering
    # if is_use_peering_vpc(config) and is_peering_firewall_allow_working_subnet(config):
    #     extended_rules += _create_allow_working_node_inbound_rules(config)

    merged_rules = itertools.chain(
        intra_cluster_rules,
        extended_rules,
    )
    return list(merged_rules)


def get_workspace_security_group(config, asc_client, vpc_id):
    return _get_security_group(config, vpc_id, asc_client, get_workspace_security_group_name(config["workspace_name"]))


def _get_security_group(config, vpc_id, asc_client, group_name):
    security_group = _get_security_groups(config, asc_client, [vpc_id], [group_name])
    return None if not security_group else security_group[0]


def _get_security_groups(config, asc_client, vpc_ids, group_names):
    unique_vpc_ids = list(set(vpc_ids))
    unique_group_names = set(group_names)
    filtered_groups = []
    for vpc_id in unique_vpc_ids:
        security_groups = [security_group for security_group in asc_client.describe_security_groups(vpc_id=vpc_id)
                           if security_group.get("SecurityGroupName") in unique_group_names]
        filtered_groups.extend(security_groups)
    return filtered_groups


def get_workspace_security_group_name(workspace_name):
    return ALIYUN_WORKSPACE_SECURITY_GROUP_NAME.format(workspace_name)


def _update_security_group(config, asc_client):
    vpc_id = get_workspace_vpc_id(config, asc_client)
    security_group = get_workspace_security_group(config, asc_client, vpc_id)
    _add_security_group_rules(config, security_group.get("SecurityGroupId"), asc_client)
    return security_group


def _upsert_security_group(config, vpc_id, asc_client):
    current_step = 1
    total_steps = 2

    with cli_logger.group(
            "Creating security group for VPC",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        security_group_id = _create_workspace_security_group(config, vpc_id, asc_client)

    with cli_logger.group(
            "Configuring rules for security group",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _add_security_group_rules(config, security_group_id, asc_client)

    return security_group_id


def _delete_security_group(config, vpc_id, asc_client):
    """ Delete any security-groups """
    workspace_security_group = get_workspace_security_group(config, asc_client, vpc_id)
    if workspace_security_group is None:
        cli_logger.print("No security groups for workspace were found under this VPC: {}...".format(vpc_id))
        return
    security_group_id = workspace_security_group.get("SecurityGroupId")
    cli_logger.print("Deleting security group: {}...".format(security_group_id))
    response = asc_client.delete_security_group(security_group_id)
    if response is None:
        cli_logger.abort("Failed to delete security group.")
    else:
        cli_logger.print("Successfully deleted security group: {}.".format(security_group_id))


# def _create_default_intra_cluster_inbound_rules(intra_cluster_sgids):
#     return [{
#         "port_range": '-1/-1',
#         "ip_protocol": "all",
#         "UserIdGroupPairs": [
#             {
#                 "GroupId": security_group_id
#             } for security_group_id in sorted(intra_cluster_sgids)
#             # sort security group IDs for deterministic IpPermission models
#             # (mainly supports more precise stub-based boto3 unit testing)
#         ]
#     }]


def _create_default_intra_cluster_inbound_rules(asc_client, config):
    vpc = get_workspace_vpc(config, asc_client)
    vpc_cidr = vpc.get("CidrBlock")
    return [{
        "port_range": "-1/-1",
        "source_cidr_ip": vpc_cidr,
        "ip_protocol": "all"
    }]


def _delete_nat_gateway(config, asc_client):
    current_step = 1
    total_steps = 3

    with cli_logger.group(
            "Dissociating elastic ip",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _dissociate_elastic_ip(config, asc_client)

    with cli_logger.group(
            "Deleting NAT Gateway",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_nat_gateway_resource(config, asc_client)

    with cli_logger.group(
            "Releasing Elastic IP",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        release_elastic_ip(config, asc_client)


def _create_and_configure_nat_gateway(config, asc_client):
    current_step = 1
    total_steps = 3

    with cli_logger.group(
            "Creating Elastic IP",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_elastic_ip(config, asc_client)

    with cli_logger.group(
            "Creating NAT Gateway",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_nat_gateway(config, asc_client)

    with cli_logger.group(
            "Creating SNAT Entry",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_snat_entry(config, asc_client)


def get_workspace_snat_entry_name(workspace_name):
    return ALIYUN_WORKSPACE_SNAT_ENTRY_NAME.format(workspace_name)


def  _create_snat_entry(config, asc_client):
    workspace_name = config["workspace_name"]
    snat_entry_name = get_workspace_snat_entry_name(workspace_name)
    vpc_id = get_workspace_vpc_id(config, asc_client)
    private_vswitch = get_workspace_private_vswitches(workspace_name, vpc_id, asc_client)[0]
    nat_gateway = get_workspace_nat_gateway(config, asc_client)
    snat_table_id = nat_gateway.get("SnatTableIds").get("SnatTableId")[0]
    elastic_ip = get_workspace_elastic_ip(config, asc_client)
    snat_ip = elastic_ip.get("IpAddress")
    cli_logger.print("Creating SNAT Entry: {}...".format(snat_entry_name))
    response = asc_client.create_snat_entry(snat_table_id, private_vswitch["VSwitchId"], snat_ip, snat_entry_name)
    if response is None:
        cli_logger.abort("Failed to create SNAT Entry: {}.".format(snat_entry_name))
    else:
        cli_logger.print("Successfully created SNAT Entry: {}.".format(snat_entry_name))


def _delete_nat_gateway_resource(config, asc_client):
    """ Delete custom nat_gateway """
    nat_gateway = get_workspace_nat_gateway(config, asc_client)
    if nat_gateway is None:
        cli_logger.print("No Nat Gateway for workspace were found...")
        return
    nat_gateway_id = nat_gateway.get("NatGatewayId")
    nat_gateway_name = nat_gateway.get("Name")
    cli_logger.print("Deleting Nat Gateway: {}...".format(nat_gateway.get("Name")))
    response = asc_client.delete_nat_gateway(nat_gateway_id)
    if response is None:
        cli_logger.abort("Failed to delete Nat Gateway: {}.".format(nat_gateway_name))
    else:
        cli_logger.print("Successfully deleted Nat Gateway: {}.".format(nat_gateway_name))


def get_workspace_nat_gateway(config, asc_client):
    return _get_workspace_nat_gateway(config, asc_client)


def _get_workspace_nat_gateway(config, asc_client):
    workspace_name = config["workspace_name"]
    nat_gateway_name = get_workspace_nat_gateway_name(workspace_name)
    vpc_id = get_workspace_vpc_id(config, asc_client)
    cli_logger.verbose("Getting the Nat Gateway for workspace: {}...".format(nat_gateway_name))
    nat_gateways = asc_client.describe_nat_gateways(vpc_id)
    if nat_gateways is None:
        cli_logger.verbose("The Nat Gateway for workspace is not found: {}.".format(nat_gateway_name))
        return None

    nat_gateways = [nat_gateway for nat_gateway in nat_gateways if nat_gateway.get('Name') == nat_gateway_name]
    if len(nat_gateways) == 0:
        cli_logger.verbose("The Nat Gateway for workspace is not found: {}.".format(nat_gateway_name))
        return None
    else:
        cli_logger.verbose_error("Successfully get the Nat Gateway: {} for workspace.".format(nat_gateway_name))
        return nat_gateways[0]


def get_workspace_nat_gateway_name(workspace_name):
    return ALIYUN_WORKSPACE_NAT_GATEWAY_NAME.format(workspace_name)


def _create_nat_gateway(config, asc_client):
    workspace_name = config["workspace_name"]
    vpc_id =  get_workspace_vpc_id(config, asc_client)
    nat_gateway_name = get_workspace_nat_gateway_name(workspace_name)
    vswitch_id = get_workspace_public_vswitches(workspace_name, vpc_id, asc_client)
    cli_logger.print("Creating nat-gateway: {}...".format(nat_gateway_name))
    nat_gateway_id = asc_client.create_nat_gateway(vpc_id, nat_gateway_name, vswitch_id)
    if nat_gateway_id is None:
        cli_logger.abort("Failed to create nat-gateway.")
    else:
        cli_logger.print("Successfully created nat-gateway: {}.".format(nat_gateway_name))


def get_workspace_elastic_ip_name(workspace_name):
    return ALIYUN_WORKSPACE_EIP_NAME.format(workspace_name)


def _create_elastic_ip(config, asc_client):
    eip_name = get_workspace_elastic_ip_name(config["workspace_name"])
    allocation_id = asc_client.allocate_eip_address(eip_name)
    if allocation_id is None:
        cli_logger.abort("Faild to allocate Elastic IP.")
    else:
        cli_logger.print("Successfully to allocate Elastic IP.")
        return allocation_id


def get_workspace_elastic_ip(config, asc_client):
    return _get_workspace_elastic_ip(config, asc_client)


def _get_workspace_elastic_ip(config, asc_client):
    workspace_name = config["workspace_name"]
    elastic_ip_name = get_workspace_elastic_ip_name(workspace_name)
    cli_logger.verbose("Getting the Elastic IP for workspace: {}...".format(elastic_ip_name))
    eip_addresses = asc_client.describe_eip_addresses()
    if eip_addresses is None:
        cli_logger.verbose("The Elastic IP for workspace is not found: {}.".format(elastic_ip_name))
        return None

    eip_addresses = [eip_address for eip_address in eip_addresses if eip_address.get('Name') == elastic_ip_name]
    if len(eip_addresses) == 0:
        cli_logger.verbose("The Elastic IP for workspace is not found: {}.".format(elastic_ip_name))
        return None
    else:
        cli_logger.verbose_error("Successfully get the Elastic IP: {} for workspace.".format(elastic_ip_name))
        return eip_addresses[0]


def release_elastic_ip(config, asc_client):
    """ Delete Elastic IP """
    elastic_ip = get_workspace_elastic_ip(config, asc_client)
    if elastic_ip is None:
        cli_logger.print("No Elastic IP for workspace were found...")
        return
    allocation_id = elastic_ip.get("AllocationId")
    elastic_ip_name = elastic_ip.get("Name")
    cli_logger.print("Releasing Elastic IP: {}...".format(elastic_ip_name))
    response = asc_client.release_eip_address(allocation_id)
    if response is None:
        cli_logger.abort("Faild to release Elastic IP: {}.".format(elastic_ip_name))
    else:
        cli_logger.print("Successfully to release Elastic IP:{}.".format(elastic_ip_name))


def _dissociate_elastic_ip(config, asc_client):
    elastic_ip = get_workspace_elastic_ip(config, asc_client)
    if elastic_ip is None:
        return
    eip_allocation_id = elastic_ip.get("AllocationId")
    instance_id = elastic_ip.get("InstanceId")
    if instance_id is None:
        return
    elastic_ip_name = elastic_ip.get("Name")
    cli_logger.print("Dissociating Elastic IP: {}...".format(elastic_ip_name))
    response = asc_client.dissociate_eip_address(eip_allocation_id, "Nat", instance_id)
    if response is None:
        cli_logger.abort("Faild to dissociate Elastic IP: {}.".format(elastic_ip_name))
    else:
        cli_logger.print("Successfully to dissociate Elastic IP:{}.".format(elastic_ip_name))


def _delete_instance_profile(config, asc_client, instance_role_name):
    cli_logger.print("Deleting instance role: {}...".format(instance_role_name))
    role = asc_client.get_role(instance_role_name)
    if role is None:
        cli_logger.warning("No instance role was found.")
        return

    policies = asc_client.list_policy_for_role(instance_role_name)

    # Detach all policies from instance role
    for policy in policies:
        policy_type = policy.get("PolicyType")
        policy_name = policy.get("PolicyName")
        asc_client.detach_policy_from_role(instance_role_name, policy_type, policy_name)

    # Delete the specified instance role. The instance role must not have an associated policies.
    asc_client.delete_role(instance_role_name)

    cli_logger.print("Successfully deleted instance role.")


def _delete_instance_role_for_head(config, asc_client):
    head_instance_role_name = "cloudtik-{}-head-role".format(config["workspace_name"])
    _delete_instance_profile(config, asc_client, head_instance_role_name)


def _delete_instance_role_for_worker(config, asc_client):
    worker_instance_role_name = "cloudtik-{}-worker-role".format(config["workspace_name"])
    _delete_instance_profile(config, asc_client, worker_instance_role_name)


def _delete_workspace_instance_role(config, asc_client):
    current_step = 1
    total_steps = 2

    with cli_logger.group(
            "Deleting instance role for head",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_instance_role_for_head(config, asc_client)

    with cli_logger.group(
            "Deleting instance role for worker",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_instance_role_for_worker(config, asc_client)


def _create_or_update_instance_role(config, asc_client, instance_role_name, is_head=True):
    role = asc_client.get_role(instance_role_name)

    if role is None:
        cli_logger.verbose(
            "Creating new RAM instance role {} for use as the default.",
            cf.bold(instance_role_name))
        assume_role_policy_document = {
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
        }
        asc_client.create_role(instance_role_name, assume_role_policy_document)
        role = asc_client.get_role(instance_role_name)
        assert role is not None, "Failed to create role"

        attach_policies = HEAD_ROLE_ATTACH_POLICIES if is_head else WORKER_ROLE_ATTACH_POLICIES
        for policy in attach_policies:
            asc_client.attach_policy_to_role(instance_role_name, "System", policy)


def _create_instance_role_for_head(config, asc_client):
    head_instance_role_name = "cloudtik-{}-head-role".format(config["workspace_name"])
    cli_logger.print("Creating head instance role: {}...".format(head_instance_role_name))
    _create_or_update_instance_role(config, asc_client, head_instance_role_name)
    cli_logger.print("Successfully created and configured head instance role.")


def _create_instance_role_for_worker(config, asc_client):
    head_instance_role_name = "cloudtik-{}-head-role".format(config["workspace_name"])
    cli_logger.print("Creating head instance role: {}...".format(head_instance_role_name))
    _create_or_update_instance_role(config, asc_client, head_instance_role_name, is_head=False)
    cli_logger.print("Successfully created and configured head instance role.")


def _create_workspace_instance_role(config, asc_client):
    current_step = 1
    total_steps = 2

    with cli_logger.group(
            "Creating instance role for head",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_instance_role_for_head(config, asc_client)

    with cli_logger.group(
            "Creating instance role for worker",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_instance_role_for_worker(config, asc_client)


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
    workspace_name = config["workspace_name"]
    asc_client = _client(config)
    vpc_id = get_workspace_vpc_id(config, asc_client)
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
            _update_security_group(config, asc_client)

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
            "IpProtocol": "tcp",
            "PortRange": "22/22",
            "SourceCidrIp": allowed_ssh_source
        }
        security_group_rule_config.append(permission)


def _get_workspace_head_nodes(provider_config, workspace_name):
    pass


def list_aliyun_clusters(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    pass


def bootstrap_aliyun_workspace(config):
    # create a copy of the input config to modify
    config = copy.deepcopy(config)
    _configure_allowed_ssh_sources(config)
    return config


def check_aliyun_workspace_existence(config):
    pass


def get_aliyun_workspace_info(config):
    pass