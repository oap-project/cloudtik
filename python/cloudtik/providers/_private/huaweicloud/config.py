import copy
import json
import logging
import os
import time
from functools import partial
from typing import Any, Dict, Optional

import requests
from huaweicloudsdkcore.exceptions.exceptions import ServiceResponseException
from huaweicloudsdkecs.v2 import ListFlavorsRequest, ListServersDetailsRequest, \
    NovaCreateKeypairOption, \
    NovaCreateKeypairRequest, \
    NovaCreateKeypairRequestBody, \
    NovaShowKeypairRequest
from huaweicloudsdkeip.v2 import CreatePublicipBandwidthOption, \
    CreatePublicipOption, CreatePublicipRequest, \
    CreatePublicipRequestBody, DeletePublicipRequest, ListPublicipsRequest
from huaweicloudsdkiam.v3 import \
    AssociateAgencyWithAllProjectsPermissionRequest, CreateAgencyOption, \
    CreateAgencyRequest, \
    CreateAgencyRequestBody, DeleteAgencyRequest, \
    KeystoneListPermissionsRequest, KeystoneListProjectsRequest, \
    ListAgenciesRequest
from huaweicloudsdknat.v2 import CreateNatGatewayOption, \
    CreateNatGatewayRequest, \
    CreateNatGatewayRequestBody, CreateNatGatewaySnatRuleOption, \
    CreateNatGatewaySnatRuleRequest, \
    CreateNatGatewaySnatRuleRequestOption, DeleteNatGatewayRequest, \
    DeleteNatGatewaySnatRuleRequest, \
    ListNatGatewaySnatRulesRequest, \
    ListNatGatewaysRequest
from huaweicloudsdkvpc.v2 import AcceptVpcPeeringRequest, \
    CreateSecurityGroupOption, \
    CreateSecurityGroupRequest, \
    CreateSecurityGroupRequestBody, CreateSecurityGroupRuleOption, \
    CreateSecurityGroupRuleRequest, \
    CreateSecurityGroupRuleRequestBody, CreateSubnetOption, \
    CreateSubnetRequest, \
    CreateSubnetRequestBody, \
    CreateVpcOption, \
    CreateVpcPeeringOption, CreateVpcPeeringRequest, \
    CreateVpcPeeringRequestBody, CreateVpcRequest, \
    CreateVpcRequestBody, DeleteSecurityGroupRequest, \
    DeleteSecurityGroupRuleRequest, DeleteSubnetRequest, \
    DeleteVpcPeeringRequest, \
    DeleteVpcRequest, ListRouteTablesRequest, \
    ListSecurityGroupRulesRequest, ListSecurityGroupsRequest, \
    ListSubnetsRequest, \
    ListVpcPeeringsRequest, ListVpcsRequest, RouteTableRoute, ShowVpcRequest, \
    UpdateRouteTableReq, UpdateRoutetableReqBody, UpdateRouteTableRequest, \
    VpcInfo
from obs import CreateBucketHeader

from cloudtik.core._private.cli_logger import cf, cli_logger
from cloudtik.core._private.utils import check_cidr_conflict, \
    is_managed_cloud_storage, \
    is_peering_firewall_allow_ssh_only, \
    is_peering_firewall_allow_working_subnet, is_use_internal_ip, \
    is_use_managed_cloud_storage, \
    is_use_peering_vpc, \
    is_use_working_vpc, is_worker_role_for_cloud_storage
from cloudtik.core.tags import CLOUDTIK_TAG_CLUSTER_NAME, \
    CLOUDTIK_TAG_NODE_KIND, NODE_KIND_HEAD
from cloudtik.core.workspace_provider import \
    CLOUDTIK_MANAGED_CLOUD_STORAGE, CLOUDTIK_MANAGED_CLOUD_STORAGE_URI, \
    Existence
from cloudtik.providers._private.huaweicloud.utils import _get_node_info, \
    _make_ecs_client, _make_obs_client, \
    _make_vpc_client, export_huaweicloud_obs_storage_config, flat_tags_map, \
    get_huaweicloud_obs_storage_config, \
    get_huaweicloud_obs_storage_config_for_update, HWC_OBS_BUCKET, \
    HWC_SERVER_STATUS_ACTIVE, make_ecs_client, \
    make_eip_client, \
    make_iam_client, \
    make_nat_client, \
    make_obs_client, make_obs_client_aksk, make_vpc_client, tags_list_to_dict
from cloudtik.providers._private.utils import StorageTestingError

logger = logging.getLogger(__name__)

HWC_WORKSPACE_NUM_CREATION_STEPS = 5
HWC_WORKSPACE_NUM_DELETION_STEPS = 5
HWC_WORKSPACE_TARGET_RESOURCES = 7
HWC_RESOURCE_NAME_PREFIX = 'cloudtik-{}'
HWC_WORKSPACE_VPC_NAME = HWC_RESOURCE_NAME_PREFIX + '-vpc'
HWC_WORKSPACE_DEFAULT_CIDR_PREFIX = '192.168.'
HWC_WORKSPACE_VPC_DEFAULT_CIDR = HWC_WORKSPACE_DEFAULT_CIDR_PREFIX + '0.0/16'
HWC_WORKSPACE_SUBNET_NAME = HWC_RESOURCE_NAME_PREFIX + '-{}-subnet'
HWC_WORKSPACE_NAT_NAME = HWC_RESOURCE_NAME_PREFIX + '-nat'
HWC_WORKSPACE_SG_NAME = HWC_RESOURCE_NAME_PREFIX + '-sg'
HWC_WORKSPACE_VPC_PEERING_NAME = HWC_RESOURCE_NAME_PREFIX + '-peering'
HWC_WORKSPACE_EIP_BANDWIDTH_NAME = HWC_RESOURCE_NAME_PREFIX + '-bandwidth'
HWC_WORKSPACE_EIP_NAME = HWC_RESOURCE_NAME_PREFIX + '-eip'
HWC_WORKSPACE_OBS_BUCKET_NAME = HWC_RESOURCE_NAME_PREFIX + '-obs-bucket'
HWC_WORKSPACE_INSTANCE_PROFILE_NAME = HWC_RESOURCE_NAME_PREFIX + '-{}-profile'
HWC_WORKSPACE_VPC_SUBNETS_COUNT = 2
HWC_VM_METADATA_URL = 'http://169.254.169.254/openstack/latest/meta_data.json'
HWC_MANAGED_CLOUD_STORAGE_OBS_BUCKET = "huaweicloud.managed.cloud.storage." \
                                       "obs.bucket"
HWC_KEY_PAIR_NAME = 'cloudtik_huaweicloud_keypair_{}'
HWC_KEY_PATH_TEMPLATE = '~/.ssh/{}.pem'


def create_huaweicloud_workspace(config):
    workspace_name = config['workspace_name']
    managed_cloud_storage = is_managed_cloud_storage(config)
    use_peering_vpc = is_use_peering_vpc(config)

    current_step = 1
    total_steps = HWC_WORKSPACE_NUM_CREATION_STEPS
    if managed_cloud_storage:
        total_steps += 1
    if use_peering_vpc:
        total_steps += 1

    try:
        with cli_logger.group("Creating workspace: {}",
                              cf.bold(workspace_name)):
            # Step1: create vpc
            with cli_logger.group("Creating VPC",
                                  _numbered=("[]", current_step, total_steps)):
                current_step += 1
                vpc_client = make_vpc_client(config)
                use_working_vpc = is_use_working_vpc(config)
                if use_working_vpc:
                    vpc = _get_current_vpc(config)
                    cli_logger.print("Use working workspace VPC: {}...",
                                     vpc.name)
                else:
                    vpc = _check_and_create_vpc(vpc_client, workspace_name)

            # Step2: create subnet
            with cli_logger.group("Creating subnets",
                                  _numbered=("[]", current_step, total_steps)):
                current_step += 1
                subnets = _check_and_create_subnets(vpc, vpc_client,
                                                    workspace_name)
            # Step3: create NAT and SNAT rules
            with cli_logger.group("Creating NAT gateway for subnets",
                                  _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _check_and_create_nat_gateway_and_eip(
                    config, subnets, vpc, workspace_name)
            # Step4: create security group
            with cli_logger.group("Creating security group",
                                  _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _check_and_create_security_group(config, vpc, vpc_client,
                                                 workspace_name)
            if use_peering_vpc:
                # Step5: create peering VPC
                with cli_logger.group("Creating VPC peering connection",
                                      _numbered=("[]", current_step,
                                                 total_steps)):
                    current_step += 1
                    _create_and_accept_vpc_peering(config, vpc, vpc_client,
                                                   workspace_name)

            # Step6: create instance profile
            with cli_logger.group("Creating instance profile",
                                  _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _create_workspace_instance_profile(config, workspace_name)

            if managed_cloud_storage:
                # Step7: create OBS(Object Storage Service) bucket
                with cli_logger.group("Creating OBS bucket",
                                      _numbered=("[]", current_step,
                                                 total_steps)):
                    current_step += 1
                    obs_client = make_obs_client(config)
                    _check_and_create_cloud_storage_bucket(obs_client,
                                                           workspace_name)
    except Exception as e:
        cli_logger.error("Failed to create workspace with the name {} at step"
                         "{}. \n{}", workspace_name, current_step - 1, str(e))
        raise e

    cli_logger.print("Successfully created workspace: {}.",
                     cf.bold(workspace_name))


def _create_workspace_instance_profile(config, workspace_name):
    current_step = 1
    total_steps = 2

    with cli_logger.group("Creating instance profile for head",
                          _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_instance_profile(config, workspace_name, for_head=True)

    with cli_logger.group("Creating instance profile for worker",
                          _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_instance_profile(config, workspace_name, for_head=False)


def _create_instance_profile(config, workspace_name, for_head):
    iam_client = make_iam_client(config)
    _prefix = 'head' if for_head else 'worker'
    profile_name = HWC_WORKSPACE_INSTANCE_PROFILE_NAME.format(workspace_name,
                                                              _prefix)
    cli_logger.print(
        "Creating instance profile: {}...", profile_name)
    domain_id = _get_current_domain_id(iam_client)
    # Instance profile is named as IAM agency in Huawei Cloud
    target_agency = iam_client.create_agency(
        CreateAgencyRequest(CreateAgencyRequestBody(
            CreateAgencyOption(name=profile_name, domain_id=domain_id,
                               duration='FOREVER',
                               trust_domain_name='op_svc_ecs')))
    ).agency

    _associate_instance_profile_with_permission(for_head, iam_client,
                                                target_agency)
    cli_logger.print(
        "Successfully created instance profile: {}.", profile_name)


def _associate_instance_profile_with_permission(for_head, iam_client,
                                                target_agency):
    ECS_access_role = "ECS FullAccess"
    ECS_role_id = _get_cloud_services_access_role_id(iam_client,
                                                     ECS_access_role)
    OBS_access_role = "OBS OperateAccess"
    OBS_role_id = _get_cloud_services_access_role_id(iam_client,
                                                     OBS_access_role)
    if for_head:
        iam_client.associate_agency_with_all_projects_permission(
            AssociateAgencyWithAllProjectsPermissionRequest(
                agency_id=target_agency.id,
                role_id=ECS_role_id))
    iam_client.associate_agency_with_all_projects_permission(
        AssociateAgencyWithAllProjectsPermissionRequest(
            agency_id=target_agency.id,
            role_id=OBS_role_id))


def _delete_workspace_instance_profile(config, workspace_name):
    current_step = 1
    total_steps = 2

    with cli_logger.group("Deleting instance profile for head",
                          _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_instance_profile(config, workspace_name, for_head=True)

    with cli_logger.group("Deleting instance profile for worker",
                          _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_instance_profile(config, workspace_name, for_head=False)


def _delete_instance_profile(config, workspace_name, for_head):
    iam_client = make_iam_client(config)
    _prefix = 'head' if for_head else 'worker'
    profile_name = HWC_WORKSPACE_INSTANCE_PROFILE_NAME.format(workspace_name,
                                                              _prefix)
    domain_id = _get_current_domain_id(iam_client)
    target_agencies = iam_client.list_agencies(
        ListAgenciesRequest(name=profile_name, domain_id=domain_id)
    ).agencies

    if not target_agencies:
        cli_logger.print("No instance profile found for {}. Skip deletion.".format(_prefix))
        return

    for _agency in target_agencies:
        cli_logger.print(
            "Deleting instance profile: {}...", _agency.name)
        iam_client.delete_agency(DeleteAgencyRequest(agency_id=_agency.id))
        cli_logger.print(
            "Successfully deleted instance profile: {}.", _agency.name)


def _get_current_domain_id(iam_client):
    domain_id = iam_client.keystone_list_projects(
        KeystoneListProjectsRequest(enabled=True, page=1,
                                    per_page=1)).projects[0].domain_id
    return domain_id


def _get_cloud_services_access_role_id(iam_client, role_name):
    target_role = iam_client.keystone_list_permissions(
        KeystoneListPermissionsRequest(display_name=role_name)
    ).roles[0]
    return target_role.id


def _check_and_create_cloud_storage_bucket(obs_client, workspace_name):
    # If the managed cloud storage for the workspace already exists
    # Skip the creation step
    bucket_name = _get_managed_obs_bucket(obs_client, workspace_name)
    if bucket_name:
        cli_logger.print("OBS bucket for the workspace already exists. "
                         "Skip creation.")
        return
    else:
        bucket_name = HWC_WORKSPACE_OBS_BUCKET_NAME.format(workspace_name)
    # Create new bucket with parallel file system enable
    resp = obs_client.createBucket(bucket_name,
                                   header=CreateBucketHeader(isPFS=True),
                                   location=obs_client.region)
    if resp.status < 300:
        cli_logger.print(
            "Successfully created OBS bucket: {}.".format(bucket_name))
    else:
        cli_logger.abort("Failed to create OBS bucket. {}", str(resp))


def _get_managed_obs_bucket(obs_client, workspace_name=None, bucket_name=None):
    if not bucket_name:
        bucket_name = HWC_WORKSPACE_OBS_BUCKET_NAME.format(workspace_name)
    resp = obs_client.headBucket(bucket_name)
    if resp.status < 300:
        return bucket_name
    elif resp.status == 404:
        return None
    else:
        raise Exception("HUAWEI CLOUD OBS service error {}".format(resp))


def _create_and_accept_vpc_peering(config, _workspace_vpc, vpc_client,
                                   workspace_name):
    current_step = 1
    total_steps = 3
    with cli_logger.group("Creating VPC peering connection",
                          _numbered=("()", current_step, total_steps)):
        current_step += 1
        _current_vpc = _get_current_vpc(config)
        _vpc_peering_name = HWC_WORKSPACE_VPC_PEERING_NAME.format(
            workspace_name)
        vpc_peering = vpc_client.create_vpc_peering(
            CreateVpcPeeringRequest(
                CreateVpcPeeringRequestBody(
                    CreateVpcPeeringOption(name=_vpc_peering_name,
                                           request_vpc_info=VpcInfo(
                                               _current_vpc.id),
                                           accept_vpc_info=VpcInfo(
                                               _workspace_vpc.id))))
        ).peering

    with cli_logger.group("Accepting VPC peering connection",
                          _numbered=("()", current_step, total_steps)):
        current_step += 1
        # If VPC peering is built between different tenants, need to accept.
        if _current_vpc.tenant_id != _workspace_vpc.tenant_id:
            vpc_peering_status = vpc_client.accept_vpc_peering(
                AcceptVpcPeeringRequest(vpc_peering.id)).status
            cli_logger.print(
                "VPC peering {} status is {}".format(vpc_peering.id,
                                                     vpc_peering_status))

    with cli_logger.group("Updating route table for peering connection",
                          _numbered=("()", current_step, total_steps)):
        current_step += 1
        _current_vpc_rts = vpc_client.list_route_tables(
            ListRouteTablesRequest(vpc_id=_current_vpc.id)
        ).routetables
        _workspace_vpc_rts = vpc_client.list_route_tables(
            ListRouteTablesRequest(vpc_id=_workspace_vpc.id)
        ).routetables
        for _current_vpc_rt in _current_vpc_rts:
            vpc_client.update_route_table(
                UpdateRouteTableRequest(
                    routetable_id=_current_vpc_rt.id,
                    body=UpdateRoutetableReqBody(
                        UpdateRouteTableReq(routes={
                            'add': [RouteTableRoute(
                                type='peering',
                                destination=_workspace_vpc.cidr,
                                nexthop=vpc_peering.id)]}
                        )
                    )
                )
            )
            cli_logger.print("Successfully add route destination to current "
                             "VPC route table {} with workspace VPC CIDR "
                             "block.".format(_current_vpc_rt.id))

        for _workspace_vpc_rt in _workspace_vpc_rts:
            vpc_client.update_route_table(
                UpdateRouteTableRequest(
                    routetable_id=_workspace_vpc_rt.id,
                    body=UpdateRoutetableReqBody(
                        UpdateRouteTableReq(routes={
                            'add': [RouteTableRoute(
                                type='peering',
                                destination=_current_vpc.cidr,
                                nexthop=vpc_peering.id)]}
                        )
                    )
                )
            )
            cli_logger.print("Successfully add route destination to "
                             "workspace VPC route table {} with current "
                             "VPC CIDR block.".format(_workspace_vpc_rt.id))


def _check_and_create_security_group(config, vpc, vpc_client, workspace_name):
    # Create security group
    sg_name = HWC_WORKSPACE_SG_NAME.format(workspace_name)

    cli_logger.print(
        "Creating and updating security group: {}...", sg_name)
    sg = vpc_client.create_security_group(
        CreateSecurityGroupRequest(
            CreateSecurityGroupRequestBody(
                CreateSecurityGroupOption(name=sg_name)))
    ).security_group

    # Create sg rules in config
    _update_security_group_rules(config, sg, vpc, vpc_client)

    cli_logger.print(
        "Successfully created and updated security group: {}.", sg_name)


def _update_security_group_rules(config, sg, vpc, vpc_client):
    # Clean old rule if exist
    _clean_security_group_rules(sg, vpc_client)
    # Add new rules
    extended_rules = config["provider"].get("security_group", {}) \
        .get("IpPermissions", [])
    for _ext_rule in extended_rules:
        vpc_client.create_security_group_rule(
            CreateSecurityGroupRuleRequest(
                CreateSecurityGroupRuleRequestBody(
                    CreateSecurityGroupRuleOption(sg.id,
                                                  direction='ingress',
                                                  remote_ip_prefix=_ext_rule)))
        )
    # Create SSH rule
    vpc_client.create_security_group_rule(
        CreateSecurityGroupRuleRequest(
            CreateSecurityGroupRuleRequestBody(
                CreateSecurityGroupRuleOption(sg.id,
                                              direction='ingress',
                                              port_range_min=22,
                                              port_range_max=22,
                                              protocol='tcp',
                                              remote_ip_prefix=vpc.cidr)))
    )
    # Create peering vpc rule
    if is_use_peering_vpc(config) and \
            is_peering_firewall_allow_working_subnet(config):
        allow_ssh_only = is_peering_firewall_allow_ssh_only(config)
        _current_vpc = _get_current_vpc(config)
        _vpc_cidr = _current_vpc.cidr
        _port_min = 22 if allow_ssh_only else 1
        _port_max = 22 if allow_ssh_only else 65535
        vpc_client.create_security_group_rule(
            CreateSecurityGroupRuleRequest(
                CreateSecurityGroupRuleRequestBody(
                    CreateSecurityGroupRuleOption(sg.id,
                                                  direction='ingress',
                                                  port_range_min=_port_min,
                                                  port_range_max=_port_max,
                                                  protocol='tcp',
                                                  remote_ip_prefix=_vpc_cidr)))
        )


def _clean_security_group_rules(sg, vpc_client):
    rule_list = vpc_client.list_security_group_rules(
        ListSecurityGroupRulesRequest(security_group_id=sg.id)
    ).security_group_rules
    for _rule in rule_list:
        vpc_client.delete_security_group_rule(
            DeleteSecurityGroupRuleRequest(_rule.id))


def _check_and_create_nat_gateway_and_eip(config, subnets, vpc, workspace_name):
    current_step = 1
    total_steps = 3
    with cli_logger.group("Creating NAT gateway",
                          _numbered=("()", current_step, total_steps)):
        current_step += 1
        nat_gateway = _check_and_create_nat_gateway(
            config, subnets, vpc, workspace_name)

    with cli_logger.group("Creating NAT EIP",
                          _numbered=("()", current_step, total_steps)):
        current_step += 1
        eip = _check_and_create_eip(config, workspace_name)

    with cli_logger.group("Creating SNAT Rules",
                          _numbered=("()", current_step, total_steps)):
        current_step += 1
        _check_and_create_snat_rules(
            config, subnets, nat_gateway, eip
        )


def _check_and_create_nat_gateway(config, subnets, vpc, workspace_name):
    nat_client = make_nat_client(config)
    nat_name = HWC_WORKSPACE_NAT_NAME.format(workspace_name)
    pub_net = subnets[0].id

    cli_logger.print("Creating NAT gateway: {}...", nat_name)
    nat_gateway = nat_client.create_nat_gateway(
        CreateNatGatewayRequest(
            CreateNatGatewayRequestBody(
                CreateNatGatewayOption(name=nat_name,
                                       router_id=vpc.id,
                                       internal_network_id=pub_net,
                                       spec='1')))
    ).nat_gateway
    cli_logger.print("Successfully created NAT gateway: {}.", nat_name)
    return nat_gateway


def _check_and_create_eip(config, workspace_name):
    # Create EIP
    _bw_name = HWC_WORKSPACE_EIP_BANDWIDTH_NAME.format(workspace_name)
    _eip_name = HWC_WORKSPACE_EIP_NAME.format(workspace_name)
    eip_client = make_eip_client(config) \

    cli_logger.print("Creating elastic IP: {}...", _eip_name)
    eip = eip_client.create_publicip(
        CreatePublicipRequest(
            CreatePublicipRequestBody(
                # Dedicated bandwidth 5 Mbit
                bandwidth=CreatePublicipBandwidthOption(name=_bw_name,
                                                        share_type='PER',
                                                        size=5),
                publicip=CreatePublicipOption(type='5_bgp',
                                              alias=_eip_name)))
    ).publicip
    cli_logger.print("Successfully created elastic IP: {}.",
                     eip.public_ip_address)
    return eip


def _check_and_create_snat_rules(config, subnets, nat_gateway, eip):
    nat_client = make_nat_client(config)

    cli_logger.print(
        "Creating SNAT rules for NAT gateway: {}...", nat_gateway.name)
    # Check workspace NAT gateway util active, then to add rules
    _wait_util_nat_gateway_active(nat_client, nat_gateway)
    # Create SNAT rule for public and private subnets
    for _subnet in subnets:
        nat_client.create_nat_gateway_snat_rule(
            CreateNatGatewaySnatRuleRequest(
                CreateNatGatewaySnatRuleRequestOption(
                    CreateNatGatewaySnatRuleOption(
                        nat_gateway_id=nat_gateway.id,
                        network_id=_subnet.id,
                        floating_ip_id=eip.id)))
        )
    cli_logger.print(
        "Successfully created SNAT rules for NAT gateway: {}.", nat_gateway.name)


def _wait_util_nat_gateway_active(nat_client, nat_gateway):
    _retry = 0
    while True:
        nat_gateway = _get_workspace_nat(nat_client, nat_id=nat_gateway.id)[0]
        if nat_gateway.status == 'ACTIVE' or _retry > 5:
            break
        else:
            _retry += 1
            time.sleep(1)


def _get_available_subnet_cidr(vpc, vpc_client, workspace_name):
    cidr_list = []
    current_vpc_subnets = vpc_client.list_subnets(
        ListSubnetsRequest(vpc_id=vpc.id)).subnets
    current_subnet_cidr = [_subnet.cidr for _subnet in
                           current_vpc_subnets]
    vpc_cidr_block = vpc.cidr
    ip_range = vpc_cidr_block.split('/')[0].split('.')
    for i in range(0, 256):
        tmp_cidr_block = '{}.{}.{}.0/24'.format(ip_range[0],
                                                ip_range[1],
                                                i)
        if check_cidr_conflict(tmp_cidr_block,
                               current_subnet_cidr):
            cidr_list.append(tmp_cidr_block)
        if len(cidr_list) >= HWC_WORKSPACE_VPC_SUBNETS_COUNT:
            break
    if len(cidr_list) < HWC_WORKSPACE_VPC_SUBNETS_COUNT:
        raise RuntimeError(
            "No enough available subnets in VPC {} "
            "for workspace {}".format(vpc.name, workspace_name)
        )
    return cidr_list


def _check_and_create_vpc(vpc_client, workspace_name):
    # Check vpc name
    vpc_name = HWC_WORKSPACE_VPC_NAME.format(workspace_name)
    response = vpc_client.list_vpcs(ListVpcsRequest())
    for _vpc in response.vpcs:
        if _vpc.name == vpc_name:
            raise RuntimeError("There is a same name VPC for workspace: {}, "
                               "if you want to create a new workspace with "
                               "the same name, you need to execute workspace "
                               "delete first!".format(workspace_name))
    # Create new vpc
    cli_logger.print("Creating workspace VPC: {}...",
                     vpc_name)
    default_cidr = HWC_WORKSPACE_VPC_DEFAULT_CIDR
    request = CreateVpcRequest(
        CreateVpcRequestBody(
            vpc=CreateVpcOption(name=vpc_name,
                                cidr=default_cidr)))
    vpc = vpc_client.create_vpc(request).vpc
    cli_logger.print("Successfully created workspace VPC: {}.",
                     vpc.name)
    return vpc


def _check_and_create_subnets(vpc, vpc_client, workspace_name):
    subnets = []
    subnet_cidr_list = _get_available_subnet_cidr(vpc, vpc_client,
                                                  workspace_name)
    for i, _cidr in enumerate(subnet_cidr_list, start=1):
        subnet_type = 'public' if i == 1 else 'private'
        subnet_name = HWC_WORKSPACE_SUBNET_NAME.format(
            workspace_name, subnet_type)
        _gateway_ip = _cidr.replace('.0/24', '.1')
        with cli_logger.group("Creating {} subnet", subnet_type,
                              _numbered=("()", i,
                                         len(subnet_cidr_list))):
            try:
                cli_logger.print("Creating subnet: {}...", subnet_name)
                _subnet = vpc_client.create_subnet(
                    CreateSubnetRequest(
                        CreateSubnetRequestBody(
                            CreateSubnetOption(name=subnet_name,
                                               cidr=_cidr,
                                               gateway_ip=_gateway_ip,
                                               vpc_id=vpc.id)))
                ).subnet
                cli_logger.print("Successfully created subnet: {}.", subnet_name)
            except Exception as e:
                cli_logger.error("Failed to create {} subnet. {}",
                                 subnet_type, str(e))
                raise e
            subnets.append(_subnet)
    return subnets


def get_workspace_vpc(vpc_client, workspace_name):
    _vpcs = vpc_client.list_vpcs(ListVpcsRequest()).vpcs
    _workspace_vpc_name = HWC_WORKSPACE_VPC_NAME.format(workspace_name)
    for _vpc in _vpcs:
        if _workspace_vpc_name == _vpc.name:
            workspace_vpc = _vpc
            break
    else:
        workspace_vpc = None
    return workspace_vpc


def _get_current_vpc(config):
    try:
        response = requests.get(HWC_VM_METADATA_URL)
        metadata_json = json.loads(response.text)
        vpc_id = metadata_json["meta"]["vpc_id"]
        region_id = metadata_json["region_id"]
        # current region maybe different with workspace region when VPC peering
        vpc_client = make_vpc_client(config, region=region_id)
        vpc = vpc_client.show_vpc(ShowVpcRequest(vpc_id=vpc_id)).vpc
        return vpc
    except Exception as e:
        raise RuntimeError("Failed to get the VPC for the current machine. "
                           "Please make sure your current machine is"
                           "a HUAWEICLOUD virtual machine. {}".format(e))


def delete_huaweicloud_workspace(config, delete_managed_storage):
    workspace_name = config['workspace_name']
    managed_cloud_storage = is_managed_cloud_storage(config)
    use_peering_vpc = is_use_peering_vpc(config)

    current_step = 1
    total_steps = HWC_WORKSPACE_NUM_DELETION_STEPS
    if use_peering_vpc:
        total_steps += 1
    if managed_cloud_storage and delete_managed_storage:
        total_steps += 1

    try:
        with cli_logger.group("Deleting workspace: {}",
                              cf.bold(workspace_name)):
            vpc_client = make_vpc_client(config)
            if use_peering_vpc:
                # Step1: delete peering vpc connection
                with cli_logger.group("Deleting peering VPC connection",
                                      _numbered=("[]",
                                                 current_step, total_steps)):
                    current_step += 1
                    _check_and_delete_vpc_peering_connection(config,
                                                             vpc_client,
                                                             workspace_name)

            # Step2: delete security group
            with cli_logger.group("Deleting security group",
                                  _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _check_and_delete_security_group(vpc_client, workspace_name)

            # Step3: delete NAT and SNAT rules
            with cli_logger.group("Deleting NAT, SNAT rules and EIP",
                                  _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _check_and_delete_nat_gateway_and_eip(config, workspace_name)

            # Step4: delete subnets
            with cli_logger.group("Deleting private and public subnets",
                                  _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _workspace_vpc = _check_and_delete_subnets(vpc_client,
                                                           workspace_name)
            # Step5: delete VPC
            with cli_logger.group("Deleting VPC",
                                  _numbered=("[]", current_step, total_steps)):
                current_step += 1
                use_working_vpc = is_use_working_vpc(config)
                _check_and_delete_vpc(config, use_working_vpc, vpc_client,
                                      _workspace_vpc)

            # Step6: delete instance profile
            with cli_logger.group("Deleting instance profile",
                                  _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _delete_workspace_instance_profile(config, workspace_name)

            if managed_cloud_storage and delete_managed_storage:
                # Step7: delete OBS(Object Storage Service) bucket
                with cli_logger.group("Deleting OBS bucket and objects",
                                      _numbered=("[]",
                                                 current_step, total_steps)):
                    current_step += 1
                    obs_client = make_obs_client(config)
                    _check_and_delete_cloud_storage_bucket(obs_client,
                                                           workspace_name)
    except Exception as e:
        cli_logger.error("Failed to delete workspace with the name {} at step"
                         "{}. \n{}", workspace_name, current_step - 1, str(e))
        raise e

    cli_logger.print("Successfully deleted workspace: {}.",
                     cf.bold(workspace_name))


def _check_and_delete_cloud_storage_bucket(obs_client, workspace_name):
    bucket_name = _get_managed_obs_bucket(obs_client,
                                          workspace_name)
    if not bucket_name:
        cli_logger.warning("No OBS bucket with the name found.")
        return
    # List and delete all objects in bucket
    _check_and_delete_bucket_objects(obs_client, bucket_name)
    # Delete bucket
    resp = obs_client.deleteBucket(bucket_name)
    if resp.status < 300:
        cli_logger.print(
            "Successfully deleted OBS bucket: {}.".format(bucket_name))
    else:
        cli_logger.abort("Failed to delete OBS bucket. {}".format(bucket_name))


def _check_and_delete_bucket_objects(obs_client, bucket_name):
    while True:
        resp = obs_client.listObjects(bucket_name)
        # quick pass if no bucket objects
        if not resp.body.contents:
            break
        for _object in resp.body.contents:
            obs_client.deleteObject(bucket_name, _object.key)
        time.sleep(1)


def _check_and_delete_vpc(config, use_working_vpc, vpc_client, _workspace_vpc):
    if not use_working_vpc:
        if _workspace_vpc:
            cli_logger.print(
                "Deleting VPC: {}...", _workspace_vpc.name)
            vpc_client.delete_vpc(DeleteVpcRequest(vpc_id=_workspace_vpc.id))
            cli_logger.print(
                "Successfully deleted VPC: {}.", _workspace_vpc.name)
        else:
            cli_logger.print("Can't find workspace VPC")
    else:
        _current_vpc = _get_current_vpc(config)
        cli_logger.print("Skip to delete working VPC {}", _current_vpc.name)


def _check_and_delete_subnets(vpc_client, workspace_name):
    _workspace_vpc = get_workspace_vpc(vpc_client, workspace_name)
    if _workspace_vpc:
        _delete_subnets_until_empty(vpc_client, _workspace_vpc, workspace_name)
    else:
        cli_logger.print("Can't find workspace VPC")
    return _workspace_vpc


def _delete_subnets_until_empty(vpc_client, workspace_vpc, workspace_name):
    cli_logger.print(
        "Deleting subnets for VPC: {}...", workspace_vpc.name)
    while True:
        subnets = _get_workspace_vpc_subnets(vpc_client, workspace_vpc,
                                             workspace_name)
        # quick pass if no subnets in VPC
        if not subnets:
            break
        # Delete target subnets
        for _subnet in subnets:
            try:
                vpc_client.delete_subnet(
                    DeleteSubnetRequest(vpc_id=workspace_vpc.id,
                                        subnet_id=_subnet.id))
            except ServiceResponseException:
                # if delete a subnet in deleting workflow,
                # maybe raise exception, try to delete in next time.
                break
        time.sleep(1)
    cli_logger.print(
        "Successfully deleted subnets for VPC: {}.", workspace_vpc.id)


def _get_workspace_vpc_subnets(vpc_client, _workspace_vpc, workspace_name,
                               category=None):
    subnets = vpc_client.list_subnets(
        ListSubnetsRequest(vpc_id=_workspace_vpc.id)
    ).subnets
    if category:
        _prefix = HWC_WORKSPACE_SUBNET_NAME.format(workspace_name, category)
        _suffix = ''
    # get all public and private subnets
    else:
        _prefix = HWC_RESOURCE_NAME_PREFIX.format(workspace_name)
        _suffix = '-subnet'

    target_subnets = []
    for _subnet in subnets:
        if _subnet.name.startswith(_prefix) and \
                _subnet.name.endswith(_suffix):
            target_subnets.append(_subnet)
    return target_subnets


def _check_and_delete_security_group(vpc_client, workspace_name):
    workspace_vpc = get_workspace_vpc(vpc_client, workspace_name)
    if workspace_vpc:
        target_sgs = _get_workspace_security_group(vpc_client, workspace_name)
        if len(target_sgs) == 0:
            cli_logger.print(
                "No security groups for workspace were found under this VPC: {}. Skip deletion.",
                workspace_vpc.name)
            return

        for _sg in target_sgs:
            cli_logger.print(
                "Deleting security group: {}...", _sg.name)
            vpc_client.delete_security_group(
                DeleteSecurityGroupRequest(_sg.id))
            cli_logger.print(
                "Successfully deleted security group: {}.", _sg.name)


def _get_workspace_security_group(vpc_client, workspace_name):
    _sgs = vpc_client.list_security_groups(
        ListSecurityGroupsRequest()).security_groups
    _sg_name = HWC_WORKSPACE_SG_NAME.format(workspace_name)
    target_sgs = []
    for _sg in _sgs:
        if _sg_name == _sg.name:
            target_sgs.append(_sg)
    return target_sgs


def _check_and_delete_eip(config, workspace_name):
    eip_client = make_eip_client(config)
    _eip_name = HWC_WORKSPACE_EIP_NAME.format(workspace_name)
    public_ips = eip_client.list_publicips(ListPublicipsRequest()).publicips
    _found = False
    for _public_ip in public_ips:
        if _public_ip.alias == _eip_name:
            cli_logger.print(
                "Deleting elastic IP: {}...", _eip_name)
            eip_client.delete_publicip(
                DeletePublicipRequest(publicip_id=_public_ip.id))
            cli_logger.print(
                "Successfully deleted elastic IP: {}.", _eip_name)
            _found = True
    if not _found:
        cli_logger.print(
            "No elastic IP with the name found: {}. Skip deletion.", _eip_name)


def _check_and_delete_vpc_peering_connection(config, vpc_client,
                                             workspace_name):
    peerings = _get_vpc_peering_conn(vpc_client, workspace_name)
    for _peering_conn in peerings:
        vpc_client.delete_vpc_peering(
            DeleteVpcPeeringRequest(
                peering_id=_peering_conn.id
            )
        )
        cli_logger.print(
            "Delete peering VPC connection {}".format(
                _peering_conn.name))

    # Delete route table rule from current VPC to workspace VPC
    _current_vpc = _get_current_vpc(config)
    _workspace_vpc = get_workspace_vpc(vpc_client, workspace_name)
    if _workspace_vpc and _current_vpc:
        # Delete route table from current vpc to workspace vpc
        _current_vpc_rts = vpc_client.list_route_tables(
            ListRouteTablesRequest(vpc_id=_current_vpc.id)
        ).routetables
        for _current_vpc_rt in _current_vpc_rts:
            vpc_client.update_route_table(
                UpdateRouteTableRequest(
                    routetable_id=_current_vpc_rt.id,
                    body=UpdateRoutetableReqBody(
                        UpdateRouteTableReq(routes={
                            'del': [RouteTableRoute(
                                destination=_workspace_vpc.cidr)]}
                        )
                    )
                )
            )
        # Delete route table from workspace vpc to current vpc
        _workspace_vpc_rts = vpc_client.list_route_tables(
            ListRouteTablesRequest(vpc_id=_workspace_vpc.id)
        ).routetables
        for _workspace_vpc_rt in _workspace_vpc_rts:
            vpc_client.update_route_table(
                UpdateRouteTableRequest(
                    routetable_id=_workspace_vpc_rt.id,
                    body=UpdateRoutetableReqBody(
                        UpdateRouteTableReq(routes={
                            'del': [RouteTableRoute(
                                destination=_current_vpc.cidr)]}
                        )
                    )
                )
            )


def _get_vpc_peering_conn(vpc_client, workspace_name):
    _peering_name = HWC_WORKSPACE_VPC_PEERING_NAME.format(
        workspace_name)
    peerings = vpc_client.list_vpc_peerings(
        ListVpcPeeringsRequest(name=_peering_name)
    ).peerings
    return peerings


def _check_and_delete_nat_gateway_and_eip(config, workspace_name):
    current_step = 1
    total_steps = 3

    with cli_logger.group("Deleting SNAT rules",
                          _numbered=("()", current_step, total_steps)):
        current_step += 1
        _check_and_delete_snat_rules(config, workspace_name)

    with cli_logger.group("Deleting NAT gateway",
                          _numbered=("()", current_step, total_steps)):
        current_step += 1
        _check_and_delete_nat_gateway(config, workspace_name)

    with cli_logger.group("Deleting elastic IP",
                          _numbered=("()", current_step, total_steps)):
        current_step += 1
        # Delete EIP
        _check_and_delete_eip(config, workspace_name)


def _check_and_delete_snat_rules(config, workspace_name):
    nat_client = make_nat_client(config)
    nat_gateways = _get_workspace_nat(nat_client, workspace_name)
    if not nat_gateways:
        cli_logger.print(
            "No NAT gateway found for workspace: {}. Skip deletion.".format(
                workspace_name))
        return

    for _nat_gateway in nat_gateways:
        _check_and_delete_snat_rules_of_nat_gateway(nat_client, _nat_gateway)


def _check_and_delete_nat_gateway(config, workspace_name):
    nat_client = make_nat_client(config)
    nat_gateways = _get_workspace_nat(nat_client, workspace_name)
    if not nat_gateways:
        cli_logger.print(
            "No NAT gateway found for workspace: {}. Skip deletion.", workspace_name)
        return

    for _nat_gateway in nat_gateways:
        # Delete NAT
        cli_logger.print(
            "Deleting NAT gateway: {}...", _nat_gateway.name)
        nat_client.delete_nat_gateway(DeleteNatGatewayRequest(
            nat_gateway_id=_nat_gateway.id))
        cli_logger.print(
            "Successfully deleted NAT gateway: {}.", _nat_gateway.name)


def _check_and_delete_snat_rules_of_nat_gateway(nat_client, nat_gateway):
    # Delete SNAT rules and wait util empty
    cli_logger.print(
        "Deleting SNAT rules for NAT gateway: {}...", nat_gateway.name)
    while True:
        _snat_rules = nat_client.list_nat_gateway_snat_rules(
            ListNatGatewaySnatRulesRequest(nat_gateway_id=[nat_gateway.id])
        ).snat_rules
        if not _snat_rules:
            break
        for _rule in _snat_rules:
            nat_client.delete_nat_gateway_snat_rule(
                DeleteNatGatewaySnatRuleRequest(
                    nat_gateway_id=nat_gateway.id,
                    snat_rule_id=_rule.id)
            )
        time.sleep(1)
    cli_logger.print(
        "Successfully deleted SNAT rules for NAT gateway: {}.", nat_gateway.name)


def _get_workspace_nat(nat_client, workspace_name=None, nat_id=None):
    _nat_name = HWC_WORKSPACE_NAT_NAME.format(
        workspace_name) if workspace_name else None
    nat_gateways = nat_client.list_nat_gateways(
        ListNatGatewaysRequest(name=_nat_name, id=nat_id)
    ).nat_gateways
    return nat_gateways


def update_huaweicloud_workspace_firewalls(config):
    vpc_client = make_vpc_client(config)
    workspace_name = config["workspace_name"]
    workspace_vpc = get_workspace_vpc(vpc_client, workspace_name)
    if not workspace_vpc:
        cli_logger.print("Can't find workspace VPC")
        return
    current_step = 1
    total_steps = 1
    try:

        with cli_logger.group("Updating workspace firewalls",
                              _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _sgs = vpc_client.list_security_groups(
                ListSecurityGroupsRequest()).security_groups
            _sg_name = HWC_WORKSPACE_SG_NAME.format(workspace_name)
            for _sg in _sgs:
                if _sg.name == _sg_name:
                    _update_security_group_rules(config, _sg, workspace_vpc,
                                                 vpc_client)
    except Exception as e:
        cli_logger.error("Failed to update the firewalls of workspace {}. {}",
                         workspace_name, str(e))
        raise e

    cli_logger.print("Successfully updated the firewalls of workspace: {}.",
                     cf.bold(workspace_name))


def get_huaweicloud_workspace_info(config):
    info = {}
    workspace_name = config["workspace_name"]
    obs_client = make_obs_client(config)
    bucket_name = _get_managed_obs_bucket(obs_client, workspace_name)

    if bucket_name:
        resp = obs_client.getBucketLocation(bucket_name)
        if resp.status < 300:
            managed_cloud_storage_uri = resp.body.location
            managed_cloud_storage = {
                HWC_MANAGED_CLOUD_STORAGE_OBS_BUCKET: bucket_name,
                CLOUDTIK_MANAGED_CLOUD_STORAGE_URI: managed_cloud_storage_uri}
            info[CLOUDTIK_MANAGED_CLOUD_STORAGE] = managed_cloud_storage

    return info


def check_huaweicloud_workspace_existence(config):
    workspace_name = config["workspace_name"]
    managed_cloud_storage = is_managed_cloud_storage(config)
    use_peering_vpc = is_use_peering_vpc(config)
    use_working_vpc = is_use_working_vpc(config)

    existing_resources = 0
    target_resources = HWC_WORKSPACE_TARGET_RESOURCES
    if managed_cloud_storage:
        target_resources += 1
    if use_peering_vpc:
        target_resources += 1

    vpc_client = make_vpc_client(config)
    if use_working_vpc:
        workspace_vpc = _get_current_vpc(config)
    else:
        workspace_vpc = get_workspace_vpc(vpc_client, workspace_name)
    # workspace VPC check
    if workspace_vpc:
        existing_resources += 1

        # private subnets check
        _private_subnets_count = len(
            _get_workspace_vpc_subnets(vpc_client, workspace_vpc,
                                       workspace_name, 'private')
        )
        if _private_subnets_count >= HWC_WORKSPACE_VPC_SUBNETS_COUNT - 1:
            existing_resources += 1

        # public subnet check
        _public_subnets_count = len(
            _get_workspace_vpc_subnets(vpc_client, workspace_vpc,
                                       workspace_name, 'public')
        )
        if _public_subnets_count >= 0:
            existing_resources += 1

        # NAT gateway check
        nat_client = make_nat_client(config)
        if len(_get_workspace_nat(nat_client, workspace_name)) > 0:
            existing_resources += 1
        # Security group check
        if len(_get_workspace_security_group(vpc_client, workspace_name)) > 0:
            existing_resources += 1
        # VPC peering connection check
        if use_peering_vpc:
            if len(_get_vpc_peering_conn(vpc_client, workspace_name)) > 0:
                existing_resources += 1

    # Check instance profile
    if _get_instance_profile(config, workspace_name, for_head=True):
        existing_resources += 1
    if _get_instance_profile(config, workspace_name, for_head=False):
        existing_resources += 1

    # Managed cloud storage
    cloud_storage_existence = False
    if managed_cloud_storage:
        obs_client = make_obs_client(config)
        if _get_managed_obs_bucket(obs_client, workspace_name):
            existing_resources += 1
            cloud_storage_existence = True

    if existing_resources == 0:
        return Existence.NOT_EXIST
    elif existing_resources == target_resources:
        return Existence.COMPLETED
    else:
        if existing_resources == 1 and cloud_storage_existence:
            return Existence.STORAGE_ONLY
        return Existence.IN_COMPLETED


def check_huaweicloud_workspace_integrity(config):
    existence = check_huaweicloud_workspace_existence(config)
    return True if existence == Existence.COMPLETED else False


def _get_instance_profile(config, workspace_name, for_head):
    iam_client = make_iam_client(config)
    _prefix = 'head' if for_head else 'worker'
    profile_name = HWC_WORKSPACE_INSTANCE_PROFILE_NAME.format(workspace_name,
                                                              _prefix)
    domain_id = _get_current_domain_id(iam_client)
    target_agencies = iam_client.list_agencies(
        ListAgenciesRequest(name=profile_name, domain_id=domain_id)
    ).agencies
    if len(target_agencies) >= 1:
        return target_agencies[0]
    else:
        return None


def list_huaweicloud_clusters(config):
    head_nodes = get_workspace_head_nodes(config)
    clusters = {}
    for head_node in head_nodes:
        cluster_name = get_cluster_name_from_head(head_node)
        if cluster_name:
            clusters[cluster_name] = _get_node_info(head_node)
    return clusters


def get_cluster_name_from_head(head_node) -> Optional[str]:
    tags = tags_list_to_dict(head_node.tags)
    for tag_k, tag_v in tags.items():
        if tag_k == CLOUDTIK_TAG_CLUSTER_NAME:
            return tag_v
    return None


def get_workspace_head_nodes(config):
    return _get_workspace_head_nodes(config['provider'],
                                     config['workspace_name'])


def _get_workspace_head_nodes(provider_config, workspace_name):
    vpc_client = _make_vpc_client(provider_config)
    workspace_vpc = get_workspace_vpc(vpc_client, workspace_name)
    if not workspace_vpc:
        raise RuntimeError(
            "Failed to get the VPC. The workspace {} doesn't exist or is in "
            "the wrong state.".format(workspace_name))

    ecs_client = _make_ecs_client(provider_config)
    _tags = flat_tags_map({CLOUDTIK_TAG_NODE_KIND: NODE_KIND_HEAD, })
    head_nodes = ecs_client.list_servers_details(
        ListServersDetailsRequest(status=HWC_SERVER_STATUS_ACTIVE,
                                  tags=_tags)).servers

    return [head_node for head_node in head_nodes if
            head_node.metadata.get('vpc_id') == workspace_vpc.id]


def bootstrap_huaweicloud_workspace(config):
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
        "IpRanges": [{"CidrIp": allowed_ssh_source} for allowed_ssh_source in
                     allowed_ssh_sources]
    }
    ip_permissions.append(ip_permission)


def with_huaweicloud_environment_variables(provider_config,
                                           node_type_config: Dict[str, Any],
                                           node_id: str):
    config_dict = {}
    export_huaweicloud_obs_storage_config(provider_config, config_dict)

    if node_type_config is not None:
        node_config = node_type_config.get("node_config")
        agency_name = node_config.get("metadata", {}).get("agency_name")
        if agency_name:
            config_dict["HUAWEICLOUD_ECS_AGENCY_NAME"] = agency_name

    return config_dict


def post_prepare_huaweicloud(config: Dict[str, Any]) -> Dict[str, Any]:
    try:
        config = fill_available_node_types_resources(config)
    except Exception as exc:
        cli_logger.warning(
            "Failed to detect node resources. Make sure you have properly "
            "configured the Huawei Cloud credentials: {}.", str(exc))
        raise
    return config


def fill_available_node_types_resources(
        cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    """Fills out missing "resources" field for available_node_types."""
    if "available_node_types" not in cluster_config:
        return cluster_config
    cluster_config = copy.deepcopy(cluster_config)

    # Get instance flavor information from cloud provider
    ecs_client = _make_ecs_client(cluster_config['provider'])
    flavors = ecs_client.list_flavors(ListFlavorsRequest()).flavors
    flavor_dict = {flavor.id: flavor for flavor in flavors}
    # resource format of node_type:
    # {'CPU': x, 'memory': y, 'GPU': z, 'Ascend': t, 'FPGA': u,
    #  'accelerator_type:{acc_name}': w}
    available_node_types = cluster_config['available_node_types']
    for key, node_type in available_node_types.items():
        detected_resources = {}
        flavor_ref = node_type['node_config']['flavor_ref']
        if flavor_ref in flavor_dict:
            provider_flavor = flavor_dict[flavor_ref]
            cpus = provider_flavor.vcpus
            detected_resources['CPU'] = int(cpus)

            memory_total = provider_flavor.ram
            memory_total_in_bytes = int(memory_total) * 1024 * 1024
            detected_resources['memory'] = memory_total_in_bytes

            # Update GPU, Ascend, FPGA and other accelerator if exist
            os_extra_specs = provider_flavor.os_extra_specs
            if os_extra_specs and os_extra_specs.resource_type:
                _name = os_extra_specs.resource_type
                detected_resources[f'accelerator_type:{_name}'] = 1
                if os_extra_specs.ecsperformancetype == 'gpu':
                    detected_resources['GPU'] = 1
                if os_extra_specs.ecsperformancetype == 'ascend':
                    detected_resources['Ascend'] = 1
                if os_extra_specs.ecsperformancetype == 'fpga':
                    detected_resources['FPGA'] = 1

            config_resources = node_type.get('resources', {})
            # Override by config resources
            detected_resources.update(config_resources)
            # Merge back when find new resources
            if detected_resources != config_resources:
                node_type['resources'] = detected_resources
                logger.debug('Updating the resources of {} to {}.'.format(
                    key, detected_resources))
        else:
            region = cluster_config['provider']['region']
            raise ValueError('Server flavor {} is not available in Huawei '
                             'Cloud {}'.format(flavor_ref, region))

    return cluster_config


def bootstrap_huaweicloud(config):
    return bootstrap_huaweicloud_from_workspace(config)


def verify_obs_storage(provider_config: Dict[str, Any]):
    obs_storage = get_huaweicloud_obs_storage_config(provider_config)
    if obs_storage is None:
        return
    obs_client = make_obs_client_aksk(obs_storage.get('obs.access.key'),
                                      obs_storage.get('obs.secret.key'),
                                      region=provider_config.get('region'))
    bucket_name = obs_storage.get(HWC_OBS_BUCKET)
    try:
        _get_managed_obs_bucket(obs_client, bucket_name=bucket_name)
    except Exception as e:
        raise StorageTestingError("Error happens when verifying OBS storage "
                                  "configurations. If you want to go without "
                                  "passing the verification, set "
                                  "'verify_cloud_storage' to False under "
                                  "provider config. Error: {}.".format(
            e.message)) from None


def bootstrap_huaweicloud_from_workspace(config):
    workspace_name = config.get("workspace_name", "")
    if not workspace_name:
        raise RuntimeError(
            "Workspace name is not specified in cluster configuration.")
    if not check_huaweicloud_workspace_integrity(config):
        cli_logger.abort(
            "HUAWEI CLOUD workspace {} doesn't exist or is in wrong state.",
            workspace_name)

    # create a copy of the input config to modify
    config = copy.deepcopy(config)

    # Used internally to store head instance profile.
    config["head_node"] = {}

    # The head node needs to have an instance profile that allows it to create
    # further ECS instances.
    config = _configure_instance_profile_from_workspace(config)

    # Set obs.bucket if use_managed_cloud_storage=True
    config = _configure_cloud_storage_from_workspace(config)

    # Configure SSH access, using an existing key pair if possible.
    config = _configure_key_pair(config)

    # Pick a reasonable subnet if not specified by the user.
    config = _configure_subnet_from_workspace(config)

    # Cluster workers should be in a security group that permits traffic within
    # the group, and also SSH access from outside.
    config = _configure_security_group_from_workspace(config)

    config = _configure_prefer_spot_node(config)

    return config


def _configure_prefer_spot_node(config):
    prefer_spot_node = config["provider"].get("prefer_spot_node")

    # if no such key, we consider user don't want to override
    if not prefer_spot_node:
        return config

    # User override, set or remove spot settings for worker node types
    for key, node_type in config["available_node_types"].items():
        # Skip head node
        if key != config["head_node_type"]:
            node_config = node_type["node_config"]
            node_config.setdefault("extendparam", {})
            if prefer_spot_node:
                # Add spot instruction
                node_config["extendparam"]["market_type"] = "spot"
            else:
                # Remove spot instruction
                node_config["extendparam"].pop("market_type")

    return config


def _configure_security_group_from_workspace(config):
    workspace_name = config["workspace_name"]
    vpc_client = make_vpc_client(config)
    workspace_vpc = get_workspace_vpc(vpc_client, workspace_name)

    if not workspace_vpc:
        error_msg = "Can't find workspace VPC in {}".format(workspace_name)
        cli_logger.abort(msg=error_msg, exc=RuntimeError(error_msg))

    sgs = _get_workspace_security_group(vpc_client, workspace_name)

    for _, node_type in config["available_node_types"].items():
        node_config = node_type["node_config"]
        node_config["security_groups"] = [{"id": sg.id} for sg in sgs]

    return config


def _configure_subnet_from_workspace(config):
    workspace_name = config["workspace_name"]
    use_internal_ips = is_use_internal_ip(config)
    vpc_client = make_vpc_client(config)
    workspace_vpc = get_workspace_vpc(vpc_client, workspace_name)

    if not workspace_vpc:
        error_msg = "Can't find workspace VPC in {}".format(workspace_name)
        cli_logger.abort(msg=error_msg, exc=RuntimeError(error_msg))

    private_subnets = _get_workspace_vpc_subnets(vpc_client, workspace_vpc,
                                                 workspace_name,
                                                 category='private')
    public_subnets = _get_workspace_vpc_subnets(vpc_client, workspace_vpc,
                                                workspace_name,
                                                category='public')

    public_subnet_ids = [{"subnet_id": public_subnet.id} for public_subnet
                         in public_subnets]
    private_subnet_ids = [{"subnet_id": private_subnet.id} for private_subnet
                          in private_subnets]

    for key, node_type in config["available_node_types"].items():
        node_config = node_type["node_config"]
        node_config["vpcid"] = workspace_vpc.id
        # head node
        if key == config["head_node_type"]:
            if use_internal_ips:
                node_config["nics"] = private_subnet_ids
            else:
                node_config["nics"] = public_subnet_ids
        # worker nodes
        else:
            node_config["nics"] = private_subnet_ids
    return config


def _configure_key_pair(config):
    # Two ways to configure cluster key in HUAWEI CLOUD provider.
    # 1. Explicit cluster private key: specify ssh_private_key in config, and
    # you also need to explicitly specify HUAWEI CLOUD key pair name of the
    # private key through 'key_name' in the 'node_config' of each node types
    # defined in 'available_node_types'
    # 2. Implicit cluster private key: don't specify ssh_private_key in config,
    # provider will create new key pair with name pattern of
    # cloudtik_huaweicloud_{index} and download private key file of the
    # key pair to ~/.ssh/cloudtik_huaweicloud_{index}.pem. If you create
    # multiple clusters on the same machine, CloudTik will use the same private
    # key, if you don't want this, you should specify explicitly key pair
    # as following.
    node_types = config["available_node_types"]
    if 'ssh_private_key' in config['auth']:
        # Explicit cluster private key
        # Use ssh_private_key and key_name in config
        for node_type_key, value in node_types.items():
            if 'key_name' not in value['node_config']:
                error_msg = 'No key_name in node_config of ' \
                            'available_node_types {}'.format(node_type_key)
                cli_logger.abort(msg=error_msg, exc=RuntimeError(error_msg))
        return config
    else:
        # Try a few times to get or create a good key pair.
        key_name, key_path = _get_available_key_pair(config)
        if key_name and key_path:
            config["auth"]["ssh_private_key"] = key_path
            for node_type in node_types.values():
                node_config = node_type["node_config"]
                node_config["key_name"] = key_name

            return config
        else:
            error_msg = "Can't find available key pair in cloud and local. " \
                        "Consider deleting some unused keys pairs " \
                        "from your account."
            cli_logger.abort(msg=error_msg, exc=RuntimeError(error_msg))


def _get_available_key_pair(config):
    key_name = key_path = None
    ecs_client = make_ecs_client(config)
    MAX_NUM_KEYS = 30
    for i in range(MAX_NUM_KEYS):
        key_name = HWC_KEY_PAIR_NAME.format(i)
        key_path = os.path.expanduser(
            HWC_KEY_PATH_TEMPLATE.format(key_name))
        if os.path.exists(key_path) and _check_key_pair_exist(ecs_client,
                                                              key_name):
            cli_logger.print(
                "find key pair {} in cloud and local {}".format(key_name,
                                                                key_path))
            break
        elif not os.path.exists(key_path) and not _check_key_pair_exist(
                ecs_client, key_name):
            # create new key pair
            target_key = ecs_client.nova_create_keypair(
                NovaCreateKeypairRequest(
                    body=NovaCreateKeypairRequestBody(
                        NovaCreateKeypairOption(name=key_name)))).keypair
            # Implicit cluster private key
            # Find available key pair and create new key pair
            os.makedirs(os.path.expanduser("~/.ssh"), exist_ok=True)
            # We need to make sure to _create_ the file with the right
            # permissions. In order to do that we need to change the
            # default os.open behavior to include the mode we want.
            with open(key_path, "w",
                      opener=partial(os.open, mode=0o600)) as f:
                f.write(target_key.private_key)
            break
        else:
            # find next key
            key_name = key_path = None
            continue
    return key_name, key_path


def _check_key_pair_exist(ecs_client, key_name):
    # check key pair in HUAWEI CLOUD
    try:
        target_key = ecs_client.nova_show_keypair(
            NovaShowKeypairRequest(keypair_name=key_name)).keypair
    except ServiceResponseException:
        # 404 not found
        target_key = None
    return target_key


def _configure_cloud_storage_from_workspace(config):
    use_managed_cloud_storage = is_use_managed_cloud_storage(config)
    if use_managed_cloud_storage:
        _configure_managed_cloud_storage_from_workspace(config,
                                                        config["provider"])
    return config


def _configure_managed_cloud_storage_from_workspace(config, cloud_provider):
    workspace_name = config["workspace_name"]
    obs_client = _make_obs_client(cloud_provider)
    obs_bucket_name = _get_managed_obs_bucket(obs_client, workspace_name)
    if obs_bucket_name is None:
        cli_logger.abort(
            "No managed OBS bucket was found. If you want to use managed OBS "
            "bucket, you should set managed_cloud_storage equal to True when "
            "you creating workspace.")

    cloud_storage = get_huaweicloud_obs_storage_config_for_update(
        config["provider"])
    cloud_storage[HWC_OBS_BUCKET] = obs_bucket_name


def _configure_instance_profile_from_workspace(config):
    worker_for_cloud_storage = is_worker_role_for_cloud_storage(config)
    workspace_name = config["workspace_name"]
    head_profile = _get_instance_profile(config,
                                         workspace_name=workspace_name,
                                         for_head=True)
    if worker_for_cloud_storage:
        worker_profile = _get_instance_profile(config,
                                               workspace_name=workspace_name,
                                               for_head=False)

    if not head_profile:
        raise RuntimeError("Workspace head instance profile not found!")
    if worker_for_cloud_storage and not worker_profile:
        raise RuntimeError("Workspace worker instance profile not found!")

    for key, node_type in config["available_node_types"].items():
        node_config = node_type["node_config"]
        node_config.setdefault("metadata", {})
        if key == config["head_node_type"]:
            node_config["metadata"]["agency_name"] = head_profile.name
            config["head_node"]["agency_name"] = head_profile.name
        elif worker_for_cloud_storage:
            node_config["metadata"]["agency_name"] = worker_profile.name

    return config
