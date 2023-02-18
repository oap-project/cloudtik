import json
import logging
import time
from typing import Any, Dict, List

from aliyunsdkcore import client
from aliyunsdkcore.request import CommonRequest
from aliyunsdkcore.acs_exception.exceptions import ClientException, ServerException

from aliyunsdkecs.request.v20140526.AllocatePublicIpAddressRequest import (
    AllocatePublicIpAddressRequest,
)
from aliyunsdkecs.request.v20140526.AuthorizeSecurityGroupRequest import (
    AuthorizeSecurityGroupRequest,
)
from aliyunsdkecs.request.v20140526.CreateInstanceRequest import CreateInstanceRequest
from aliyunsdkecs.request.v20140526.CreateKeyPairRequest import CreateKeyPairRequest
from aliyunsdkecs.request.v20140526.CreateSecurityGroupRequest import (
    CreateSecurityGroupRequest,
)

from aliyunsdkecs.request.v20140526.CreateVSwitchRequest import CreateVSwitchRequest
from aliyunsdkecs.request.v20140526.DeleteVSwitchRequest import DeleteVSwitchRequest
from aliyunsdkecs.request.v20140526.DeleteInstanceRequest import DeleteInstanceRequest
from aliyunsdkecs.request.v20140526.DeleteInstancesRequest import DeleteInstancesRequest
from aliyunsdkecs.request.v20140526.DeleteKeyPairsRequest import DeleteKeyPairsRequest
from aliyunsdkecs.request.v20140526.DescribeInstancesRequest import (
    DescribeInstancesRequest,
)
from aliyunsdkecs.request.v20140526.DescribeKeyPairsRequest import (
    DescribeKeyPairsRequest,
)
from aliyunsdkecs.request.v20140526.DescribeSecurityGroupsRequest import (
    DescribeSecurityGroupsRequest,
)
from aliyunsdkecs.request.v20140526.DescribeSecurityGroupAttributeRequest import DescribeSecurityGroupAttributeRequest
from aliyunsdkecs.request.v20140526.RevokeSecurityGroupRequest import RevokeSecurityGroupRequest
from aliyunsdkecs.request.v20140526.DeleteSecurityGroupRequest import DeleteSecurityGroupRequest
from aliyunsdkecs.request.v20140526.DescribeVSwitchesRequest import (
    DescribeVSwitchesRequest,
)
from aliyunsdkecs.request.v20140526.DescribeZonesRequest import DescribeZonesRequest
from aliyunsdkecs.request.v20140526.ImportKeyPairRequest import ImportKeyPairRequest
from aliyunsdkecs.request.v20140526.RunInstancesRequest import RunInstancesRequest
from aliyunsdkecs.request.v20140526.StartInstanceRequest import StartInstanceRequest
from aliyunsdkecs.request.v20140526.StopInstanceRequest import StopInstanceRequest
from aliyunsdkecs.request.v20140526.StopInstancesRequest import StopInstancesRequest
from aliyunsdkecs.request.v20140526.TagResourcesRequest import TagResourcesRequest

from aliyunsdkvpc.request.v20160428.CreateNatGatewayRequest import CreateNatGatewayRequest
from aliyunsdkvpc.request.v20160428.DescribeNatGatewaysRequest import DescribeNatGatewaysRequest
from aliyunsdkvpc.request.v20160428.DeleteNatGatewayRequest import DeleteNatGatewayRequest
from aliyunsdkvpc.request.v20160428.AllocateEipAddressRequest import AllocateEipAddressRequest
from aliyunsdkvpc.request.v20160428.AssociateEipAddressRequest import AssociateEipAddressRequest
from aliyunsdkvpc.request.v20160428.DescribeEipAddressesRequest import DescribeEipAddressesRequest
from aliyunsdkvpc.request.v20160428.UnassociateEipAddressRequest import UnassociateEipAddressRequest
from aliyunsdkvpc.request.v20160428.ReleaseEipAddressRequest import ReleaseEipAddressRequest
from aliyunsdkvpc.request.v20160428.CreateSnatEntryRequest import CreateSnatEntryRequest
from aliyunsdkvpc.request.v20160428.DescribeSnatTableEntriesRequest import DescribeSnatTableEntriesRequest
from aliyunsdkvpc.request.v20160428.DeleteSnatEntryRequest import DeleteSnatEntryRequest
from aliyunsdkvpc.request.v20160428.CreateVpcRequest import CreateVpcRequest
from aliyunsdkvpc.request.v20160428.DeleteVpcRequest import DeleteVpcRequest
from aliyunsdkvpc.request.v20160428.DescribeVpcsRequest import DescribeVpcsRequest
from aliyunsdkvpc.request.v20160428.TagResourcesRequest import TagResourcesRequest
from aliyunsdkvpc.request.v20160428.UnTagResourcesRequest import UnTagResourcesRequest
from aliyunsdkvpc.request.v20160428.DescribeRouteTableListRequest import DescribeRouteTableListRequest
from aliyunsdkvpc.request.v20160428.CreateRouteEntryRequest import CreateRouteEntryRequest
from aliyunsdkvpc.request.v20160428.DescribeRouteEntryListRequest import DescribeRouteEntryListRequest
from aliyunsdkvpc.request.v20160428.DeleteRouteEntryRequest import DeleteRouteEntryRequest

from aliyunsdkram.request.v20150501.CreateRoleRequest import CreateRoleRequest
from aliyunsdkram.request.v20150501.GetRoleRequest import GetRoleRequest
from aliyunsdkram.request.v20150501.DeleteRoleRequest import DeleteRoleRequest
from aliyunsdkram.request.v20150501.AttachPolicyToRoleRequest import AttachPolicyToRoleRequest
from aliyunsdkram.request.v20150501.DetachPolicyFromRoleRequest import DetachPolicyFromRoleRequest
from aliyunsdkram.request.v20150501.ListPoliciesForRoleRequest import ListPoliciesForRoleRequest


from cloudtik.core._private.constants import env_integer, CLOUDTIK_DEFAULT_CLOUD_STORAGE_URI
from cloudtik.core._private.utils import get_storage_config_for_update, format_exception_message

ALIYUN_OSS_BUCKET = "oss.bucket"

ACS_MAX_RETRIES = env_integer("ACS_MAX_RETRIES", 12)


from cloudtik.core._private.cli_logger import cli_logger, cf

from alibabacloud_credentials.client import Client as CredentialClient
from alibabacloud_credentials.models import Config
from alibabacloud_ecs20140526 import models as ecs_models
from alibabacloud_ecs20140526.client import Client as ecs_client
from alibabacloud_oss20190517 import models as oss_models
from alibabacloud_oss20190517.client import Client as oss_client
from alibabacloud_ram20150501 import models as ram_models
from alibabacloud_ram20150501.client import Client as ram_client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_vpc20160428 import models as vpc_models
from alibabacloud_vpc20160428.client import Client as vpc_client
from alibabacloud_vpcpeer20220101 import models as vpc_peer_models
from alibabacloud_vpcpeer20220101.client import Client as vpc_peer_client
from Tea.exceptions import TeaException, UnretryableException


class AcsClient:
    """
    A wrapper around Aliyun SDK.

    Parameters:
        access_key: The AccessKey ID of your aliyun account.
        access_key_secret: The AccessKey secret of your aliyun account.
        region_id: A region is a geographic area where a data center resides.
                   Region_id is the ID of region (e.g., cn-hangzhou,
                   us-west-1, etc.)
        max_retries: The maximum number of retries each connection.
    """

    def __init__(self, region_id, max_retries, access_key=None, access_key_secret=None):
        if (access_key and access_key_secret):
            self.cli = client.AcsClient(
                ak=access_key,
                secret=access_key_secret,
                max_retry_time=max_retries,
                region_id=region_id)
        else:
            client.AcsClient(
                max_retry_time=max_retries,
                region_id=region_id)

    def describe_instances(self, tags=None, instance_ids=None):
        """Query the details of one or more Elastic Compute Service (ECS) instances.

        :param tags: The tags of the instance.
        :param instance_ids: The IDs of ECS instances
        :return: ECS instance list
        """
        request = DescribeInstancesRequest()
        if tags is not None:
            request.set_Tags(tags)
        if instance_ids is not None:
            request.set_InstanceIds(instance_ids)
        response = self._send_request(request)
        if response is not None:
            instance_list = response.get("Instances").get("Instance")
            return instance_list
        return None

    def create_instance(
        self,
        instance_type,
        image_id,
        tags,
        key_pair_name,
        optimized="optimized",
        instance_charge_type="PostPaid",
        spot_strategy="SpotWithPriceLimit",
        internet_charge_type="PayByTraffic",
        internet_max_bandwidth_out=5,
    ):
        """Create a subscription or pay-as-you-go ECS instance.

        :param instance_type: The instance type of the ECS.
        :param image_id: The ID of the image used to create the instance.
        :param tags: The tags of the instance.
        :param key_pair_name: The name of the key pair to be bound to
                              the instance.
        :param optimized: Specifies whether the instance is I/O optimized
        :param instance_charge_type: The billing method of the instance.
                                     Default value: PostPaid.
        :param spot_strategy: The preemption policy for the pay-as-you-go
                              instance.
        :param internet_charge_type: The billing method for network usage.
                                     Default value: PayByTraffic.
        :param internet_max_bandwidth_out: The maximum inbound public
                                           bandwidth. Unit: Mbit/s.
        :return: The created instance ID.
        """
        request = CreateInstanceRequest()
        request.set_InstanceType(instance_type)
        request.set_ImageId(image_id)
        request.set_IoOptimized(optimized)
        request.set_InstanceChargeType(instance_charge_type)
        request.set_SpotStrategy(spot_strategy)
        request.set_InternetChargeType(internet_charge_type)
        request.set_InternetMaxBandwidthOut(internet_max_bandwidth_out)
        request.set_KeyPairName(key_pair_name)
        request.set_Tags(tags)

        response = self._send_request(request)
        if response is not None:
            instance_id = response.get("InstanceId")
            logging.info("instance %s created task submit successfully.", instance_id)
            return instance_id
        logging.error("instance created failed.")
        return None

    def run_instances(
        self,
        instance_type,
        image_id,
        tags,
        security_group_id,
        vswitch_id,
        key_pair_name,
        amount=1,
        optimized="optimized",
        instance_charge_type="PostPaid",
        spot_strategy="SpotWithPriceLimit",
        internet_charge_type="PayByTraffic",
        internet_max_bandwidth_out=1,
    ):
        """Create one or more pay-as-you-go or subscription
            Elastic Compute Service (ECS) instances

        :param instance_type: The instance type of the ECS.
        :param image_id: The ID of the image used to create the instance.
        :param tags: The tags of the instance.
        :param security_group_id: The ID of the security group to which to
                                  assign the instance. Instances in the same
                                  security group can communicate with
                                  each other.
        :param vswitch_id: The ID of the vSwitch to which to connect
                           the instance.
        :param key_pair_name: The name of the key pair to be bound to
                              the instance.
        :param amount: The number of instances that you want to create.
        :param optimized: Specifies whether the instance is I/O optimized
        :param instance_charge_type: The billing method of the instance.
                                     Default value: PostPaid.
        :param spot_strategy: The preemption policy for the pay-as-you-go
                              instance.
        :param internet_charge_type: The billing method for network usage.
                                     Default value: PayByTraffic.
        :param internet_max_bandwidth_out: The maximum inbound public
                                           bandwidth. Unit: Mbit/s.
        :return: The created instance IDs.
        """
        request = RunInstancesRequest()
        request.set_InstanceType(instance_type)
        request.set_ImageId(image_id)
        request.set_IoOptimized(optimized)
        request.set_InstanceChargeType(instance_charge_type)
        request.set_SpotStrategy(spot_strategy)
        request.set_InternetChargeType(internet_charge_type)
        request.set_InternetMaxBandwidthOut(internet_max_bandwidth_out)
        request.set_Tags(tags)
        request.set_Amount(amount)
        request.set_SecurityGroupId(security_group_id)
        request.set_VSwitchId(vswitch_id)
        request.set_KeyPairName(key_pair_name)

        response = self._send_request(request)
        if response is not None:
            instance_ids = response.get("InstanceIdSets").get("InstanceIdSet")
            return instance_ids
        logging.error("instance created failed.")
        return None

    def create_security_group(self, vpc_id, name):
        """Create a security group

        :param vpc_id: The ID of the VPC in which to create
                       the security group.
        :return: The created security group ID.
        """
        request = CreateSecurityGroupRequest()
        request.set_VpcId(vpc_id)
        request.set_SecurityGroupName(name)
        response = self._send_request(request)
        if response is not None:
            security_group_id = response.get("SecurityGroupId")
            return security_group_id
        return None

    def describe_security_groups(self, vpc_id=None, tags=None):
        """Query basic information of security groups.

        :param vpc_id: The ID of the VPC to which the security group belongs.
        :param tags: The tags of the security group.
        :return: Security group list.
        """
        request = DescribeSecurityGroupsRequest()
        if vpc_id is not None:
            request.set_VpcId(vpc_id)
        if tags is not None:
            request.set_Tags(tags)
        response = self._send_request(request)
        if response is not None:
            security_groups = response.get("SecurityGroups").get("SecurityGroup")
            return security_groups
        logging.error("describe security group failed.")
        return None
    
    def revoke_security_group(self, ip_protocol, port_range, security_group_id, source_cidr_ip):
        """Revoke an inbound security group rule.

                :param ip_protocol: The transport layer protocol.
                :param port_range: The range of destination ports relevant to
                                   the transport layer protocol.
                :param security_group_id: The ID of the destination security group.
                :param source_cidr_ip: The range of source IPv4 addresses.
                                       CIDR blocks and IPv4 addresses are supported.
                """
        request = RevokeSecurityGroupRequest()
        request.set_IpProtocol(ip_protocol)
        request.set_PortRange(port_range)
        request.set_SecurityGroupId(security_group_id)
        request.set_SourceCidrIp(source_cidr_ip)
        self._send_request(request)
    
    def describe_security_group_attribute(self, security_group_id, tags=None):
        """Query basic information of security groups.

        :param vpc_id: The ID of the VPC to which the security group belongs.
        :param tags: The tags of the security group.
        :return: Security group list.
        """
        request = DescribeSecurityGroupAttributeRequest()
        request.set_SecurityGroupId(security_group_id)
        response = self._send_request(request)
        if response is not None:
            security_groups = response.get("SecurityGroups").get("SecurityGroup")
            return security_groups
        logging.error("describe security group attribute failed.")
        return None

    def authorize_security_group(
        self, ip_protocol, port_range, security_group_id, source_cidr_ip
    ):
        """Create an inbound security group rule.

        :param ip_protocol: The transport layer protocol.
        :param port_range: The range of destination ports relevant to
                           the transport layer protocol.
        :param security_group_id: The ID of the destination security group.
        :param source_cidr_ip: The range of source IPv4 addresses.
                               CIDR blocks and IPv4 addresses are supported.
        """
        request = AuthorizeSecurityGroupRequest()
        request.set_IpProtocol(ip_protocol)
        request.set_PortRange(port_range)
        request.set_SecurityGroupId(security_group_id)
        request.set_SourceCidrIp(source_cidr_ip)
        self._send_request(request)

    def delete_security_group(self, security_group_id):
        """Delete security group.

                :return: The request response.
                """
        request = DeleteSecurityGroupRequest()
        request.set_SecurityGroupId(security_group_id)
        return self._send_request(request)

    def create_v_switch(self, vpc_id, zone_id, cidr_block, vswitch_name):
        """Create vSwitches to divide the VPC into one or more subnets

        :param vpc_id: The ID of the VPC to which the VSwitch belongs.
        :param zone_id: The ID of the zone to which
                        the target VSwitch belongs.
        :param cidr_block: The CIDR block of the VSwitch.
        :param vswitch_name: The name of VSwitch
        :return:
        """
        request = CreateVSwitchRequest()
        request.set_ZoneId(zone_id)
        request.set_VpcId(vpc_id)
        request.set_CidrBlock(cidr_block)
        request.set_VSwitchName(vswitch_name)
        response = self._send_request(request)
        if response is not None:
            return response.get("VSwitchId")
        else:
            logging.error("create_v_switch vpc_id %s failed.", vpc_id)
        return None

    def create_vpc_peering_connection(self, region_id, vpc_id, accepted_ali_uid, accepted_vpc_id, accepted_region_id, name):
        request = CommonRequest()
        request.set_accept_format('json')
        request.set_domain('vpcpeer.aliyuncs.com')
        request.set_method('POST')
        request.set_protocol_type('https')  # https | http
        request.set_version('2022-01-01')
        request.set_action_name('CreateVpcPeerConnection')
        request.add_query_param('RegionId', region_id)
        request.add_query_param('VpcId', vpc_id)
        request.add_query_param('AcceptingAliUid', accepted_ali_uid)
        request.add_query_param('AcceptingVpcId', accepted_vpc_id)
        request.add_query_param('AcceptingRegionId', accepted_region_id)
        request.add_query_param('Name', name)
        response = self._send_request(request)
        if response is not None:
            return response.get("InstanceId")
        return None

    def delete_vpc_peering_connection(self, instance_id):
        request = CommonRequest()
        request.set_accept_format('json')
        request.set_domain('vpcpeer.aliyuncs.com')
        request.set_method('POST')
        request.set_protocol_type('https')  # https | http
        request.set_version('2022-01-01')
        request.set_action_name('CreateVpcPeerConnection')
        request.add_query_param('InstanceId', instance_id)
        return self._send_request(request)

    def describe_vpc_peering_connections(self, vpc_id=None, vpc_peering_connection_name=None):
        """Queries VPC peering connection.

        :return: VPC peering connection list.
        """
        request = CommonRequest()
        request.set_accept_format('json')
        request.set_domain('vpcpeer.aliyuncs.com')
        request.set_method('POST')
        request.set_protocol_type('https')  # https | http
        request.set_version('2022-01-01')
        request.set_action_name('ListVpcPeerConnections')
        if vpc_peering_connection_name is  not None:
            request.add_query_param('Name', vpc_peering_connection_name)
        if vpc_id is not None:
            request.add_query_param('VpcId.1', vpc_id)
        response = self._send_request(request)
        if response is not None:
            return response.get("VpcPeerConnects")
        return None

    def describe_route_tables(self, vpc_id=None):
        request = DescribeRouteTableListRequest()
        if vpc_id is not None:
            request.set_VpcId(vpc_id)
        response = self._send_request(request)
        if response is not None:
            return response.get("RouterTableList").get("RouterTableListType")
        return None

    def create_route_entry(self, route_table_id, cidr_block, next_hop_id, next_hop_type, name):
        request = CreateRouteEntryRequest()
        request.set_RouteTableId(route_table_id)
        request.set_DestinationCidrBlock(cidr_block)
        request.set_NextHopId(next_hop_id)
        request.set_NextHopType(next_hop_type)
        request.set_RouteEntryName(name)
        return self._send_request(request)

    def describe_route_entry_list(self, route_table_id, cidr_block=None, entry_name=None):
        request = DescribeRouteEntryListRequest()
        request.set_RouteTableId(route_table_id)
        if cidr_block is not None:
            request.set_DestinationCidrBlock(cidr_block)
        if entry_name is not None:
            request.set_RouteEntryName(entry_name)
        response = self._send_request(request)
        if response is not None:
            return response.get("RouteEntrys").get("RouteEntry")
        return None

    def delete_route_entry(self, route_entry_id, route_table_id=None, cidr_block=None):
        request = DeleteRouteEntryRequest()
        request.set_RouteEntryId(route_entry_id)
        if route_table_id is None:
            request.set_RouteTableId(route_table_id)
        if cidr_block is None:
            request.set_DestinationCidrBlock(cidr_block)
        return self._send_request(request)

    def create_vpc(self, vpc_name, cidr_block):
        """Creates a virtual private cloud (VPC).

        :return: The created VPC ID.
        """
        request = CreateVpcRequest()
        request.set_VpcName(vpc_name)
        request.set_CidrBlock(cidr_block)
        response = self._send_request(request)
        if response is not None:
            return response.get("VpcId")
        return None

    def delete_vpc(self, vpc_id):
        """Delete virtual private cloud (VPC).

                :return: The request response.
                """
        request = DeleteVpcRequest()
        request.set_VpcId(vpc_id)
        return self._send_request(request)

    def describe_vpcs(self, vpc_id=None, vpc_name=None):
        """Queries one or more VPCs in a region.

        :return: VPC list.
        """
        request = DescribeVpcsRequest()
        if vpc_id is not None:
            request.set_VpcId(vpc_id)
        if vpc_name is not None:
            request.set_VpcName(vpc_name)
        response = self._send_request(request)
        if response is not None:
            return response.get("Vpcs").get("Vpc")
        return None

    def tag_vpc_resource(self, resource_id, resource_type, tags):
        """

        :param resource_id:
        :param resource_type:
        :param tags:
        :return:
        """
        request = TagResourcesRequest()
        request.set_ResourceIds(resource_id)
        request.set_ResourceType(resource_type)
        request.set_Tags(tags)
        return self._send_request(request)

    def untag_vpc_resource(self, resource_id, resource_type, tag_keys):
        """

        :param resource_id:
        :param resource_type:
        :param tag_keys:
        :return:
        """
        request = UnTagResourcesRequest()
        request.set_ResourceIds(resource_id)
        request.set_ResourceType(resource_type)
        request.set_TagKeys(tag_keys)

    def tag_ecs_resource(self, resource_ids, tags, resource_type="instance"):
        """Create and bind tags to specified ECS resources.

        :param resource_ids: The IDs of N resources.
        :param tags: The tags of the resource.
        :param resource_type: The type of the resource.
        """
        request = TagResourcesRequest()
        request.set_Tags(tags)
        request.set_ResourceType(resource_type)
        request.set_ResourceIds(resource_ids)
        response = self._send_request(request)
        if response is not None:
            logging.info("instance %s create tag successfully.", resource_ids)
        else:
            logging.error("instance %s create tag failed.", resource_ids)

    def start_instance(self, instance_id):
        """Start an ECS instance.

        :param instance_id: The Ecs instance ID.
        """
        request = StartInstanceRequest()
        request.set_InstanceId(instance_id)
        response = self._send_request(request)

        if response is not None:
            logging.info("instance %s start successfully.", instance_id)
        else:
            logging.error("instance %s start failed.", instance_id)

    def stop_instance(self, instance_id, force_stop=False):
        """Stop an ECS instance that is in the Running state.

        :param instance_id: The Ecs instance ID.
        :param force_stop: Specifies whether to forcibly stop the instance.
        :return:
        """
        request = StopInstanceRequest()
        request.set_InstanceId(instance_id)
        request.set_ForceStop(force_stop)
        logging.info("Stop %s command submit successfully.", instance_id)
        self._send_request(request)

    def stop_instances(self, instance_ids, stopped_mode="StopCharging"):
        """Stop one or more ECS instances that are in the Running state.

        :param instance_ids: The IDs of instances.
        :param stopped_mode: Specifies whether billing for the instance
                             continues after the instance is stopped.
        """
        request = StopInstancesRequest()
        request.set_InstanceIds(instance_ids)
        request.set_StoppedMode(stopped_mode)
        response = self._send_request(request)
        if response is None:
            logging.error("stop_instances failed")

    def delete_instance(self, instance_id):
        """Release a pay-as-you-go instance or
            an expired subscription instance.

        :param instance_id: The ID of the instance that you want to release.
        """
        request = DeleteInstanceRequest()
        request.set_InstanceId(instance_id)
        request.set_Force(True)
        logging.info("Delete %s command submit successfully", instance_id)
        self._send_request(request)

    def delete_instances(self, instance_ids):
        """Release one or more pay-as-you-go instances or
            expired subscription instances.

        :param instance_ids: The IDs of instances that you want to release.
        """
        request = DeleteInstancesRequest()
        request.set_Force(True)
        request.set_InstanceIds(instance_ids)
        self._send_request(request)

    def allocate_public_address(self, instance_id):
        """Assign a public IP address to an ECS instance.

        :param instance_id: The ID of the instance to which you want to
                            assign a public IP address.
        :return: The assigned ip.
        """
        request = AllocatePublicIpAddressRequest()
        request.set_InstanceId(instance_id)
        response = self._send_request(request)
        if response is not None:
            return response.get("IpAddress")

    def create_key_pair(self, key_pair_name):
        """Create an SSH key pair.

        :param key_pair_name: The name of the key pair.
        :return: The created keypair data.
        """
        request = CreateKeyPairRequest()
        request.set_KeyPairName(key_pair_name)
        response = self._send_request(request)
        if response is not None:
            logging.info("Create Key Pair %s Successfully", response.get("KeyPairId"))
            return response
        else:
            logging.error("Create Key Pair Failed")
            return None

    def import_key_pair(self, key_pair_name, public_key_body):
        """Import the public key of an RSA-encrypted key pair
            that is generated by a third-party tool.

        :param key_pair_name: The name of the key pair.
        :param public_key_body: The public key of the key pair.
        """
        request = ImportKeyPairRequest()
        request.set_KeyPairName(key_pair_name)
        request.set_PublicKeyBody(public_key_body)
        self._send_request(request)

    def delete_key_pairs(self, key_pair_names):
        """Delete one or more SSH key pairs.

        :param key_pair_names: The name of the key pair.
        :return:
        """
        request = DeleteKeyPairsRequest()
        request.set_KeyPairNames(key_pair_names)
        self._send_request(request)

    def describe_key_pairs(self, key_pair_name=None):
        """Query one or more key pairs.

        :param key_pair_name: The name of the key pair.
        :return:
        """
        request = DescribeKeyPairsRequest()
        if key_pair_name is not None:
            request.set_KeyPairName(key_pair_name)
        response = self._send_request(request)
        if response is not None:
            return response.get("KeyPairs").get("KeyPair")
        else:
            return None

    def describe_v_switches(self, vpc_id=None):
        """Queries one or more VSwitches.

        :param vpc_id: The ID of the VPC to which the VSwitch belongs.
        :return: VSwitch list.
        """
        request = DescribeVSwitchesRequest()
        if vpc_id is not None:
            request.set_VpcId(vpc_id)
        response = self._send_request(request)
        if response is not None:
            return response.get("VSwitches").get("VSwitch")
        else:
            logging.error("Describe VSwitches Failed.")
            return None

    def delete_v_switch(self, vswitch_id):
        """Delete virtual switch (VSwitch).

                :return: The request response.
                """
        request = DeleteVSwitchRequest()
        request.set_VSwitchId(vswitch_id)
        return self._send_request(request)

    def describe_zones(self):
        """Queries all available zones in a region.

        :return: Zone list.
        """
        request = DescribeZonesRequest()
        request.set_AcceptLanguage("en-US")
        response = self._send_request(request)
        if response is not None:
            return response.get("Zones").get("Zone")
        else:
            logging.error("Describe Zones Failed.")
            return None

    def describe_nat_gateways(self, vpc_id):
        """Queries all available nat-gateway.
        :return: Zone list.
        """
        request = DescribeNatGatewaysRequest()
        request.set_VpcId(vpc_id)
        response = self._send_request(request)
        if response is not None:
            return response.get("NatGateways").get("NatGateway")
        else:
            logging.error("Describe NatGateways Failed.")
            return None

    def delete_nat_gateway(self, nat_gateway_id):
        """Delete Nat Gateway.
        :return: The request response.
        """
        request = DeleteNatGatewayRequest()
        request.set_NatGatewayId(nat_gateway_id)
        return self._send_request(request)

    def create_nat_gateway(self, vpc_id, vswitch_id, nat_gateway_name):
        """Create Nat Gateway.
        :return: The Nat Gateway Id.
        """
        request = CreateNatGatewayRequest()
        request.set_VpcId(vpc_id)
        request.set_Name(nat_gateway_name)
        request.set_NatType("Enhanced")
        request.set_VSwitchId(vswitch_id)
        response = self._send_request(request)
        if response is not None:
            return response.get("NatGatewayId")
        return None

    def allocate_eip_address(self, eip_name):
        """Allocate elastic ip address
        :return allocation_id:
        """
        request = AllocateEipAddressRequest()
        request.set_Name(eip_name)
        response = self._send_request(request)
        if response is not None:
            return response.get("AllocationId")
        return None

    def associate_eip_address(self, eip_allocation_id, instance_type, instance_id):
        """Bind elastic ip address to cloud instance
        :return The request response:
        """
        request = AssociateEipAddressRequest()
        request.set_AllocationId(eip_allocation_id)
        request.set_InstanceType(instance_type)
        request.set_InstanceId(instance_id)
        return self._send_request(request)

    def describe_eip_addresses(self):
        """Queries all available eip.

        :return eips:
        """
        request = DescribeEipAddressesRequest()
        response = self._send_request(request)
        if response is not None:
            return response.get("EipAddresses").get("EipAddress")
        else:
            logging.error("Describe EIP Failed.")
            return None

    def dissociate_eip_address(self, eip_allocation_id, instance_type, instance_id):
        """

        :param eip_allocation_id:
        :param instance_type:
        :param instance_id:
        :return:
        """
        request = UnassociateEipAddressRequest()
        request.set_AllocationId(eip_allocation_id)
        request.set_InstanceType(instance_type)
        request.set_InstanceId(instance_id)
        return self._send_request(request)

    def release_eip_address(self, eip_allocation_id):
        """
        Release EIP resource
        :param eip_allocation_id:
        :return:
        """
        request = ReleaseEipAddressRequest()
        request.set_AllocationId(eip_allocation_id)
        return self._send_request(request)

    def create_snat_entry(self, snat_table_id, vswitch_id, snat_ip, snat_entry_name):
        """

        :param snat_table_id:
        :param vswitch_id:
        :param snat_ip:
        :param snat_entry_name:
        :return:
        """
        request = CreateSnatEntryRequest()
        request.set_SnatTableId(snat_table_id)
        request.set_SourceVSwitchId(vswitch_id)
        request.set_SnatIp(snat_ip)
        request.set_SnatEntryName(snat_entry_name)
        response = self._send_request(request)
        if response is not None:
            return response.get("SnatEntryId")
        else:
            logging.error("Failed to create SNAT Entry.")
            return None

    def describe_snat_entries(self, snat_table_id):
        """

        :param snat_table_id:
        :return:
        """
        request = DescribeSnatTableEntriesRequest()
        request.set_SnatTableId(snat_table_id)
        response = self._send_request(request)
        if response is not None:
            return response.get("SnatTableEntries").get("SnatTableEntry")
        else:
            logging.error("Failed to describe SNAT Entries.")
            return None

    def delete_snat_entry(self, snat_table_id, snat_entry_id):
        """
        :param snat_table_id:
        :param snap_entry_id:
        :return:
        """
        request = DeleteSnatEntryRequest()
        request.set_SnatTableId(snat_table_id)
        request.set_SnatEntryId(snat_entry_id)
        return self._send_request(request)

    def create_role(self, role_name, assume_role_policy_document):
        """

        :param role_name:
        :param assume_role_policy_document:
        :return:
        """
        request = CreateRoleRequest()
        request.set_RoleName(role_name)
        request.set_AssumeRolePolicyDocument(assume_role_policy_document)
        response = self._send_request(request)
        if response is not None:
            return response.get("Role")
        else:
            logging.error("Failed to create Role.")
            return None

    def get_role(self, role_name):
        """

        :param role_name:
        :return:
        """
        request = GetRoleRequest()
        request.set_RoleName(role_name)
        response = self._send_request(request)
        if response is not None:
            return response.get("Role")
        else:
            logging.error("Failed to get Role.")
            return None

    def delete_role(self, role_name):
        """

        :param role_name:
        :return:
        """
        request = DeleteRoleRequest()
        request.set_RoleName(role_name)
        return self._send_request(request)

    def attach_policy_to_role(self, role_name, policy_type, policy_name):
        """

        :param role_name:
        :param policy_type:
        :param policy_name:
        :return:
        """
        request = AttachPolicyToRoleRequest()
        request.set_RoleName(role_name)
        request.set_PolicyType(policy_type)
        request.set_PolicyName(policy_name)
        return self._send_request(request)

    def detach_policy_from_role(self, role_name, policy_type, policy_name):
        """

        :param role_name:
        :param policy_type:
        :param policy_name:
        :return:
        """
        request = DetachPolicyFromRoleRequest()
        request.set_RoleName(role_name)
        request.set_PolicyType(policy_type)
        request.set_PolicyName(policy_name)
        return self._send_request(request)

    def list_policy_for_role(self, role_name):
        request = ListPoliciesForRoleRequest()
        request.set_RoleName(role_name)
        response = self._send_request(request)
        if response is not None:
            return response.get("Policies").get("Policy")
        else:
            logging.error("Failed to list policy for role.")
            return None

    def _send_request(self, request):
        """send open api request"""
        request.set_accept_format("json")
        try:
            response_str = self.cli.do_action_with_exception(request)
            response_detail = json.loads(response_str)
            return response_detail
        except (ClientException, ServerException) as e:
            logging.error(request.get_action_name())
            logging.error(e)
            return None


def get_aliyun_oss_storage_config(provider_config: Dict[str, Any]):
    if "storage" in provider_config and "aliyun_oss_storage" in provider_config["storage"]:
        return provider_config["storage"]["aliyun_oss_storage"]

    return None


def get_aliyun_oss_storage_config_for_update(provider_config: Dict[str, Any]):
    storage_config = get_storage_config_for_update(provider_config)
    if "aliyun_oss_storage" not in storage_config:
        storage_config["aliyun_oss_storage"] = {}
    return storage_config["aliyun_oss_storage"]


def export_aliyun_oss_storage_config(provider_config, config_dict: Dict[str, Any]):
    cloud_storage = get_aliyun_oss_storage_config(provider_config)
    if cloud_storage is None:
        return
    config_dict["ALIYUN_CLOUD_STORAGE"] = True

    oss_bucket = cloud_storage.get(ALIYUN_OSS_BUCKET)
    if oss_bucket:
        config_dict["ALIYUN_OSS_BUCKET"] = oss_bucket

    oss_access_key_id = cloud_storage.get("oss.access.key.id")
    if oss_access_key_id:
        config_dict["ALIYUN_OSS_ACCESS_KEY_ID"] = oss_access_key_id

    oss_access_key_secret = cloud_storage.get("oss.access.key.secret")
    if oss_access_key_secret:
        config_dict["ALIYUN_OSS_ACCESS_KEY_SECRET"] = oss_access_key_secret


def get_aliyun_cloud_storage_uri(aliyun_cloud_storage):
    oss_bucket = aliyun_cloud_storage.get(ALIYUN_OSS_BUCKET)
    if oss_bucket is None:
        return None

    return "oss://{}".format(oss_bucket)


def get_default_aliyun_cloud_storage(provider_config):
    cloud_storage = get_aliyun_oss_storage_config(provider_config)
    if cloud_storage is None:
        return None

    cloud_storage_info = {}
    cloud_storage_info.update(cloud_storage)

    cloud_storage_uri = get_aliyun_cloud_storage_uri(cloud_storage)
    if cloud_storage_uri:
        cloud_storage_info[CLOUDTIK_DEFAULT_CLOUD_STORAGE_URI] = cloud_storage_uri

    return cloud_storage_info


def tags_list_to_dict(tags: list):
    tags_dict = {}
    for item in tags:
        tags_dict[item.tag_key] = item.tag_value
    return tags_dict


def _get_node_info(node):
    private_ip = None
    if (node.vpc_attributes is not None
            and node.vpc_attributes.private_ip_address is not None
            and len(node.vpc_attributes.private_ip_address) > 0):
        private_ip = node.vpc_attributes.private_ip_address[0]
    public_ip = None
    if (node.public_ip_address is not None
        and node.public_ip_address.ip_address is not None
            and len(node.public_ip_address.ip_address) > 0):
        public_ip = node.public_ip_address.ip_address[0]
    node_info = {"node_id": node.instance_id,
                 "instance_type": node.instance_type,
                 "private_ip": private_ip,
                 "public_ip": public_ip,
                 "instance_status": node.status}
    if (node.tags is not None and
            node.tags.tag is not None):
        node_info.update(tags_list_to_dict(node.tags.tag))
    return node_info


def get_credential(provider_config):
    aliyun_credentials = provider_config.get("aliyun_credentials")
    return _get_credential(aliyun_credentials)


def _get_credential(aliyun_credentials=None):
    if aliyun_credentials is not None:
        ak = aliyun_credentials.get("aliyun_access_key_id")
        sk = aliyun_credentials.get("aliyun_access_key_secret")
        credential_config = Config(
            type='access_key',  # credential type
            access_key_id=ak,  # AccessKeyId
            access_key_secret=sk,  # AccessKeySecret
        )
        credential = CredentialClient(credential_config)
    else:
        credential = CredentialClient()

    return credential


def make_vpc_client(provider_config):
    credential = get_credential(provider_config)
    config = open_api_models.Config(credential=credential)
    config.endpoint = f'vpc.aliyuncs.com'
    return vpc_client(config)


def make_vpc_peer_client(provider_config):
    credential = get_credential(provider_config)
    config = open_api_models.Config(credential=credential)
    config.endpoint = f'vpcpeer.aliyuncs.com'
    return vpc_peer_client(config)


def make_ecs_client(provider_config, region_id=None):
    region_id = region_id if region_id is not None else provider_config["region"]
    credential = get_credential(provider_config)
    config = open_api_models.Config(credential=credential)
    config.endpoint = f'ecs.{region_id}.aliyuncs.com'
    return ecs_client(config)


def make_ram_client(provider_config):
    credential = get_credential(provider_config)
    config = open_api_models.Config(credential=credential)
    config.endpoint = f'ram.aliyuncs.com'
    return ram_client(config)


def make_oss_client(provider_config, region_id=None):
    credentials_config = provider_config.get("aliyun_credentials")
    region_id = region_id if region_id is not None else provider_config["region"]
    return _make_oss_client(credentials_config, region_id)


def _make_oss_client(credentials_config, region_id):
    credential = _get_credential(credentials_config)
    config = open_api_models.Config(credential=credential)
    config.endpoint = f"oss-{region_id}.aliyuncs.com"
    return oss_client(config)


def check_resource_status(time_default_out, default_time, describe_attribute_func, check_status, resource_id):
    for i in range(time_default_out):
        time.sleep(default_time)
        try:
            resource_attributes = describe_attribute_func(resource_id)
            if isinstance(resource_attributes, list):
                status = "" if len(resource_attributes) == 0 else resource_attributes[0].to_map().get("Status", "")
            else:
                status = resource_attributes.to_map().get("Status", "")
            if status == check_status:
                return True
        except UnretryableException as e:
            cli_logger.error("Failed to get attributes of resource. {}".format(format_exception_message(str(e))))
            continue
    return False


class OssClient:
    """
    A wrapper around Aliyun OSS client.
    Parameters:
        provider_config: The cloud provider configuration from which to create client.
    """

    def __init__(self, provider_config, credentials_config=None, region_id=None):
        self.region_id = provider_config["region"] if region_id is None else region_id
        if credentials_config is None:
            credentials_config = provider_config.get("aliyun_credentials")
        self.client = _make_oss_client(credentials_config, self.region_id)
        self.runtime_options = util_models.RuntimeOptions()

    def put_bucket(self, name):
        put_bucket_request = oss_models.PutBucketRequest(oss_models.CreateBucketConfiguration())
        try:
            response = self.client.put_bucket(name, put_bucket_request)
            # return response
        except TeaException as e:
            cli_logger.error("Failed to create bucket. {}".format(format_exception_message(str(e))))
            raise e
        except Exception as e:
            cli_logger.error("Ignore the exception. {}".format(format_exception_message(str(e))))

    def delete_bucket(self, name):
        try:
            self.client.delete_bucket(name)
        except Exception as e:
            cli_logger.error("Failed to delete bucket. {}".format(format_exception_message(str(e))))
            raise e
    
    def list_buckets(self):
        list_bucket_request = oss_models.ListBucketsRequest()
        try:
            response = self.client.list_buckets(list_bucket_request)
            return response.body.buckets.buckets
        except Exception as e:
            cli_logger.error("Failed to list buckets. {}".format(format_exception_message(str(e))))
            raise e
        
    def list_objects(self, bucket_name):
        list_objects_request = oss_models.ListObjectsRequest()
        try:
            response = self.client.list_objects(bucket_name, list_objects_request)
            return response.body.contents
        except Exception as e:
            cli_logger.error("Failed to list objects of the bucket: {}. {}".
                             format(bucket_name, format_exception_message(str(e))))
            raise e

    def delete_object(self, bucket_name, object_key):
        delete_object_request = oss_models.DeleteObjectRequest()
        try:
            self.client.delete_object(bucket_name, object_key, delete_object_request)
        except Exception as e:
            cli_logger.error("Failed to delete the object: {} of the bucket: {}. ".
                             format(object_key, bucket_name, format_exception_message(str(e))))
            raise e


class VpcClient:
    """
    A wrapper around Aliyun VPC client.
    Parameters:
        provider_config: The cloud provider configuration from which to create client.
    """

    def __init__(self, provider_config, region_id=None):
        self.region_id = provider_config["region"] if region_id is None else region_id
        self.client = make_vpc_client(provider_config)
        self.runtime_options = util_models.RuntimeOptions()

    def describe_vpcs(self, vpc_id=None, vpc_name=None):
        """Queries one or more VPCs in a region.
        :return: VPC list.
        """
        describe_vpcs_request = vpc_models.DescribeVpcsRequest(
            region_id=self.region_id,
            vpc_id=vpc_id,
            vpc_name=vpc_name
        )
        try:
            response = self.client.describe_vpcs_with_options(
                describe_vpcs_request, self.runtime_options)
            return response.body.vpcs.vpc
        except Exception as e:
            cli_logger.error("Failed to describe VPCs. {}".format(format_exception_message(str(e))))
            raise e

    def describe_vpc_attribute(self, vpc_id):
        """Queries attribute of the VPC in a region.
        :return: VPC attribute.
        """
        describe_vpc_attribute_request = vpc_models.DescribeVpcAttributeRequest(
            region_id=self.region_id,
            vpc_id=vpc_id
        )
        try:
            response = self.client.describe_vpc_attribute_with_options(
                describe_vpc_attribute_request, self.runtime_options)
            return response.body
        except Exception as e:
            cli_logger.error("Failed to get the attribute of the VPC. {}".format(format_exception_message(str(e))))
            raise e

    def create_vpc(self, vpc_name, cidr_block):
        """Creates a virtual private cloud (VPC).
        :return: The created VPC ID.
        """
        create_vpc_request = vpc_models.CreateVpcRequest(
            region_id=self.region_id,
            cidr_block=cidr_block,
            vpc_name=vpc_name
        )
        try:
            response = self.client.create_vpc_with_options(
                create_vpc_request, self.runtime_options)
            return response.body.vpc_id
        except Exception as e:
            cli_logger.error("Failed to create VPC. {}".format(format_exception_message(str(e))))
            raise e

    def delete_vpc(self, vpc_id):
        """Delete virtual private cloud (VPC)."""
        delete_vpc_request = vpc_models.DeleteVpcRequest(
            vpc_id=vpc_id
        )
        try:
            self.client.delete_vpc_with_options(
                delete_vpc_request, self.runtime_options)
        except Exception as e:
            cli_logger.error("Failed to delete VPC. {}".format(format_exception_message(str(e))))
            raise e

    def tag_resource(self, resource_id, tags, resource_type="VPC"):
        """Create and bind tags to specified VPC resource.
        :param resource_id: The ID of resource.
        :param tags: The tags of the resource.
        :param resource_type: The type of the resource.
        """
        request_tags = [vpc_models.TagResourcesRequestTag(
            key=tag["Key"],
            value=tag["Value"]
        ) for tag in tags] if tags else None
        tag_resources_request = vpc_models.TagResourcesRequest(
            resource_type=resource_type,
            resource_id=[resource_id],
            region_id=self.region_id,
            tag=request_tags
        )
        try:
            self.client.tag_resources_with_options(
                tag_resources_request, self.runtime_options)
        except Exception as e:
            cli_logger.error("Failed to tag VPC. {}".format(format_exception_message(str(e))))
            raise e

    def untag_resource(self, resource_id, tag_keys, resource_type="VPC"):
        """Untag from specified VPC resource"""
        un_tag_resources_request = vpc_models.UnTagResourcesRequest(
            resource_type=resource_type,
            resource_id=[resource_id],
            tag_key=tag_keys,
            region_id=self.region_id
        )
        try:
            self.client.un_tag_resources_with_options(
                un_tag_resources_request, self.runtime_options)
        except Exception as e:
            cli_logger.error("Failed to untag VPC. {}".format(format_exception_message(str(e))))
            raise e

    def describe_zones(self):
        """Queries all available zones in a region.
        :return: Zone list.
        """
        describe_zones_request = vpc_models.DescribeZonesRequest(
            region_id=self.region_id,
            accept_language='en-us'
        )
        try:
            response = self.client.describe_zones_with_options(
                describe_zones_request, self.runtime_options)
            return response.body.zones.zone
        except Exception as e:
            cli_logger.error("Failed to describe zones. {}".format(format_exception_message(str(e))))
            raise e

    def list_enhanced_nat_gateway_available_zones(self):
        list_enhanhced_nat_gateway_available_zones_request = vpc_models.ListEnhanhcedNatGatewayAvailableZonesRequest(
            region_id=self.region_id,
            accept_language='en-us'
        )
        try:
            response = self.client.list_enhanhced_nat_gateway_available_zones_with_options(
                list_enhanhced_nat_gateway_available_zones_request, self.runtime_options)
            return response.body.zones
        except Exception as e:
            cli_logger.error("Failed to list enhanced nat gate-way available zones. {}".
                             format(format_exception_message(str(e))))
            raise e

    def describe_vswitch_attributes(self, vswitch_id):
        describe_vswitch_attributes_request = vpc_models.DescribeVSwitchAttributesRequest(
            region_id=self.region_id,
            v_switch_id=vswitch_id
        )
        try:
            response = self.client.describe_vswitch_attributes_with_options(
                describe_vswitch_attributes_request, self.runtime_options)
            return response.body
        except Exception as e:
            cli_logger.error("Failed to describe the attributes of the vswitch. {}".
                             format(format_exception_message(str(e))))
            raise e

    def describe_vswitches(self, vpc_id=None):
        """Queries one or more VSwitches.
        :param vpc_id: The ID of the VPC to which the VSwitch belongs.
        :return: VSwitch list.
        """
        describe_vswitches_request = vpc_models.DescribeVSwitchesRequest(
            vpc_id=vpc_id
        )
        try:
            response = self.client.describe_vswitches_with_options(
                describe_vswitches_request, self.runtime_options)
            return response.body.v_switches.v_switch
        except Exception as e:
            cli_logger.error("Failed to describe vswitches. {}".format(format_exception_message(str(e))))
            raise e

    def delete_vswitch(self, vswitch_id):
        """Delete virtual switch (VSwitch)."""
        delete_vswitch_request = vpc_models.DeleteVSwitchRequest(
            v_switch_id=vswitch_id
        )
        try:
            self.client.delete_vswitch_with_options(
                delete_vswitch_request, self.runtime_options)
        except Exception as e:
            cli_logger.error("Failed to delete vswitch. {}".format(format_exception_message(str(e))))
            raise e

    def create_vswitch(self, vpc_id, zone_id, cidr_block, vswitch_name):
        """Create vSwitches to divide the VPC into one or more subnets
        :param vpc_id: The ID of the VPC to which the VSwitch belongs.
        :param zone_id: The ID of the zone to which
                        the target VSwitch belongs.
        :param cidr_block: The CIDR block of the VSwitch.
        :param vswitch_name: The name of VSwitch
        :return:
        """
        create_vswitch_request = vpc_models.CreateVSwitchRequest(
            vpc_id=vpc_id,
            zone_id=zone_id,
            cidr_block=cidr_block,
            v_switch_name=vswitch_name
        )
        try:
            response = self.client.create_vswitch_with_options(
                create_vswitch_request, self.runtime_options)
            return response.body.v_switch_id
        except Exception as e:
            cli_logger.error("Failed to create vswitch. {}".format(format_exception_message(str(e))))
            raise e
    
    def describe_route_tables(self, vpc_id=None):
        describe_route_table_list_request = vpc_models.DescribeRouteTableListRequest(
            region_id=self.region_id,
            vpc_id=vpc_id
        )
        try:
            response = self.client.describe_route_table_list_with_options(
                describe_route_table_list_request, self.runtime_options)
            return response.body.router_table_list.router_table_list_type
        except Exception as e:
            cli_logger.error("Failed to describe route tables. {}".format(format_exception_message(str(e))))
            raise e
    
    def describe_nat_gateways(self, vpc_id, name):
        """Queries all available nat-gateway.
        :return: nat-gateway list.
        """
        describe_nat_gateways_request = vpc_models.DescribeNatGatewaysRequest(
            region_id=self.region_id,
            vpc_id=vpc_id,
            name=name
        )
        try:
            response = self.client.describe_nat_gateways_with_options(
                describe_nat_gateways_request, self.runtime_options)
            return response.body.nat_gateways.nat_gateway
        except Exception as e:
            cli_logger.error("Failed to describe nat-gateways. {}".
                             format(format_exception_message(str(e))))
            raise e

    def get_nat_gateway_attribute(self, nat_gateway_id):
        get_nat_gateway_attribute_request = vpc_models.GetNatGatewayAttributeRequest(
            region_id=self.region_id,
            nat_gateway_id=nat_gateway_id
        )
        try:
            response = self.client.get_nat_gateway_attribute_with_options(
                get_nat_gateway_attribute_request, self.runtime_options)
            return response.body
        except Exception as e:
            cli_logger.error("Failed to get the attribute of the nat-gateway. {}".
                             format(format_exception_message(str(e))))
            raise e

    def delete_nat_gateway(self, nat_gateway_id):
        """Delete Nat Gateway.
        :return: The request response.
        """
        delete_nat_gateway_request = vpc_models.DeleteNatGatewayRequest(
            region_id=self.region_id,
            nat_gateway_id=nat_gateway_id
        )
        try:
            self.client.delete_nat_gateway_with_options(
                delete_nat_gateway_request, self.runtime_options)
        except Exception as e:
            cli_logger.error("Failed to delete nat-gateway. {}".format(format_exception_message(str(e))))
            raise e

    def create_nat_gateway(self, vpc_id, vswitch_id, nat_gateway_name):
        """Create Nat Gateway.
        :return: The Nat Gateway Id.
        """
        create_nat_gateway_request = vpc_models.CreateNatGatewayRequest(
            region_id=self.region_id,
            vpc_id=vpc_id,
            v_switch_id=vswitch_id,
            name=nat_gateway_name,
            nat_type="Enhanced"
        )
        try:
            response = self.client.create_nat_gateway_with_options(
                create_nat_gateway_request, self.runtime_options)
            return response.body.nat_gateway_id
        except Exception as e:
            cli_logger.error("Failed to create nat-gateway. {}".format(format_exception_message(str(e))))
            raise e

    def allocate_eip_address(self, name):
        """Allocate elastic ip address
        :return allocation_id:
        """
        allocate_eip_address_request = vpc_models.AllocateEipAddressRequest(
            region_id=self.region_id,
            name=name
        )
        try:
            response = self.client.allocate_eip_address_with_options(
                allocate_eip_address_request, self.runtime_options)
            return response.body.allocation_id
        except Exception as e:
            cli_logger.error("Failed to allocate EIP. {}".format(format_exception_message(str(e))))
            raise e

    def associate_eip_address(self, eip_allocation_id, instance_id, instance_type):
        """Bind elastic ip address to cloud instance"""
        associate_eip_address_request = vpc_models.AssociateEipAddressRequest(
            region_id=self.region_id,
            allocation_id=eip_allocation_id,
            instance_id=instance_id,
            instance_type=instance_type
        )
        try:
            self.client.associate_eip_address_with_options(
                associate_eip_address_request, self.runtime_options)
        except Exception as e:
            cli_logger.error("Failed to associate EIP to instance. {}".format(format_exception_message(str(e))))
            raise e

    def describe_eip_addresses(self, eip_allocation_id=None, eip_name=None):
        """Queries all available eip.
        :return eips:
        """
        describe_eip_addresses_request = vpc_models.DescribeEipAddressesRequest(
            region_id=self.region_id,
            eip_name=eip_name,
            allocation_id=eip_allocation_id
        )
        try:
            response = self.client.describe_eip_addresses_with_options(
                describe_eip_addresses_request, self.runtime_options)
            return response.body.eip_addresses.eip_address
        except Exception as e:
            cli_logger.error("Failed to describe EIP addresses. {}".format(format_exception_message(str(e))))
            raise e

    def unassociate_eip_address(self, eip_allocation_id, instance_id, instance_type):
        """Dissociate eip address from instance"""
        unassociate_eip_address_request = vpc_models.UnassociateEipAddressRequest(
            allocation_id=eip_allocation_id,
            instance_id=instance_id,
            instance_type=instance_type
        )
        try:
            self.client.unassociate_eip_address_with_options(
                unassociate_eip_address_request, self.runtime_options)
        except Exception as e:
            cli_logger.error("Failed to unassociate EIP address from instance. {}".format(format_exception_message(str(e))))
            raise e

    def release_eip_address(self, eip_allocation_id):
        """Release EIP resource"""
        release_eip_address_request = vpc_models.ReleaseEipAddressRequest(
            allocation_id=eip_allocation_id
        )
        try:
            self.client.release_eip_address_with_options(
                release_eip_address_request, self.runtime_options)
        except Exception as e:
            cli_logger.error("Failed to release EIP address. {}".format(format_exception_message(str(e))))
            raise e

    def create_snat_entry(self, snat_table_id, vswitch_id, snat_ip, snat_entry_name):
        """Create snat entry for nat-gateway"""
        create_snat_entry_request = vpc_models.CreateSnatEntryRequest(
            region_id=self.region_id,
            source_vswitch_id=vswitch_id,
            snat_table_id=snat_table_id,
            snat_ip=snat_ip,
            snat_entry_name=snat_entry_name
        )
        try:
            response = self.client.create_snat_entry_with_options(
                create_snat_entry_request, self.runtime_options)
            return response.body.snat_entry_id
        except Exception as e:
            cli_logger.error("Failed to create SNAT Entry. {}".format(format_exception_message(str(e))))
            raise e

    def describe_snat_entries(self, snat_table_id=None, snat_entry_id=None):
        """Describe SNAT Entries for snat table"""
        describe_snat_table_entries_request = vpc_models.DescribeSnatTableEntriesRequest(
            region_id=self.region_id,
            snat_table_id=snat_table_id,
            snat_entry_id=snat_entry_id
        )
        try:
            response = self.client.describe_snat_table_entries_with_options(
                describe_snat_table_entries_request, self.runtime_options)
            return response.body.snat_table_entries.snat_table_entry
        except Exception as e:
            cli_logger.error("Failed to describe SNAT Entries. {}".format(format_exception_message(str(e))))
            raise e

    def delete_snat_entry(self, snat_table_id, snat_entry_id):
        """Delete snat entry"""
        delete_snat_entry_request = vpc_models.DeleteSnatEntryRequest(
            region_id=self.region_id,
            snat_table_id=snat_table_id,
            snat_entry_id=snat_entry_id
        )
        try:
            self.client.delete_snat_entry_with_options(
                delete_snat_entry_request, self.runtime_options)
        except Exception as e:
            cli_logger.error("Failed to delete SNAT Entry. {}".format(format_exception_message(str(e))))
            raise e
    
    def create_route_entry(self, route_table_id, cidr_block, next_hop_id, next_hop_type, name):
        create_route_entry_request = vpc_models.CreateRouteEntryRequest(
            route_table_id=route_table_id,
            destination_cidr_block=cidr_block,
            next_hop_id=next_hop_id,
            route_entry_name=name,
            next_hop_type=next_hop_type
        )
        try:
            self.client.create_route_entry_with_options(
                create_route_entry_request, self.runtime_options)
        except Exception as e:
            cli_logger.error("Failed to create route entry. {}".format(format_exception_message(str(e))))
            raise e

    def describe_route_entry_list(self, route_table_id, cidr_block=None, entry_name=None):
        describe_route_entry_list_request = vpc_models.DescribeRouteEntryListRequest(
            region_id=self.region_id,
            route_table_id=route_table_id,
            destination_cidr_block=cidr_block,
            route_entry_name=entry_name
        )
        try:
            response = self.client.describe_route_entry_list_with_options(
                describe_route_entry_list_request, self.runtime_options)
            return response.body.route_entrys.route_entry
        except Exception as e:
            cli_logger.error("Failed to describe route entries. {}".format(format_exception_message(str(e))))
            raise e

    def delete_route_entry(self, route_entry_id, route_table_id=None, cidr_block=None):
        delete_route_entry_request = vpc_models.DeleteRouteEntryRequest(
            route_entry_id=route_entry_id,
            route_table_id=route_table_id,
            destination_cidr_block=cidr_block
        )
        try:
            self.client.delete_route_entry_with_options(
                delete_route_entry_request, self.runtime_options)
        except Exception as e:
            cli_logger.error("Failed to delete route entry. {}".format(format_exception_message(str(e))))
            raise e


class VpcPeerClient:
    """
    A wrapper around Aliyun VPC Peer client.
    Parameters:
        provider_config: The cloud provider configuration from which to create client.
    """

    def __init__(self, provider_config, region_id=None):
        self.region_id = provider_config["region"] if region_id is None else region_id
        self.client = make_vpc_peer_client(provider_config)
        self.runtime_options = util_models.RuntimeOptions()

    def create_vpc_peer_connection(
            self, region_id, vpc_id, accepted_ali_uid, accepted_vpc_id, accepted_region_id, name):
        create_vpc_peer_connection_request = vpc_peer_models.CreateVpcPeerConnectionRequest(
            region_id=region_id,
            vpc_id=vpc_id,
            accepting_ali_uid=accepted_ali_uid,
            accepting_region_id=accepted_region_id,
            accepting_vpc_id=accepted_vpc_id,
            name=name
        )
        try:
            response = self.client.create_vpc_peer_connection_with_options(
                create_vpc_peer_connection_request, self.runtime_options)
            return response.body.instance_id
        except Exception as e:
            cli_logger.error("Failed to create vpc peer connection. {}".format(format_exception_message(str(e))))
            raise e

    def delete_vpc_peer_connection(self, instance_id):
        delete_vpc_peer_connection_request = vpc_peer_models.DeleteVpcPeerConnectionRequest(
            instance_id=instance_id
        )
        try:
            self.client.delete_vpc_peer_connection_with_options(
                delete_vpc_peer_connection_request, self.runtime_options)
        except Exception as e:
            cli_logger.error("Failed to delete vpc peer connection. {}".format(format_exception_message(str(e))))
            raise e

    def describe_vpc_peer_connections(self, vpc_id=None, vpc_peer_connection_name=None):
        """Queries VPC peering connection.
        :return: VPC peering connection list.
        """
        list_vpc_peer_connections_request = vpc_peer_models.ListVpcPeerConnectionsRequest(
            region_id=self.region_id,
            name=vpc_peer_connection_name,
            vpc_id=[vpc_id]
        )
        try:
            response = self.client.list_vpc_peer_connections_with_options(
                list_vpc_peer_connections_request, self.runtime_options)
            return response.body.vpc_peer_connects
        except Exception as e:
            cli_logger.error("Failed to describe vpc peer connections. {}".format(format_exception_message(str(e))))
            raise e


class RamClient:
    """
    A wrapper around Aliyun RAM client.
    Parameters:
        provider_config: The cloud provider configuration from which to create client.
    """

    def __init__(self, provider_config):
        self.region_id = provider_config["region"]
        self.client = make_ram_client(provider_config)
        self.runtime_options = util_models.RuntimeOptions()

    def create_role(self, role_name, assume_role_policy_document):
        """Create RAM role"""
        create_role_request = ram_models.CreateRoleRequest(
            role_name=role_name,
            assume_role_policy_document=assume_role_policy_document
        )
        try:
            response = self.client.create_role_with_options(
                create_role_request, self.runtime_options)
            return response.body.role
        except Exception as e:
            cli_logger.error("Failed to create RAM role. {}".format(format_exception_message(str(e))))
            raise e

    def get_role(self, role_name):
        """get RAM role"""
        get_role_request = ram_models.GetRoleRequest(
            role_name=role_name
        )
        try:
            response = self.client.get_role_with_options(
                get_role_request, self.runtime_options)
            return response.body.role
        except Exception as e:
            cli_logger.error("Failed to get RAM role. {}".format(format_exception_message(str(e))))
            return None

    def delete_role(self, role_name):
        """Delete RAM role"""
        delete_role_request = ram_models.DeleteRoleRequest(
            role_name=role_name
        )
        try:
            self.client.delete_role_with_options(
                delete_role_request, self.runtime_options)
        except Exception as e:
            cli_logger.error("Failed to delete RAM role. {}".format(format_exception_message(str(e))))
            raise e

    def attach_policy_to_role(self, role_name, policy_type, policy_name):
        """Attach policy to RAM role"""
        attach_policy_to_role_request = ram_models.AttachPolicyToRoleRequest(
            policy_type=policy_type,
            policy_name=policy_name,
            role_name=role_name
        )
        try:
            self.client.attach_policy_to_role_with_options(
                attach_policy_to_role_request, self.runtime_options)
        except Exception as e:
            cli_logger.error("Failed to attach the policy to RAM role. {}".format(format_exception_message(str(e))))
            raise e

    def detach_policy_from_role(self, role_name, policy_type, policy_name):
        """Detach the policy from RAM role"""
        detach_policy_from_role_request = ram_models.DetachPolicyFromRoleRequest(
            policy_type=policy_type,
            policy_name=policy_name,
            role_name=role_name
        )
        try:
            self.client.detach_policy_from_role_with_options(
                detach_policy_from_role_request, self.runtime_options)
        except Exception as e:
            cli_logger.error("Failed to detach the policy from RAM role. {}".format(format_exception_message(str(e))))
            raise e

    def list_policy_for_role(self, role_name):
        """List the policies for RAM role"""
        list_policies_for_role_request = ram_models.ListPoliciesForRoleRequest(
            role_name=role_name
        )
        try:
            response = self.client.list_policies_for_role_with_options(
                list_policies_for_role_request, self.runtime_options)
            return response.body.policies.policy
        except Exception as e:
            cli_logger.error("Failed to list the policies for RAM role. {}".format(format_exception_message(str(e))))
            raise e


class EcsClient:
    """
    A wrapper around Aliyun ECS client.

    Parameters:
        provider_config: The cloud provider configuration from which to create client.
    """

    def __init__(self, provider_config, region_id=None):
        self.region_id = provider_config["region"] if region_id is None else region_id
        self.client = make_ecs_client(provider_config, region_id)
        self.runtime_options = util_models.RuntimeOptions()

    @staticmethod
    def _get_request_instance_ids(instance_ids):
        if not instance_ids:
            return None
        return "[" + ",".join(['"' + instance_id + '"' for instance_id in instance_ids]) + "]"

    @staticmethod
    def _merge_tags(tags: List[Dict[str, Any]],
                    user_tags: List[Dict[str, Any]]) -> None:
        """
        Merges user-provided node config tag specifications into a base
        list of node provider tag specifications. The base list of
        node provider tag specs is modified in-place.

        Args:
            tags (List[Dict[str, Any]]): base node provider tag specs
            user_tags (List[Dict[str, Any]]): user's node config tag specs
        """
        for user_tag in user_tags:
            exists = False
            for tag in tags:
                if user_tag["Key"] == tag["Key"]:
                    exists = True
                    tag["Value"] = user_tag["Value"]
                    break
            if not exists:
                tags += [user_tag]

    def describe_instance_types(self, next_token: str = None):
        """Query the details of instance types
        :return: ECS instance type list
        """
        describe_instance_types_request = ecs_models.DescribeInstanceTypesRequest(
            next_token=next_token
        )
        response = self.client.describe_instance_types_with_options(
            describe_instance_types_request, self.runtime_options)
        if response is not None:
            return response.body
        return None

    def describe_images(self, image_family):
        """List the images available
        :return: The list of images matched
        """
        describe_images_request = ecs_models.DescribeImagesRequest(
            region_id=self.region_id,
            architecture='x86_64',
            ostype='linux',
            status='Available',
            image_family=image_family
        )
        response = self.client.describe_images_with_options(
            describe_images_request, self.runtime_options)
        if (response is not None
                and response.body is not None
                and response.body.images is not None):
            return response.body.images.image
        return None

    def describe_launch_template_versions(self, query_params):
        """Query the details of launch template
        :return: The launch template details
        """
        describe_launch_template_versions_request = ecs_models.DescribeLaunchTemplateVersionsRequest(
            region_id=self.region_id
        )
        describe_launch_template_versions_request.from_map(query_params)
        response = self.client.describe_launch_template_versions_with_options(
            describe_launch_template_versions_request, self.runtime_options)
        if (response is not None
                and response.body is not None
                and response.body.launch_template_version_sets is not None):
            return response.body.launch_template_version_sets.launch_template_version_set
        return None

    def describe_key_pair(self, key_pair_name):
        """Query the details of a key pair
        :return: The key pair details
        """
        describe_key_pairs_request = ecs_models.DescribeKeyPairsRequest(
            region_id=self.region_id,
            key_pair_name=key_pair_name
        )
        response = self.client.describe_key_pairs_with_options(
            describe_key_pairs_request, self.runtime_options)
        if (response is not None
                and response.body is not None
                and response.body.key_pairs is not None
                and response.body.key_pairs.key_pair is not None
                and len(response.body.key_pairs.key_pair) > 0):
            return response.body.key_pairs.key_pair[0]
        return None

    def create_key_pair(self, key_pair_name):
        """Create a new key pair
        :return: The key pair details with the private key
        """
        create_key_pair_request = ecs_models.CreateKeyPairRequest(
            region_id=self.region_id,
            key_pair_name=key_pair_name
        )
        response = self.client.create_key_pair_with_options(
            create_key_pair_request, self.runtime_options)
        if response is not None:
            return response.body
        return None

    def describe_instances(
            self, tags=None, instance_ids=None, vpc_id=None, status=None):
        """Query the details of one or more Elastic Compute Service (ECS) instances.

        :param tags: The tags of the instance.
        :param instance_ids: The IDs of ECS instances
        :param vpc_id: The VPC of the instances
        :param status: The status of the instances
        :return: ECS instance list
        """
        request_tags = [ecs_models.DescribeInstancesRequestTag(
            key=tag["Key"],
            value=tag["Value"]
        ) for tag in tags] if tags else None
        request_instance_ids = self._get_request_instance_ids(instance_ids)
        describe_instances_request = ecs_models.DescribeInstancesRequest(
            region_id=self.region_id,
            tag=request_tags,
            instance_ids=request_instance_ids,
            vpc_id=vpc_id,
            status=status
        )

        response = self.client.describe_instances_with_options(
            describe_instances_request, self.runtime_options)
        if (response is not None
                and response.body is not None
                and response.body.instances is not None):
            return response.body.instances.instance
        return None

    def run_instances(
            self,
            node_config: dict,
            tags,
            count=1
    ):
        """Create one or more pay-as-you-go or subscription
            Elastic Compute Service (ECS) instances
        :param node_config: The base config map of the instance.
        :param tags: The tags of the instance.
        :param count: The number of instances that you want to create.
        :return: The created instance IDs.
        """
        # Update tags and count to the config
        conf_map = node_config.copy()
        conf_map["Amount"] = count

        instance_tags = tags.copy()
        user_tags = conf_map.get("Tag", [])
        self._merge_tags(instance_tags, user_tags)
        conf_map["Tag"] = instance_tags

        run_instances_request = ecs_models.RunInstancesRequest(
            region_id=self.region_id
        )
        run_instances_request.from_map(conf_map)

        response = self.client.run_instances_with_options(
            run_instances_request, self.runtime_options)
        if (response is not None
                and response.body is not None
                and response.body.instance_id_sets is not None):
            return response.body.instance_id_sets.instance_id_set
        logging.error("instance created failed.")
        return None

    def tag_ecs_resource(self, resource_ids, tags, resource_type="Instance"):
        """Create and bind tags to specified ECS resources.

        :param resource_ids: The IDs of N resources.
        :param tags: The tags of the resource.
        :param resource_type: The type of the resource.
        """
        request_tags = [ecs_models.TagResourcesRequestTag(
            key=tag.get("Key"),
            value=tag.get("Value")
        ) for tag in tags]
        tag_resources_request = ecs_models.TagResourcesRequest(
            region_id=self.region_id,
            resource_id=resource_ids,
            resource_type=resource_type,
            tag=request_tags
        )
        response = self.client.tag_resources_with_options(
            tag_resources_request, self.runtime_options)
        if response is not None:
            logging.info("instance %s create tag successfully.", resource_ids)
        else:
            logging.error("instance %s create tag failed.", resource_ids)

    def start_instance(self, instance_id):
        """Start an ECS instance.

        :param instance_id: The Ecs instance ID.
        """
        start_instance_request = ecs_models.StartInstanceRequest(
            instance_id=instance_id
        )
        response = self.client.start_instance_with_options(
            start_instance_request, self.runtime_options)
        if response is not None:
            logging.info("instance %s start successfully.", instance_id)
        else:
            logging.error("instance %s start failed.", instance_id)

    def stop_instance(self, instance_id, force_stop=False):
        """Stop an ECS instance that is in the Running state.

        :param instance_id: The Ecs instance ID.
        :param force_stop: Specifies whether to forcibly stop the instance.
        :return:
        """
        stop_instance_request = ecs_models.StopInstanceRequest(
            instance_id=instance_id,
            force_stop=force_stop
        )

        self.client.stop_instance_with_options(
            stop_instance_request, self.runtime_options)
        logging.info("Stop %s command successfully.", instance_id)

    def stop_instances(self, instance_ids, stopped_mode="StopCharging"):
        """Stop one or more ECS instances that are in the Running state.

        :param instance_ids: The IDs of instances.
        :param stopped_mode: Specifies whether billing for the instance
                             continues after the instance is stopped.
        """
        stop_instances_request = ecs_models.StopInstancesRequest(
            region_id=self.region_id,
            stopped_mode=stopped_mode,
            instance_id=instance_ids
        )
        response = self.client.stop_instances_with_options(
            stop_instances_request, self.runtime_options)
        if response is None:
            logging.error("stop_instances failed")

    def delete_instance(self, instance_id):
        """Release a pay-as-you-go instance or
            an expired subscription instance.

        :param instance_id: The ID of the instance that you want to release.
        """
        delete_instance_request = ecs_models.DeleteInstanceRequest(
            instance_id=instance_id,
            force=True
        )
        self.client.delete_instance_with_options(
            delete_instance_request, self.runtime_options)
        logging.info("Delete %s command successfully", instance_id)

    def delete_instances(self, instance_ids):
        """Release one or more pay-as-you-go instances or
            expired subscription instances.

        :param instance_ids: The IDs of instances that you want to release.
        """
        delete_instances_request = ecs_models.DeleteInstancesRequest(
            region_id=self.region_id,
            instance_id=instance_ids,
            force=True
        )
        self.client.delete_instances_with_options(
            delete_instances_request, self.runtime_options)

    def create_security_group(self, vpc_id, name):
        """Create a security group
        :param vpc_id: The ID of the VPC in which to create
                       the security group.
        :return: The created security group ID.
        """
        create_security_group_request = ecs_models.CreateSecurityGroupRequest(
            region_id=self.region_id,
            security_group_name=name,
            vpc_id=vpc_id
        )
        try:
            response = self.client.create_security_group_with_options(
                create_security_group_request, self.runtime_options)
            return response.body.security_group_id
        except Exception as e:
            cli_logger.error("Failed to create security group. {}".format(format_exception_message(str(e))))
            raise e

    def describe_security_groups(self, vpc_id=None, name=None):
        """Query basic information of security groups.
        :param vpc_id: The ID of the VPC to which the security group belongs.
        :param tags: The tags of the security group.
        :return: Security group list.
        """
        describe_security_groups_request = ecs_models.DescribeSecurityGroupsRequest(
            region_id=self.region_id,
            vpc_id=vpc_id,
            security_group_name=name
        )
        try:
            response = self.client.describe_security_groups_with_options(
                describe_security_groups_request, self.runtime_options)
            security_groups = response.body.security_groups.security_group
            return security_groups
        except Exception as e:
            cli_logger.error("Failed to describe security groups. {}".format(format_exception_message(str(e))))
            raise e

    def revoke_security_group(self, ip_protocol, port_range, security_group_id, source_cidr_ip):
        """Revoke an inbound security group rule.
                :param ip_protocol: The transport layer protocol.
                :param port_range: The range of destination ports relevant to
                                   the transport layer protocol.
                :param security_group_id: The ID of the destination security group.
                :param source_cidr_ip: The range of source IPv4 addresses.
                                       CIDR blocks and IPv4 addresses are supported.
                """
        revoke_security_group_request = ecs_models.RevokeSecurityGroupRequest(
            region_id=self.region_id,
            security_group_id=security_group_id,
            permissions=[
                ecs_models.RevokeSecurityGroupRequestPermissions(
                    ip_protocol=ip_protocol,
                    source_cidr_ip=source_cidr_ip,
                    port_range=port_range)
            ]
        )
        try:
            self.client.revoke_security_group_with_options(
                revoke_security_group_request, self.runtime_options)
        except Exception as e:
            cli_logger.error("Failed to revoke security group rule. {}".format(format_exception_message(str(e))))
            raise e

    def describe_security_group_attribute(self, security_group_id):
        """Query basic information of security group"""
        describe_security_group_attribute_request = ecs_models.DescribeSecurityGroupAttributeRequest(
            region_id=self.region_id,
            security_group_id=security_group_id
        )
        try:
            response = self.client.describe_security_group_attribute_with_options(
                describe_security_group_attribute_request, self.runtime_options)
            return response.body
        except Exception as e:
            cli_logger.error("Failed to describe security group attribute. {}".format(format_exception_message(str(e))))
            raise e

    def authorize_security_group(
            self, ip_protocol, port_range, security_group_id, source_cidr_ip
    ):
        """Create an inbound security group rule.
        :param ip_protocol: The transport layer protocol.
        :param port_range: The range of destination ports relevant to
                           the transport layer protocol.
        :param security_group_id: The ID of the destination security group.
        :param source_cidr_ip: The range of source IPv4 addresses.
                               CIDR blocks and IPv4 addresses are supported.
        """
        authorize_security_group_request = ecs_models.AuthorizeSecurityGroupRequest(
            region_id=self.region_id,
            security_group_id=security_group_id,
            permissions=[
                ecs_models.AuthorizeSecurityGroupRequestPermissions(
                    ip_protocol=ip_protocol,
                    source_cidr_ip=source_cidr_ip,
                    port_range=port_range
                )
            ]
        )
        try:
            self.client.authorize_security_group_with_options(
                authorize_security_group_request, self.runtime_options)
        except Exception as e:
            cli_logger.error("Failed to authorize security group rule. {}".format(format_exception_message(str(e))))
            raise e

    def delete_security_group(self, security_group_id):
        """Delete security group."""
        delete_security_group_request = ecs_models.DeleteSecurityGroupRequest(
            region_id=self.region_id,
            security_group_id=security_group_id
        )
        try:
            self.client.delete_security_group_with_options(
                delete_security_group_request, self.runtime_options)
        except Exception as e:
            cli_logger.error("Failed to delete security group. {}".format(format_exception_message(str(e))))
            raise e
