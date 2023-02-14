import json
import logging

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


from cloudtik.core._private.constants import env_integer
ACS_MAX_RETRIES = env_integer("ACS_MAX_RETRIES", 12)


from alibabacloud_vpc20160428.client import Client as VpcClient
from alibabacloud_credentials.client import Client as CredentialClient
from alibabacloud_credentials.models import Config
from alibabacloud_tea_openapi import models as open_api_models


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


def get_credential(provider_config):
    aliyun_credentials = provider_config.get("aliyun_credentials")
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


def construct_vpc_client(config):
    credential = get_credential(config["provider_config"])
    config = open_api_models.Config(
        credential=credential
    )
    config.endpoint = f'vpc.aliyuncs.com'
    return VpcClient(config)

