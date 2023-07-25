import logging
import copy
import time
import math
from typing import Any, Dict, List

from cloudtik.core._private.constants import CLOUDTIK_DEFAULT_CLOUD_STORAGE_URI
from cloudtik.core._private.utils import get_storage_config_for_update, get_config_for_update

from cloudtik.core._private.cli_logger import cli_logger

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

ALIYUN_OSS_BUCKET = "oss.bucket"
ALIYUN_OSS_INTERNAL_ENDPOINT = "oss.internal.endpoint"
CLIENT_MAX_RETRY_ATTEMPTS = 5


def get_aliyun_oss_storage_config(provider_config: Dict[str, Any]):
    if "storage" in provider_config and "aliyun_oss_storage" in provider_config["storage"]:
        return provider_config["storage"]["aliyun_oss_storage"]

    return None


def get_aliyun_oss_storage_config_for_update(provider_config: Dict[str, Any]):
    storage_config = get_storage_config_for_update(provider_config)
    return get_config_for_update(storage_config, "aliyun_oss_storage")


def export_aliyun_oss_storage_config(provider_config, config_dict: Dict[str, Any]):
    cloud_storage = get_aliyun_oss_storage_config(provider_config)
    if cloud_storage is None:
        return
    config_dict["ALIYUN_CLOUD_STORAGE"] = True

    oss_bucket = cloud_storage.get(ALIYUN_OSS_BUCKET)
    if oss_bucket:
        config_dict["ALIYUN_OSS_BUCKET"] = oss_bucket

    oss_internal_endpoint = cloud_storage.get(ALIYUN_OSS_INTERNAL_ENDPOINT)
    if oss_internal_endpoint:
        config_dict["ALIYUN_OSS_INTERNAL_ENDPOINT"] = oss_internal_endpoint

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
            and len(node.vpc_attributes.private_ip_address.ip_address) > 0):
        private_ip = node.vpc_attributes.private_ip_address.ip_address[0]
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
    aliyun_ram_role_name = provider_config.get("ram_role_name")
    return _get_credential(aliyun_credentials, aliyun_ram_role_name)


def _get_credential(aliyun_credentials=None, ram_role_name=None):
    if aliyun_credentials is not None:
        ak = aliyun_credentials.get("aliyun_access_key_id")
        sk = aliyun_credentials.get("aliyun_access_key_secret")
        st = aliyun_credentials.get("aliyun_security_token")
        if st:
            credential_config = Config(
                type='sts',  # credential type
                access_key_id=ak,  # AccessKeyId
                access_key_secret=sk,  # AccessKeySecret
                security_token=st,  # Security Token
            )
        else:
            credential_config = Config(
                type='access_key',  # credential type
                access_key_id=ak,  # AccessKeyId
                access_key_secret=sk,  # AccessKeySecret
            )
        credential = CredentialClient(credential_config)
    elif ram_role_name is not None:
        credential_config = Config(
            type='ecs_ram_role',
            role_name=ram_role_name
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
    config.endpoint = _get_oss_public_endpoint(region_id)
    return oss_client(config)


def _get_oss_internal_endpoint(region_id):
    return f"oss-{region_id}-internal.aliyuncs.com"


def _get_oss_public_endpoint(region_id):
    return f"oss-{region_id}.aliyuncs.com"


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
            cli_logger.error("Failed to get attributes of resource. {}", str(e))
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
        self.runtime_options = util_models.RuntimeOptions(
            autoretry=True,
            max_attempts=CLIENT_MAX_RETRY_ATTEMPTS
        )

    def put_bucket(self, name):
        put_bucket_request = oss_models.PutBucketRequest(oss_models.CreateBucketConfiguration())
        try:
            self.client.put_bucket(name, put_bucket_request)
        except TeaException as e:
            cli_logger.error("Failed to create bucket. {}", str(e))
            raise e
        except Exception as e:
            cli_logger.error("Ignore the exception. {}", str(e))

    def delete_bucket(self, name):
        try:
            self.client.delete_bucket(name)
        except Exception as e:
            cli_logger.error("Failed to delete bucket. {}", str(e))
            raise e
    
    def list_buckets(self):
        list_bucket_request = oss_models.ListBucketsRequest()
        try:
            response = self.client.list_buckets(list_bucket_request)
            if (response is not None
                    and response.body is not None
                    and response.body.buckets is not None):
                return response.body.buckets.buckets
            return []
        except Exception as e:
            cli_logger.error("Failed to list buckets. {}", str(e))
            raise e
        
    def list_objects(self, bucket_name):
        list_objects_request = oss_models.ListObjectsRequest()
        try:
            response = self.client.list_objects(bucket_name, list_objects_request)
            return response.body.contents if response.body is not None else []
        except Exception as e:
            cli_logger.error("Failed to list objects of the bucket: {}. {}", bucket_name, str(e))
            raise e

    def delete_object(self, bucket_name, object_key):
        delete_object_request = oss_models.DeleteObjectRequest()
        try:
            self.client.delete_object(bucket_name, object_key, delete_object_request)
        except Exception as e:
            cli_logger.error("Failed to delete the object: {} of the bucket: {}. {}",
                             object_key, bucket_name, str(e))
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
        self.runtime_options = util_models.RuntimeOptions(
            autoretry=True,
            max_attempts=CLIENT_MAX_RETRY_ATTEMPTS
        )

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
            cli_logger.error("Failed to describe VPCs. {}", str(e))
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
            cli_logger.error("Failed to get the attribute of the VPC. {}", str(e))
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
            cli_logger.error("Failed to create VPC. {}", str(e))
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
            cli_logger.error("Failed to delete VPC. {}", str(e))
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
            cli_logger.error("Failed to tag VPC. {}", str(e))
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
            cli_logger.error("Failed to untag VPC. {}", str(e))
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
            cli_logger.error("Failed to describe zones. {}", str(e))
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
            cli_logger.error("Failed to list enhanced nat gate-way available zones. {}", str(e))
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
            cli_logger.error("Failed to describe the attributes of the vswitch. {}", str(e))
            raise e

    def describe_vswitches(self, vpc_id=None):
        """Queries one or more VSwitches.
        :param vpc_id: The ID of the VPC to which the VSwitch belongs.
        :return: VSwitch list.
        """
        vswitches_list = []
        try:
            describe_vswitches_request = vpc_models.DescribeVSwitchesRequest(
                vpc_id=vpc_id,
            )
            response = self.client.describe_vswitches_with_options(
                describe_vswitches_request, self.runtime_options)
            vswitches_list.extend(
                copy.deepcopy(response.body.v_switches.v_switch))
            total_count = response.body.total_count
            page_size = response.body.page_size
            total_page_num = math.ceil(float(total_count / page_size))
            if total_page_num > 1:
                for page_num in range(2, total_page_num + 1):
                    describe_vswitches_request = vpc_models.DescribeVSwitchesRequest(
                        vpc_id=vpc_id,
                        page_number=page_num
                    )
                    response = self.client.describe_vswitches_with_options(
                        describe_vswitches_request, self.runtime_options)
                    vswitches_list.extend(
                        copy.deepcopy(response.body.v_switches.v_switch))
            return vswitches_list
        except Exception as e:
            cli_logger.error("Failed to describe vswitches. {}", str(e))
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
            cli_logger.error("Failed to delete vswitch. {}", str(e))
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
            cli_logger.error("Failed to create vswitch. {}", str(e))
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
            cli_logger.error("Failed to describe route tables. {}", str(e))
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
            cli_logger.error("Failed to describe nat-gateways. {}", str(e))
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
            cli_logger.error("Failed to get the attribute of the nat-gateway. {}", str(e))
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
            cli_logger.error("Failed to delete nat-gateway. {}", str(e))
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
            cli_logger.error("Failed to create nat-gateway. {}", str(e))
            raise e

    def allocate_eip_address(
            self, name, bandwidth='100', instance_charge_type='PostPaid', internet_charge_type='PayByTraffic'):
        """Allocate elastic ip address
        :return allocation_id:
        """
        allocate_eip_address_request = vpc_models.AllocateEipAddressRequest(
            region_id=self.region_id,
            name=name,
            bandwidth=bandwidth,
            instance_charge_type=instance_charge_type,
            internet_charge_type=internet_charge_type
        )
        try:
            response = self.client.allocate_eip_address_with_options(
                allocate_eip_address_request, self.runtime_options)
            return response.body.allocation_id
        except Exception as e:
            cli_logger.error("Failed to allocate EIP. {}", str(e))
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
            cli_logger.error("Failed to associate EIP to instance. {}", str(e))
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
            cli_logger.error("Failed to describe EIP addresses. {}", str(e))
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
            cli_logger.error("Failed to unassociate EIP address from instance. {}", str(e))
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
            cli_logger.error("Failed to release EIP address. {}", str(e))
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
            cli_logger.error("Failed to create SNAT Entry. {}", str(e))
            raise e

    def describe_snat_entries(self, snat_table_id=None, snat_entry_id=None):
        """Describe SNAT Entries for snat table"""
        snat_table_entry_list = []
        try:
            describe_snat_table_entries_request = vpc_models.DescribeSnatTableEntriesRequest(
                region_id=self.region_id,
                snat_table_id=snat_table_id,
                snat_entry_id=snat_entry_id
            )
            response = self.client.describe_snat_table_entries_with_options(
                describe_snat_table_entries_request, self.runtime_options)
            snat_table_entry_list.extend(
                copy.deepcopy(response.body.snat_table_entries.snat_table_entry))
            total_count = response.body.total_count
            page_size = response.body.page_size
            total_page_num = math.ceil(float(total_count / page_size))
            if total_page_num > 1:
                for page_num in range(2, total_page_num + 1):
                    describe_snat_table_entries_request = vpc_models.DescribeSnatTableEntriesRequest(
                        region_id=self.region_id,
                        snat_table_id=snat_table_id,
                        snat_entry_id=snat_entry_id,
                        page_number=page_num
                    )
                    response = self.client.describe_snat_table_entries_with_options(
                        describe_snat_table_entries_request, self.runtime_options)
                    snat_table_entry_list.extend(
                        copy.deepcopy(response.body.snat_table_entries.snat_table_entry))
            return snat_table_entry_list
        except Exception as e:
            cli_logger.error("Failed to describe SNAT Entries. {}", str(e))
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
            cli_logger.error("Failed to delete SNAT Entry. {}", str(e))
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
            cli_logger.error("Failed to create route entry. {}", str(e))
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
            cli_logger.error("Failed to describe route entries. {}", str(e))
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
            cli_logger.error("Failed to delete route entry. {}", str(e))
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
        self.runtime_options = util_models.RuntimeOptions(
            autoretry=True,
            max_attempts=CLIENT_MAX_RETRY_ATTEMPTS
        )

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
            cli_logger.error("Failed to create vpc peer connection. {}", str(e))
            raise e

    def delete_vpc_peer_connection(self, instance_id):
        delete_vpc_peer_connection_request = vpc_peer_models.DeleteVpcPeerConnectionRequest(
            instance_id=instance_id
        )
        try:
            self.client.delete_vpc_peer_connection_with_options(
                delete_vpc_peer_connection_request, self.runtime_options)
        except Exception as e:
            cli_logger.error("Failed to delete vpc peer connection. {}", str(e))
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
            cli_logger.error("Failed to describe vpc peer connections. {}", str(e))
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
        self.runtime_options = util_models.RuntimeOptions(
            autoretry=True,
            max_attempts=CLIENT_MAX_RETRY_ATTEMPTS
        )

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
            cli_logger.error("Failed to create RAM role. {}", str(e))
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
        except TeaException as e:
            if e.code == "EntityNotExist.Role":
                return None
            raise e
        except Exception as e:
            cli_logger.error("Failed to get RAM role. {}", str(e))
            raise e

    def delete_role(self, role_name):
        """Delete RAM role"""
        delete_role_request = ram_models.DeleteRoleRequest(
            role_name=role_name
        )
        try:
            self.client.delete_role_with_options(
                delete_role_request, self.runtime_options)
        except Exception as e:
            cli_logger.error("Failed to delete RAM role. {}", str(e))
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
            cli_logger.error("Failed to attach the policy to RAM role. {}", str(e))
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
            cli_logger.error("Failed to detach the policy from RAM role. {}", str(e))
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
            cli_logger.error("Failed to list the policies for RAM role. {}", str(e))
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
        self.runtime_options = util_models.RuntimeOptions(
            autoretry=True,
            max_attempts=CLIENT_MAX_RETRY_ATTEMPTS
        )

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

    def describe_images(self, image_family, image_name):
        """List the images available
        :return: The list of images matched
        """
        describe_images_request = ecs_models.DescribeImagesRequest(
            region_id=self.region_id,
            architecture='x86_64',
            ostype='linux',
            status='Available',
            image_family=image_family,
            image_name=image_name
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
            cli_logger.error("Failed to create security group. {}", str(e))
            raise e

    def describe_security_groups(self, vpc_id=None, name=None):
        """Query basic information of security groups.
        :param vpc_id: The ID of the VPC to which the security group belongs.
        :param name: The name of the security group.
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
            cli_logger.error("Failed to describe security groups. {}", str(e))
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
            cli_logger.error("Failed to revoke security group rule. {}", str(e))
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
            cli_logger.error("Failed to describe security group attribute. {}", str(e))
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
            cli_logger.error("Failed to authorize security group rule. {}", str(e))
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
            cli_logger.error("Failed to delete security group. {}", str(e))
            raise e
