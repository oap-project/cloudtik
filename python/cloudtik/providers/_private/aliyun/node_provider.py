import logging
import random
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from alibabacloud_ecs20140526 import models as ecs_models
from alibabacloud_tea_util import models as util_models

from cloudtik.providers._private.aliyun.config import (
    PENDING,
    RUNNING,
    STOPPED,
    STOPPING,
    bootstrap_aliyun, verify_oss_storage, post_prepare_aliyun, with_aliyun_environment_variables,
)
from cloudtik.providers._private.aliyun.utils import make_ecs_client, get_default_aliyun_cloud_storage, \
    get_aliyun_oss_storage_config, _get_node_info, tags_list_to_dict
from cloudtik.core._private.cli_logger import cli_logger
from cloudtik.providers._private.aliyun.utils import ACS_MAX_RETRIES
from cloudtik.core._private.log_timer import LogTimer
from cloudtik.core.node_provider import NodeProvider
from cloudtik.core.tags import (
    CLOUDTIK_TAG_CLUSTER_NAME, 
    CLOUDTIK_TAG_NODE_NAME,
    CLOUDTIK_TAG_LAUNCH_CONFIG,
    CLOUDTIK_TAG_NODE_KIND,
    CLOUDTIK_TAG_USER_NODE_TYPE,
    CLOUDTIK_TAG_NODE_STATUS
)
from cloudtik.providers._private.utils import validate_config_dict

logger = logging.getLogger(__name__)

TAG_BATCH_DELAY = 1
STOPPING_NODE_DELAY = 1


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
        if (response is not None
                and response.body is not None):
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
                and response.body.images is not None
                and response.body.images.image is not None):
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
                and response.body.launch_template_version_sets is not None
                and response.body.launch_template_version_sets.launch_template_version_set is not None):

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
        if (response is not None
                and response.body is not None):
            return response.body
        return None

    def describe_instances(self, tags=None, instance_ids=None):
        """Query the details of one or more Elastic Compute Service (ECS) instances.

        :param tags: The tags of the instance.
        :param instance_ids: The IDs of ECS instances
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
            instance_ids=request_instance_ids
        )

        response = self.client.describe_instances_with_options(
            describe_instances_request, self.runtime_options)
        if (response is not None
                and response.body is not None
                and response.body.instances is not None
                and response.body.instances.instance is not None):
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
                and response.body.instance_id_sets is not None
                and response.body.instance_id_sets.instance_id_set is not None):
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
            cli_logger.error("Failed to create security group. {}".format(e))
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
            cli_logger.error("Failed to describe security groups. {}".format(e))
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
            cli_logger.error("Failed to revoke security group rule. {}".format(e))
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
            cli_logger.error("Failed to describe security group attribute. {}".format(e))
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
            cli_logger.error("Failed to authorize security group rule. {}".format(e))
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
            cli_logger.error("Failed to delete security group. {}".format(e))
            raise e


class AliyunNodeProvider(NodeProvider):
    def __init__(self, provider_config, cluster_name):
        NodeProvider.__init__(self, provider_config, cluster_name)
        self.cache_stopped_nodes = provider_config.get("cache_stopped_nodes", True)
        self.ecs = EcsClient(provider_config)

        # Try availability zones round-robin, starting from random offset
        self.subnet_idx = random.randint(0, 100)

        # Tags that we believe to actually be on the node.
        self.tag_cache = {}
        # Tags that we will soon upload.
        self.tag_cache_pending = defaultdict(dict)
        # Number of threads waiting for a batched tag update.
        self.batch_thread_count = 0
        self.batch_update_done = threading.Event()
        self.batch_update_done.set()
        self.ready_for_new_batch = threading.Event()
        self.ready_for_new_batch.set()
        self.tag_cache_lock = threading.Lock()
        self.count_lock = threading.Lock()

        # Cache of node objects from the last nodes() call. This avoids
        # excessive DescribeInstances requests.
        self.cached_nodes = {}

    def non_terminated_nodes(self, tag_filters: Dict[str, str]) -> List[str]:
        tags = [
            {
                "Key": CLOUDTIK_TAG_CLUSTER_NAME,
                "Value": self.cluster_name,
            },
        ]
        for k, v in tag_filters.items():
            tags.append(
                {
                    "Key": k,
                    "Value": v,
                }
            )

        instances = self.ecs.describe_instances(tags=tags)
        non_terminated_instance = []
        for instance in instances:
            if instance.status == RUNNING or instance.status == PENDING:
                non_terminated_instance.append(instance.instance_id)
                self.cached_nodes[instance.instance_id] = instance
        return non_terminated_instance

    def is_running(self, node_id: str) -> bool:
        instances = self.ecs.describe_instances(instance_ids=[node_id])
        if instances:
            assert len(instances) == 1
            instance = instances[0]
            return instance.status == RUNNING
        cli_logger.error("Invalid node id: %s", node_id)
        return False

    def is_terminated(self, node_id: str) -> bool:
        instances = self.ecs.describe_instances(instance_ids=[node_id])
        if instances:
            assert len(instances) == 1
            instance = instances[0]
            return instance.status == STOPPED
        cli_logger.error("Invalid node id: %s", node_id)
        return False

    def node_tags(self, node_id: str) -> Dict[str, str]:
        instances = self.ecs.describe_instances(instance_ids=[node_id])
        if instances:
            assert len(instances) == 1
            instance = instances[0]
            if instance.tags is not None \
                    and instance.tags.tag is not None:
                node_tags = dict()
                for tag in instance.tags.tag:
                    node_tags[tag.tag_key] = tag.tag_value
                return node_tags
        return dict()

    def external_ip(self, node_id: str) -> str:
        while True:
            instances = self.ecs.describe_instances(instance_ids=[node_id])
            if instances:
                assert len(instances) == 1
                instance = instances[0]
                if (
                    instance.public_ip_address is not None
                    and instance.public_ip_address.ip_address is not None
                ):
                    if len(instance.public_ip_address.ip_address) > 0:
                        return instance.public_ip_address.ip_address[0]
            cli_logger.error("PublicIpAddress attribute is not exist. %s" % instance)
            time.sleep(STOPPING_NODE_DELAY)

    def internal_ip(self, node_id: str) -> str:
        while True:
            instances = self.ecs.describe_instances(instance_ids=[node_id])
            if instances:
                assert len(instances) == 1
                instance = instances[0]
                if (
                    instance.vpc_attributes is not None
                    and instance.vpc_attributes.private_ip_address is not None
                    and len(instance.vpc_attributes.private_ip_address) > 0
                ):
                    return (
                        instance.vpc_attributes.private_ip_address[0]
                    )
            cli_logger.error("InnerIpAddress attribute is not exist. %s" % instance)
            time.sleep(STOPPING_NODE_DELAY)

    def set_node_tags(self, node_id: str, tags: Dict[str, str]) -> None:
        is_batching_thread = False
        with self.tag_cache_lock:
            if not self.tag_cache_pending:
                is_batching_thread = True
                # Wait for threads in the last batch to exit
                self.ready_for_new_batch.wait()
                self.ready_for_new_batch.clear()
                self.batch_update_done.clear()
            self.tag_cache_pending[node_id].update(tags)

        if is_batching_thread:
            time.sleep(TAG_BATCH_DELAY)
            with self.tag_cache_lock:
                self._update_node_tags()
                self.batch_update_done.set()

        with self.count_lock:
            self.batch_thread_count += 1
        self.batch_update_done.wait()

        with self.count_lock:
            self.batch_thread_count -= 1
            if self.batch_thread_count == 0:
                self.ready_for_new_batch.set()

    def _update_node_tags(self):
        batch_updates = defaultdict(list)

        for node_id, tags in self.tag_cache_pending.items():
            for x in tags.items():
                batch_updates[x].append(node_id)
            self.tag_cache[node_id] = tags

        self.tag_cache_pending = defaultdict(dict)

        self._create_tags(batch_updates)

    def _create_tags(self, batch_updates):

        for (k, v), node_ids in batch_updates.items():
            m = "Set tag {}={} on {}".format(k, v, node_ids)
            with LogTimer("AliyunNodeProvider: {}".format(m)):
                if k == CLOUDTIK_TAG_NODE_NAME:
                    k = "Name"

                self.ecs.tag_ecs_resource(node_ids, [{"Key": k, "Value": v}])

    def create_node(
        self, node_config: Dict[str, Any], tags: Dict[str, str], count: int
    ) -> Optional[Dict[str, Any]]:
        filter_tags = [
            {
                "Key": CLOUDTIK_TAG_CLUSTER_NAME,
                "Value": self.cluster_name,
            },
            {"Key": CLOUDTIK_TAG_NODE_KIND, "Value": tags[CLOUDTIK_TAG_NODE_KIND]},
            {"Key": CLOUDTIK_TAG_USER_NODE_TYPE, "Value": tags[CLOUDTIK_TAG_USER_NODE_TYPE]},
            {"Key": CLOUDTIK_TAG_LAUNCH_CONFIG, "Value": tags[CLOUDTIK_TAG_LAUNCH_CONFIG]},
            {"Key": CLOUDTIK_TAG_NODE_NAME, "Value": tags[CLOUDTIK_TAG_NODE_NAME]},
        ]

        reused_nodes_dict = {}
        if self.cache_stopped_nodes:
            reuse_nodes_candidate = self.ecs.describe_instances(tags=filter_tags)
            if reuse_nodes_candidate:
                with cli_logger.group("Stopping instances to reuse"):
                    reuse_node_ids = []
                    for node in reuse_nodes_candidate:
                        node_id = node.instance_id
                        status = node.status
                        if status != STOPPING and status != STOPPED:
                            continue
                        if status == STOPPING:
                            # wait for node stopped
                            while (
                                self.ecs.describe_instances(instance_ids=[node_id])[
                                    0
                                ].status == STOPPING
                            ):
                                logging.info("wait for %s stop" % node_id)
                                time.sleep(STOPPING_NODE_DELAY)
                        # logger.info("reuse %s" % node_id)
                        reuse_node_ids.append(node_id)
                        reused_nodes_dict[node.instance_id] = node
                        self.ecs.start_instance(node_id)
                        node_tags = None
                        if (node.tags is not None and
                                node.tags.tag is not None):
                            node_tags = tags_list_to_dict(node.tags.tag)
                        self.tag_cache[node_id] = node_tags
                        self.set_node_tags(node_id, tags)
                        if len(reuse_node_ids) == count:
                            break
                count -= len(reuse_node_ids)

        created_nodes_dict = {}
        if count > 0:
            filter_tags.append(
                {"Key": CLOUDTIK_TAG_NODE_STATUS, "Value": tags[CLOUDTIK_TAG_NODE_STATUS]}
            )
            instance_id_sets = self.ecs.run_instances(
                node_config=node_config,
                tags=filter_tags,
                count=count
            )
            instances = self.ecs.describe_instances(instance_ids=instance_id_sets)

            if instances is not None:
                for instance in instances:
                    created_nodes_dict[instance.instance_id] = instance

        all_created_nodes = reused_nodes_dict
        all_created_nodes.update(created_nodes_dict)
        return all_created_nodes

    def terminate_node(self, node_id: str) -> None:
        logger.info("terminate node: %s" % node_id)
        if self.cache_stopped_nodes:
            logger.info(
                "Stopping instance {} (to terminate instead, "
                "set `cache_stopped_nodes: False` "
                "under `provider` in the cluster configuration)"
            ).format(node_id)
            self.ecs.stop_instance(node_id)
        else:
            self.ecs.delete_instance(node_id)

    def terminate_nodes(self, node_ids: List[str]) -> None:
        if not node_ids:
            return
        if self.cache_stopped_nodes:
            logger.info(
                "Stopping instances {} (to terminate instead, "
                "set `cache_stopped_nodes: False` "
                "under `provider` in the cluster configuration)".format(node_ids)
            )

            self.ecs.stop_instances(node_ids)
        else:
            self.ecs.delete_instances(node_ids)

    def _get_node(self, node_id):
        """Refresh and get info for this node, updating the cache."""
        self.non_terminated_nodes({})  # Side effect: updates cache

        if node_id in self.cached_nodes:
            return self.cached_nodes[node_id]

        # Node not in {pending, running} -- retry with a point query. This
        # usually means the node was recently preempted or terminated.
        matches = self.ecs.describe_instances(instance_ids=[node_id])

        assert len(matches) == 1, "Invalid instance id {}".format(node_id)
        return matches[0]

    def _get_cached_node(self, node_id):
        """Return node info from cache if possible, otherwise fetches it."""
        if node_id in self.cached_nodes:
            return self.cached_nodes[node_id]

        return self._get_node(node_id)

    def prepare_for_head_node(
            self, cluster_config: Dict[str, Any], remote_config: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a new remote cluster config with custom configs for head node.
        The cluster config may also be updated for setting up the head"""
        # Since the head will use the instance profile and role to access cloud,
        # remove the client credentials from config
        if "aliyun_credentials" in remote_config["provider"]:
            remote_config.pop("aliyun_credentials", None)

        return remote_config

    def get_node_info(self, node_id: str) -> Dict[str, str]:
        node = self._get_cached_node(node_id)
        return _get_node_info(node)

    def with_environment_variables(self, node_type_config: Dict[str, Any], node_id: str):
        """Export necessary environment variables for running node commands"""
        return with_aliyun_environment_variables(self.provider_config, node_type_config, node_id)

    def get_default_cloud_storage(self):
        """Return the managed cloud storage if configured."""
        return get_default_aliyun_cloud_storage(self.provider_config)

    @staticmethod
    def post_prepare(
            cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Fills out missing fields after the user config is merged with defaults
        This happens after prepare_config is done.
        """
        return post_prepare_aliyun(cluster_config)

    @staticmethod
    def validate_config(
            provider_config: Dict[str, Any]) -> None:
        """Check the provider configuration validation.
        This happens after post_prepare is done and before bootstrap_config
        """
        config_dict = {
            "region": provider_config.get("region")}
        validate_config_dict(provider_config["type"], config_dict)

        storage_config = get_aliyun_oss_storage_config(provider_config)
        if storage_config is not None:
            config_dict = {
                "oss.bucket": storage_config.get("oss.bucket"),
                # The access key is no longer a must since we have role access
                # "oss.access.key.id": storage_config.get("oss.access.key.id"),
                # "oss.access.key.secret": storage_config.get("oss.access.key.secret")
            }

            validate_config_dict(provider_config["type"], config_dict)

    @staticmethod
    def bootstrap_config(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Bootstraps the cluster config by adding env defaults if needed.
        This happens after validate_config is done.
        """
        return bootstrap_aliyun(cluster_config)

    @staticmethod
    def verify_config(
            provider_config: Dict[str, Any]) -> None:
        """Verify provider configuration. Verification usually means to check it is working.
        This happens after bootstrap_config is done.
        """
        verify_cloud_storage = provider_config.get("verify_cloud_storage", True)
        cloud_storage = get_aliyun_oss_storage_config(provider_config)
        if verify_cloud_storage and cloud_storage is not None:
            cli_logger.verbose("Verifying OSS storage configurations...")
            verify_oss_storage(provider_config)
            cli_logger.verbose("Successfully verified OSS storage configurations.")
