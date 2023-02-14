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
    bootstrap_aliyun,
)
from cloudtik.providers._private.aliyun.utils import make_ecs_client
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

logger = logging.getLogger(__name__)

TAG_BATCH_DELAY = 1
STOPPING_NODE_DELAY = 1


class EcsClient:
    """
    A wrapper around Aliyun ECS client.

    Parameters:
        provider_config: The cloud provider configuration from which to create client.
    """

    def __init__(self, provider_config):
        self.region_id = provider_config["region"]
        self.client = make_ecs_client(provider_config)
        self.runtime_options = util_models.RuntimeOptions()

    @staticmethod
    def get_request_instance_ids(instance_ids):
        if not instance_ids:
            return None
        return "[" + ",".join(['"' + instance_id + '"' for instance_id in instance_ids]) + "]"

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
        request_instance_ids = self.get_request_instance_id(instance_ids)
        describe_instances_request = ecs_models.DescribeInstancesRequest(
            region_id=self.region_id,
            tag=request_tags,
            instance_ids=request_instance_ids
        )

        response = self.client.describe_instances_with_options(
            describe_instances_request, self.runtime_options)
        if response is not None:
            instance_list = response.get("Instances").get("Instance")
            return instance_list
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
        request_tags = [ecs_models.RunInstancesRequestTag(
            key=tag["Key"],
            value=tag["Value"]
        ) for tag in tags] if tags else None
        run_instances_request = ecs_models.RunInstancesRequest(
            region_id=self.region_id,
            instance_type=instance_type,
            image_id=image_id,
            tag=request_tags,
            security_group_id=security_group_id,
            v_switch_id=vswitch_id,
            amount=amount,
            key_pair_name=key_pair_name,
            io_optimized=optimized,
            instance_charge_type=instance_charge_type,
            spot_strategy=spot_strategy,
            internet_charge_type=internet_charge_type,
            internet_max_bandwidth_out=internet_max_bandwidth_out
        )
        response = self.client.run_instances_with_options(
            run_instances_request, self.runtime_options)
        if response is not None:
            instance_ids = response.get("InstanceIdSets").get("InstanceIdSet")
            return instance_ids
        logging.error("instance created failed.")
        return None

    def tag_ecs_resource(self, resource_ids, tags, resource_type="instance"):
        """Create and bind tags to specified ECS resources.

        :param resource_ids: The IDs of N resources.
        :param tags: The tags of the resource.
        :param resource_type: The type of the resource.
        """
        # TODO
        response = None
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
            if instance.get("Status") == RUNNING or instance.get("Status") == PENDING:
                non_terminated_instance.append(instance.get("InstanceId"))
                self.cached_nodes[instance.get("InstanceId")] = instance
        return non_terminated_instance

    def is_running(self, node_id: str) -> bool:
        instances = self.ecs.describe_instances(instance_ids=[node_id])
        if instances is not None:
            instance = instances[0]
            return instance.get("Status") == "Running"
        cli_logger.error("Invalid node id: %s", node_id)
        return False

    def is_terminated(self, node_id: str) -> bool:
        instances = self.ecs.describe_instances(instance_ids=[node_id])
        if instances is not None:
            assert len(instances) == 1
            instance = instances[0]
            return instance.get("Status") == "Stopped"
        cli_logger.error("Invalid node id: %s", node_id)
        return False

    def node_tags(self, node_id: str) -> Dict[str, str]:
        instances = self.ecs.describe_instances(instance_ids=[node_id])
        if instances is not None:
            assert len(instances) == 1
            instance = instances[0]
            if instance.get("Tags") is not None:
                node_tags = dict()
                for tag in instance.get("Tags").get("Tag"):
                    node_tags[tag.get("TagKey")] = tag.get("TagValue")
                return node_tags
        return dict()

    def external_ip(self, node_id: str) -> str:
        while True:
            instances = self.ecs.describe_instances(instance_ids=[node_id])
            if instances is not None:
                assert len(instances)
                instance = instances[0]
                if (
                    instance.get("PublicIpAddress") is not None
                    and instance.get("PublicIpAddress").get("IpAddress") is not None
                ):
                    if len(instance.get("PublicIpAddress").get("IpAddress")) > 0:
                        return instance.get("PublicIpAddress").get("IpAddress")[0]
            cli_logger.error("PublicIpAddress attribute is not exist. %s" % instance)
            time.sleep(STOPPING_NODE_DELAY)

    def internal_ip(self, node_id: str) -> str:
        while True:
            instances = self.ecs.describe_instances(instance_ids=[node_id])
            if instances is not None:
                assert len(instances) == 1
                instance = instances[0]
                if (
                    instance.get("VpcAttributes") is not None
                    and instance.get("VpcAttributes").get("PrivateIpAddress")
                    is not None
                    and len(
                        instance.get("VpcAttributes")
                        .get("PrivateIpAddress")
                        .get("IpAddress")
                    )
                    > 0
                ):
                    return (
                        instance.get("VpcAttributes")
                        .get("PrivateIpAddress")
                        .get("IpAddress")[0]
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
                        node_id = node.get("InstanceId")
                        status = node.get("Status")
                        if status != STOPPING and status != STOPPED:
                            continue
                        if status == STOPPING:
                            # wait for node stopped
                            while (
                                self.ecs.describe_instances(instance_ids=[node_id])[
                                    0
                                ].get("Status")
                                == STOPPING
                            ):
                                logging.info("wait for %s stop" % node_id)
                                time.sleep(STOPPING_NODE_DELAY)
                        # logger.info("reuse %s" % node_id)
                        reuse_node_ids.append(node_id)
                        reused_nodes_dict[node.get("InstanceId")] = node
                        self.ecs.start_instance(node_id)
                        self.tag_cache[node_id] = node.get("Tags")
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
                instance_type=node_config["InstanceType"],
                image_id=node_config["ImageId"],
                tags=filter_tags,
                amount=count,
                vswitch_id=self.provider_config["v_switch_id"],
                security_group_id=self.provider_config["security_group_id"],
                key_pair_name=self.provider_config["key_name"],
            )
            instances = self.ecs.describe_instances(instance_ids=instance_id_sets)

            if instances is not None:
                for instance in instances:
                    created_nodes_dict[instance.get("InstanceId")] = instance

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

    @staticmethod
    def bootstrap_config(cluster_config):
        return bootstrap_aliyun(cluster_config)
