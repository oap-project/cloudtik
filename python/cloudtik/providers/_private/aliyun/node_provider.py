import logging
import random
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from Tea.exceptions import TeaException

from cloudtik.providers._private.aliyun.config import (
    PENDING,
    RUNNING,
    STOPPED,
    STOPPING,
    STARTING,
    bootstrap_aliyun, verify_oss_storage, post_prepare_aliyun, with_aliyun_environment_variables,
)
from cloudtik.providers._private.aliyun.utils import CLIENT_MAX_RETRY_ATTEMPTS, get_default_aliyun_cloud_storage, \
    get_aliyun_oss_storage_config, _get_node_info, tags_list_to_dict, EcsClient
from cloudtik.core._private.cli_logger import cli_logger
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
STARTING_NODE_DELAY = 1
MAX_CREATE_STATUS_ATTEMPTS = 30


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
            if instance.status in [RUNNING, PENDING, STARTING]:
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
                    and len(instance.vpc_attributes.private_ip_address.ip_address) > 0
                ):
                    return (
                        instance.vpc_attributes.private_ip_address.ip_address[0]
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
        reused_nodes_dict = {}
        if self.cache_stopped_nodes:
            filter_tags = [
                {"Key": CLOUDTIK_TAG_CLUSTER_NAME, "Value": self.cluster_name},
                {"Key": CLOUDTIK_TAG_NODE_KIND, "Value": tags[CLOUDTIK_TAG_NODE_KIND]},
                {"Key": CLOUDTIK_TAG_USER_NODE_TYPE, "Value": tags[CLOUDTIK_TAG_USER_NODE_TYPE]},
                {"Key": CLOUDTIK_TAG_LAUNCH_CONFIG, "Value": tags[CLOUDTIK_TAG_LAUNCH_CONFIG]},
                {"Key": CLOUDTIK_TAG_NODE_NAME, "Value": tags[CLOUDTIK_TAG_NODE_NAME]},
            ]
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
                                self.ecs.describe_instances(instance_ids=[node_id])[0].status == STOPPING
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
            tags_for_creation = [{"Key": key, "Value": value} for key, value in tags.items()]
            tags_for_creation.append(
                {"Key": CLOUDTIK_TAG_CLUSTER_NAME, "Value": self.cluster_name},
            )

            conf = node_config.copy()
            # VSwitchIds is not a real config key: we must resolve to a
            # single VSwitchId before invoking the Aliyun API.
            vswitch_ids = conf.pop("VSwitchIds")
            vswitch_idx = 0
            cli_logger_tags = {}
            # NOTE: This ensures that we try ALL availability zones before
            # throwing an error.
            max_tries = max(CLIENT_MAX_RETRY_ATTEMPTS, len(vswitch_ids))
            for attempt in range(1, max_tries + 1):
                try:
                    vswitch_id = vswitch_ids[vswitch_idx % len(vswitch_ids)]
                    conf["VSwitchId"] = vswitch_id
                    cli_logger_tags["vswitch_id"] = vswitch_id
                    instance_id_sets = self.ecs.run_instances(
                        node_config=conf,
                        tags=tags_for_creation,
                        count=count
                    )
                    # The status checking is for workaround one issue of Alibaba Cloud
                    # After a node is created, it will go through the following status
                    # Pending -> Stopped -> Starting -> Running
                    # The Stopped status in the middle causes problem because a node created
                    # will be considered to be stopped status. So we wait here until they are
                    # at least in Starting status.
                    status_attempt = 0
                    while status_attempt < MAX_CREATE_STATUS_ATTEMPTS:
                        instances = self.ecs.describe_instances(instance_ids=instance_id_sets)
                        if instances:
                            # Counting on both STARTING and RUNNING nodes
                            started_nodes = 0
                            for instance in instances:
                                status = instance.status
                                if status == STARTING or status == RUNNING:
                                    started_nodes += 1
                            if started_nodes == len(instance_id_sets):
                                # All in starting or running status
                                with cli_logger.group(
                                        "Launched {} nodes", count, _tags=cli_logger_tags):
                                    for instance in instances:
                                        created_nodes_dict[instance.instance_id] = instance
                                        cli_logger.print(
                                            "Launched instance {}",
                                            instance.instance_id,
                                            _tags=dict(
                                                type=instance.instance_type,
                                                status=instance.status))
                                break
                        if status_attempt >= int(MAX_CREATE_STATUS_ATTEMPTS / 2):
                            # Show a message only for abnormal cases
                            cli_logger.warning(
                                "Bad status for node creation after {} attempts.", status_attempt)
                        status_attempt += 1
                        time.sleep(STARTING_NODE_DELAY)
                    break
                except TeaException as e:
                    if attempt == max_tries:
                        cli_logger.abort("Failed to launch instances. Max attempts exceeded.", str(e))
                    if e.code == "InvalidDiskCategory.NotSupported":
                        cli_logger.warning("Create instances attempt failed: the specified disk category"
                                           " is not supported. Retrying...")
                    elif e.code =="InvalidResourceType.NotSupported":
                        cli_logger.warning("Create instances attempt failed: {}. Retrying...", e.message)
                    else:
                        cli_logger.warning("Create instances attempt failed. Retrying...", str(e))
                    vswitch_idx += 1
                except Exception as e:
                    if attempt == max_tries:
                        cli_logger.abort("Failed to launch instances. Max attempts exceeded.", str(e))
                    else:
                        cli_logger.warning("Create instances attempt failed: {}. Retrying...", str(e))
                    # Launch failure may be due to instance type availability in
                    # the given AZ
                    vswitch_idx += 1

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

    def prepare_config_for_head(
            self, cluster_config: Dict[str, Any], remote_config: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a new remote cluster config with custom configs for head node.
        The cluster config may also be updated for setting up the head"""

        head_node_type = cluster_config["head_node_type"]
        head_node_config = cluster_config["available_node_types"][head_node_type]["node_config"]
        ram_role_name = head_node_config.get("RamRoleName")
        if ram_role_name:
            remote_config["provider"]["ram_role_name"] = ram_role_name

        # Since the head will use the instance profile and role to access cloud,
        # remove the client credentials from config
        if "aliyun_credentials" in remote_config["provider"]:
            remote_config["provider"].pop("aliyun_credentials", None)
        
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
