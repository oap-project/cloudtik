import copy
from typing import Any, Dict, List
from functools import wraps
from threading import RLock
import time
import logging

from cloudtik.core._private.cli_logger import cli_logger
from cloudtik.core.node_provider import NodeProvider

from cloudtik.providers._private.gcp.config import (
    construct_clients_from_provider_config, get_node_type, verify_gcs_storage, bootstrap_gcp)

# The logic has been abstracted away here to allow for different GCP resources
# (API endpoints), which can differ widely, making it impossible to use
# the same logic for everything.
from cloudtik.providers._private.gcp.node import (
    GCPResource, GCPNode, GCPCompute, GCPTPU, GCPNodeType,
    INSTANCE_NAME_MAX_LEN, INSTANCE_NAME_UUID_LEN)

from cloudtik.providers._private.gcp.utils import get_gcs_config
from cloudtik.providers._private.utils import validate_config_dict

logger = logging.getLogger(__name__)


def _retry(method, max_tries=5, backoff_s=1):
    """Retry decorator for methods of GCPNodeProvider.

    Upon catching BrokenPipeError, API clients are rebuilt and
    decorated methods are retried.

    Work-around for issue #16072.
    Based on https://github.com/kubeflow/pipelines/pull/5250/files.
    """

    @wraps(method)
    def method_with_retries(self, *args, **kwargs):
        try_count = 0
        while try_count < max_tries:
            try:
                return method(self, *args, **kwargs)
            except BrokenPipeError:
                logger.warning("Caught a BrokenPipeError. Retrying.")
                try_count += 1
                if try_count < max_tries:
                    self._construct_clients()
                    time.sleep(backoff_s)
                else:
                    raise

    return method_with_retries


class GCPNodeProvider(NodeProvider):
    def __init__(self, provider_config: dict, cluster_name: str):
        NodeProvider.__init__(self, provider_config, cluster_name)
        self.lock = RLock()
        self._construct_clients()

        # Cache of node objects from the last nodes() call. This avoids
        # excessive DescribeInstances requests.
        self.cached_nodes: Dict[str, GCPNode] = {}

    def with_provider_environment_variables(self):
        return get_gcs_config(self.provider_config)

    def _construct_clients(self):
        _, _, compute, tpu = construct_clients_from_provider_config(
            self.provider_config)

        # Dict of different resources provided by GCP.
        # At this moment - Compute and TPUs
        self.resources: Dict[GCPNodeType, GCPResource] = {}

        # Compute is always required
        self.resources[GCPNodeType.COMPUTE] = GCPCompute(
            compute, self.provider_config["project_id"],
            self.provider_config["availability_zone"], self.cluster_name)

        # if there are no TPU nodes defined in config, tpu will be None.
        if tpu is not None:
            self.resources[GCPNodeType.TPU] = GCPTPU(
                tpu, self.provider_config["project_id"],
                self.provider_config["availability_zone"], self.cluster_name)

    def _get_resource_depending_on_node_name(self,
                                             node_name: str) -> GCPResource:
        """Return the resource responsible for the node, based on node_name.

        This expects the name to be in format '[NAME]-[UUID]-[TYPE]',
        where [TYPE] is either 'compute' or 'tpu' (see ``GCPNodeType``).
        """
        return self.resources[GCPNodeType.name_to_type(node_name)]

    @_retry
    def non_terminated_nodes(self, tag_filters: dict):
        with self.lock:
            instances = []

            for resource in self.resources.values():
                node_instances = resource.list_instances(tag_filters)
                instances += node_instances

            # Note: All the operations use "name" as the unique instance id
            self.cached_nodes = {i["name"]: i for i in instances}
            return [i["name"] for i in instances]

    def is_running(self, node_id: str):
        with self.lock:
            node = self._get_cached_node(node_id)
            return node.is_running()

    def is_terminated(self, node_id: str):
        with self.lock:
            node = self._get_cached_node(node_id)
            return node.is_terminated()

    def node_tags(self, node_id: str):
        with self.lock:
            node = self._get_cached_node(node_id)
            return node.get_labels()

    def get_node_info(self, node_id):
        node = self._get_cached_node(node_id)
        node_info = {"node_id": node["id"],
                     "instance_type": node["machineType"].split("/")[-1],
                     "private_ip": node.get_internal_ip(),
                     "public_ip": node.get_external_ip(),
                     "instance_status": node["status"]}
        node_info.update(self.node_tags(node_id))
        return node_info

    @_retry
    def set_node_tags(self, node_id: str, tags: dict):
        with self.lock:
            labels = tags
            node = self._get_node(node_id)

            resource = self._get_resource_depending_on_node_name(node_id)

            result = resource.set_labels(node=node, labels=labels)

            return result

    def external_ip(self, node_id: str):
        with self.lock:
            node = self._get_cached_node(node_id)

            ip = node.get_external_ip()
            if ip is None:
                node = self._get_node(node_id)
                ip = node.get_external_ip()

            return ip

    def internal_ip(self, node_id: str):
        with self.lock:
            node = self._get_cached_node(node_id)

            ip = node.get_internal_ip()
            if ip is None:
                node = self._get_node(node_id)
                ip = node.get_internal_ip()

            return ip

    @_retry
    def create_node(self, base_config: dict, tags: dict, count: int) -> None:
        with self.lock:
            labels = tags  # gcp uses "labels" instead of aws "tags"

            node_type = get_node_type(base_config)
            resource = self.resources[node_type]

            resource.create_instances(base_config, labels, count)

    @_retry
    def terminate_node(self, node_id: str):
        with self.lock:
            resource = self._get_resource_depending_on_node_name(node_id)
            result = resource.delete_instance(node_id=node_id, )
            return result

    @_retry
    def _get_node(self, node_id: str) -> GCPNode:
        self.non_terminated_nodes({})  # Side effect: updates cache

        with self.lock:
            if node_id in self.cached_nodes:
                return self.cached_nodes[node_id]

            resource = self._get_resource_depending_on_node_name(node_id)
            instance = resource.get_instance(node_id=node_id)

            return instance

    def _get_cached_node(self, node_id: str) -> GCPNode:
        if node_id in self.cached_nodes:
            return self.cached_nodes[node_id]

        return self._get_node(node_id)

    @staticmethod
    def bootstrap_config(cluster_config):
        bootstrap_gcp(cluster_config)

    @staticmethod
    def get_cluster_resources(
            cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Fills out spark executor resource for available_node_types."""
        if "available_node_types" not in cluster_config:
            return cluster_config

        _, _, compute, tpu = construct_clients_from_provider_config(
            cluster_config)

        response = compute.machineTypes().list(
            project=cluster_config["provider"]["project_id"],
            zone=cluster_config["provider"]["availability_zone"],
        ).execute()

        instances_list = response.get("items", [])
        instances_dict = {
            instance["name"]: instance
            for instance in instances_list
        }
        available_node_types = cluster_config["available_node_types"]
        head_node_type = cluster_config["head_node_type"]
        cluster_resource = {}
        for node_type in available_node_types:
            instance_type = available_node_types[node_type]["node_config"][
                "machineType"]
            if instance_type in instances_dict:
                memory_total = instances_dict[instance_type]["memoryMb"]
                if node_type != head_node_type:
                    cluster_resource["worker_memory"] = memory_total
                    cluster_resource["worker_cpu"] = instances_dict[instance_type]["guestCpus"]
                else:
                    cluster_resource["head_memory"] = memory_total
        return cluster_resource

    @staticmethod
    def validate_config(
            provider_config: Dict[str, Any]) -> None:
        config_dict = {"project_id": provider_config.get("project_id"),
                       "availability_zone": provider_config.get("availability_zone")}

        validate_config_dict(provider_config["type"], config_dict)

    @staticmethod
    def validate_storage_config(
            provider_config: Dict[str, Any]) -> None:
        config_dict = {"gcs.bucket": provider_config.get("gcp_cloud_storage", {}).get("gcs.bucket"),
                       "gcs.service.account.client.email": provider_config.get("gcp_cloud_storage", {}).get(
                           "gcs.service.account.client.email"),
                       "gcs.service.account.private.key.id": provider_config.get("gcp_cloud_storage", {}).get(
                           "gcs.service.account.private.key.id"),
                       "gcs.service.account.private.key": provider_config.get("gcp_cloud_storage", {}).get(
                           "gcs.service.account.private.key")}

        validate_config_dict(provider_config["type"], config_dict)

        verify_cloud_storage = provider_config.get("verify_cloud_storage", True)
        if verify_cloud_storage:
            cli_logger.verbose("Verifying GCS storage configurations...")
            verify_gcs_storage(provider_config)
            cli_logger.verbose("Successfully verified GCS storage configurations.")
