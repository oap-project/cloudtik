import copy
from typing import Any, Dict, List
from functools import wraps
from threading import RLock
import time
import logging

import googleapiclient

from cloudtik.core._private.cli_logger import cli_logger
from cloudtik.core.node_provider import NodeProvider

from cloudtik.providers._private.gcp.config import (
    construct_clients_from_provider_config, get_node_type, verify_gcs_storage, bootstrap_gcp, post_prepare_gcp)

# The logic has been abstracted away here to allow for different GCP resources
# (API endpoints), which can differ widely, making it impossible to use
# the same logic for everything.
from cloudtik.providers._private.gcp.node import (
    GCPResource, GCPNode, GCPCompute, GCPTPU, GCPNodeType,
    INSTANCE_NAME_MAX_LEN, INSTANCE_NAME_UUID_LEN)

from cloudtik.providers._private.gcp.utils import get_gcs_config, _get_node_info
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

    def with_environment_variables(self, node_type_config: Dict[str, Any], node_id: str):
        return get_gcs_config(self.provider_config, node_type_config, node_id)

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
        with self.lock:
            node = self._get_cached_node(node_id)
            return _get_node_info(node)

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
            try:
                result = resource.delete_instance(
                    node_id=node_id,
                )
            except googleapiclient.errors.HttpError as http_error:
                if http_error.resp.status == 404:
                    logger.warning(
                        f"Tried to delete the node with id {node_id} "
                        "but it was already gone."
                    )
                else:
                    raise http_error from None
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
        return bootstrap_gcp(cluster_config)

    @staticmethod
    def post_prepare(
            cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Fills out missing fields after the user config is merged with defaults and before validate"""
        return post_prepare_gcp(cluster_config)

    @staticmethod
    def validate_config(
            provider_config: Dict[str, Any]) -> None:
        config_dict = {"project_id": provider_config.get("project_id"),
                       "availability_zone": provider_config.get("availability_zone")}

        validate_config_dict(provider_config["type"], config_dict)

        if "gcp_cloud_storage" in provider_config:
            storage_config = provider_config["gcp_cloud_storage"]
            config_dict = {"gcs.bucket": storage_config.get("gcs.bucket"),
                           # The private key is no longer a must since we have role access
                           # "gcs.service.account.client.email": storage_config.get(
                           #    "gcs.service.account.client.email"),
                           # "gcs.service.account.private.key.id": storage_config.get(
                           #    "gcs.service.account.private.key.id"),
                           # "gcs.service.account.private.key": storage_config.get(
                           #    "gcs.service.account.private.key")
                           }

            validate_config_dict(provider_config["type"], config_dict)

    @staticmethod
    def verify_config(
            provider_config: Dict[str, Any]) -> None:
        verify_cloud_storage = provider_config.get("verify_cloud_storage", True)
        if ("gcp_cloud_storage" in provider_config) and verify_cloud_storage:
            cli_logger.verbose("Verifying GCS storage configurations...")
            verify_gcs_storage(provider_config)
            cli_logger.verbose("Successfully verified GCS storage configurations.")
