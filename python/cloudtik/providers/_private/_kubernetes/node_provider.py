import copy
import logging
import time
from typing import Dict, Any
from uuid import uuid4

from kubernetes.client.rest import ApiException

from cloudtik.core._private.call_context import CallContext
from cloudtik.core._private.utils import is_use_internal_ip, _is_use_internal_ip
from cloudtik.core.node_provider import NodeProvider
from cloudtik.core.tags import NODE_KIND_HEAD
from cloudtik.core.tags import CLOUDTIK_TAG_CLUSTER_NAME
from cloudtik.core.tags import CLOUDTIK_TAG_NODE_KIND

from cloudtik.core._private.command_executor import KubernetesCommandExecutor

from cloudtik.providers._private._kubernetes import core_api, log_prefix, \
    networking_api
from cloudtik.providers._private._kubernetes.config import bootstrap_kubernetes, \
    post_prepare_kubernetes, _add_service_name_to_service_port, head_service_selector, \
    bootstrap_kubernetes_for_api, cleanup_kubernetes_cluster, with_kubernetes_environment_variables, get_head_hostname, \
    get_worker_hostname, prepare_kubernetes_config, get_head_external_service_address, _get_node_info, \
    _get_node_public_ip, get_default_kubernetes_cloud_storage
from cloudtik.providers._private._kubernetes.utils import to_label_selector, \
    create_and_configure_pvc_for_pod, delete_persistent_volume_claims, get_pod_persistent_volume_claims, \
    delete_persistent_volume_claims_by_name
from cloudtik.providers._private.utils import validate_config_dict

logger = logging.getLogger(__name__)

MAX_TAG_RETRIES = 3
DELAY_BEFORE_TAG_RETRY = .5


class KubernetesNodeProvider(NodeProvider):
    def __init__(self, provider_config, cluster_name):
        NodeProvider.__init__(self, provider_config, cluster_name)
        self.cluster_name = cluster_name
        self.namespace = provider_config["namespace"]

    def with_environment_variables(self, node_type_config: Dict[str, Any], node_id: str):
        """Export necessary environment variables for running node commands"""
        return with_kubernetes_environment_variables(self.provider_config, node_type_config, node_id)

    def non_terminated_nodes(self, tag_filters):
        # Match pods that are in the 'Pending' or 'Running' phase.
        # Unfortunately there is no OR operator in field selectors, so we
        # have to match on NOT any of the other phases.
        field_selector = ",".join([
            "status.phase!=Failed",
            "status.phase!=Unknown",
            "status.phase!=Succeeded",
            "status.phase!=Terminating",
        ])

        tag_filters[CLOUDTIK_TAG_CLUSTER_NAME] = self.cluster_name
        label_selector = to_label_selector(tag_filters)
        pod_list = core_api().list_namespaced_pod(
            self.namespace,
            field_selector=field_selector,
            label_selector=label_selector)

        # Don't return pods marked for deletion,
        # i.e. pods with non-null metadata.DeletionTimestamp.
        return [
            pod.metadata.name for pod in pod_list.items
            if pod.metadata.deletion_timestamp is None
        ]

    def get_node_info(self, node_id):
        pod = core_api().read_namespaced_pod(node_id, self.namespace)
        return _get_node_info(pod, self.provider_config, self.namespace, self.cluster_name)

    def is_running(self, node_id):
        pod = core_api().read_namespaced_pod(node_id, self.namespace)
        return pod.status.phase == "Running"

    def is_terminated(self, node_id):
        pod = core_api().read_namespaced_pod(node_id, self.namespace)
        return pod.status.phase not in ["Running", "Pending"]

    def node_tags(self, node_id):
        pod = core_api().read_namespaced_pod(node_id, self.namespace)
        return pod.metadata.labels

    def external_ip(self, node_id):
        if _is_use_internal_ip(self.provider_config):
            return None
        # For kubernetes, this is not really the external IP of a node,
        # but the external address of the service connected to the head node
        # check node is head
        tags = self.node_tags(node_id)
        return _get_node_public_ip(tags, self.namespace, self.cluster_name)

    def internal_ip(self, node_id):
        pod = core_api().read_namespaced_pod(node_id, self.namespace)
        return pod.status.pod_ip

    def get_node_id(self, ip_address, use_internal_ip=True) -> str:
        if not use_internal_ip:
            raise ValueError("Must use internal IPs with Kubernetes.")
        return super().get_node_id(ip_address, use_internal_ip=use_internal_ip)

    def set_node_tags(self, node_ids, tags):
        for _ in range(MAX_TAG_RETRIES - 1):
            try:
                self._set_node_tags(node_ids, tags)
                return
            except ApiException as e:
                if e.status == 409:
                    logger.info(log_prefix + "Caught a 409 error while setting"
                                " node tags. Retrying...")
                    time.sleep(DELAY_BEFORE_TAG_RETRY)
                    continue
                else:
                    raise
        # One more try
        self._set_node_tags(node_ids, tags)

    def _set_node_tags(self, node_id, tags):
        pod = core_api().read_namespaced_pod(node_id, self.namespace)
        pod.metadata.labels.update(tags)
        core_api().patch_namespaced_pod(node_id, self.namespace, pod)

    def create_node(self, node_config, tags, count):
        conf = copy.deepcopy(node_config)
        data_disks = conf.get("dataDisks")
        pod_spec = conf["pod"]
        service_spec = conf.get("service")
        ingress_spec = conf.get("ingress")
        node_uuid = str(uuid4())
        tags[CLOUDTIK_TAG_CLUSTER_NAME] = self.cluster_name
        tags["cloudtik-node-uuid"] = node_uuid
        pod_spec["metadata"]["namespace"] = self.namespace
        if "labels" in pod_spec["metadata"]:
            pod_spec["metadata"]["labels"].update(tags)
        else:
            pod_spec["metadata"]["labels"] = tags

        # Allow Operator-configured service to access the head node.
        if tags[CLOUDTIK_TAG_NODE_KIND] == NODE_KIND_HEAD:
            head_selector = head_service_selector(self.cluster_name)
            pod_spec["metadata"]["labels"].update(head_selector)

        logger.debug(log_prefix + "calling create_namespaced_pod "
                                  "(count={}).".format(count))

        new_nodes = []
        for _ in range(count):
            _pod_spec = copy.deepcopy(pod_spec)
            # Generate a random hostname
            _pod_spec["spec"]["hostname"] = get_head_hostname() if (
                    tags[CLOUDTIK_TAG_NODE_KIND] == NODE_KIND_HEAD) else get_worker_hostname()
            created_pvcs = create_and_configure_pvc_for_pod(
                _pod_spec, data_disks, self.cluster_name, self.namespace)
            try:
                pod = core_api().create_namespaced_pod(self.namespace, _pod_spec)
                new_nodes.append(pod)
            except ApiException:
                logger.error("Error happened when creating the pod. Try clean up its PVCs...")
                delete_persistent_volume_claims(created_pvcs, self.namespace)
                raise

        new_svcs = []
        if service_spec is not None:
            logger.debug(log_prefix + "calling create_namespaced_service "
                                      "(count={}).".format(count))

            for new_node in new_nodes:
                metadata = service_spec.get("metadata", {})
                metadata["name"] = new_node.metadata.name
                service_spec["metadata"] = metadata
                service_spec["spec"]["selector"] = {"cloudtik-node-uuid": node_uuid}
                svc = core_api().create_namespaced_service(
                    self.namespace, service_spec)
                new_svcs.append(svc)

        if ingress_spec is not None:
            logger.debug(log_prefix + "calling create_namespaced_ingress "
                         "(count={}).".format(count))
            for new_svc in new_svcs:
                metadata = ingress_spec.get("metadata", {})
                metadata["name"] = new_svc.metadata.name
                ingress_spec["metadata"] = metadata
                ingress_spec = _add_service_name_to_service_port(
                    ingress_spec, new_svc.metadata.name)
                networking_api().create_namespaced_ingress(self.namespace, ingress_spec)

    def terminate_node(self, node_id):
        logger.debug(log_prefix + f"Deleting PVCs for pod: {node_id}.")

        pod_pvcs = get_pod_persistent_volume_claims(node_id, self.cluster_name, self.namespace)

        logger.debug(log_prefix + "Calling delete_namespaced_pod")
        try:
            core_api().delete_namespaced_pod(node_id, self.namespace)
        except ApiException as e:
            if e.status == 404:
                logger.warning(log_prefix + f"Tried to delete pod {node_id},"
                               " but the pod was not found (404).")
            else:
                raise

        try:
            delete_persistent_volume_claims_by_name(pod_pvcs, self.namespace)
        except ApiException:
            logger.warning(log_prefix + f"Error happened when deleting PVCs of pod {node_id}.")
            pass

        try:
            core_api().delete_namespaced_service(node_id, self.namespace)
        except ApiException:
            pass

        try:
            networking_api().delete_namespaced_ingress(
                node_id,
                self.namespace,
            )
        except ApiException:
            pass

    def terminate_nodes(self, node_ids):
        for node_id in node_ids:
            self.terminate_node(node_id)

    def get_command_executor(self,
                             call_context: CallContext,
                             log_prefix,
                             node_id,
                             auth_config,
                             cluster_name,
                             process_runner,
                             use_internal_ip,
                             docker_config=None):
        return KubernetesCommandExecutor(call_context, log_prefix, self.namespace,
                                         node_id, auth_config, process_runner)

    def prepare_for_head_node(
            self, cluster_config: Dict[str, Any], remote_config: Dict[str, Any]) -> Dict[str, Any]:
        # rsync ssh_public_key to head node authorized_keys,
        if not is_use_internal_ip(cluster_config):
            cluster_config["file_mounts"].update({
                "~/.ssh/authorized_keys": cluster_config["auth"]["ssh_public_key"]
            })
        return remote_config

    def get_default_cloud_storage(self):
        """Return the managed cloud storage if configured."""
        return get_default_kubernetes_cloud_storage(self.provider_config)

    @staticmethod
    def bootstrap_config(cluster_config):
        return bootstrap_kubernetes(cluster_config)

    def cleanup_cluster(self, cluster_config: Dict[str, Any]):
        """Finalize the cluster by cleanup additional resources other than the nodes."""
        cleanup_kubernetes_cluster(cluster_config, self.cluster_name, self.namespace)

    @staticmethod
    def bootstrap_config_for_api(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        return bootstrap_kubernetes_for_api(cluster_config)

    @staticmethod
    def prepare_config(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        return prepare_kubernetes_config(cluster_config)


    @staticmethod
    def post_prepare(cluster_config):
        """Fills out missing fields after the user config is merged with defaults and before validate"""
        return post_prepare_kubernetes(cluster_config)

    @staticmethod
    def validate_config(
            provider_config: Dict[str, Any]) -> None:
        config_dict = {
            "namespace": provider_config.get("namespace"),
            }

        validate_config_dict(provider_config["type"], config_dict)
