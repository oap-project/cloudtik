import json
import logging
import time
import traceback
from typing import Optional, Dict

from cloudtik.core._private.cluster.cluster_metrics import ClusterMetrics
from cloudtik.core._private.cluster.event_summarizer import EventSummarizer
from cloudtik.core._private.constants import CLOUDTIK_RESOURCE_REQUESTS
from cloudtik.core._private.state.kv_store import kv_initialized, kv_get
from cloudtik.core._private.state.scaling_state import ScalingStateClient
from cloudtik.core._private.state.state_utils import NODE_STATE_NODE_IP

logger = logging.getLogger(__name__)

MAX_FAILURES_FOR_LOGGING = 16


class ClusterMetricsUpdater:
    def __init__(self,
                 cluster_metrics: ClusterMetrics,
                 event_summarizer: Optional[EventSummarizer],
                 scaling_state_client: ScalingStateClient):
        self.cluster_metrics = cluster_metrics
        self.event_summarizer = event_summarizer
        self.scaling_state_client = scaling_state_client
        self.last_avail_resources = None
        self.cluster_metrics_failures = 0

    def update(self):
        self._update_cluster_metrics()
        self._update_resource_requests()
        self._update_event_summary()

    def _update_cluster_metrics(self):
        try:
            heartbeat_nodes = {}
            self._update_node_heartbeats(heartbeat_nodes)
            self._update_scaling_metrics(heartbeat_nodes)

            # reset if there is a success
            self.cluster_metrics_failures = 0
        except Exception as e:
            if self.cluster_metrics_failures == 0 or self.cluster_metrics_failures == MAX_FAILURES_FOR_LOGGING:
                # detailed form
                error = traceback.format_exc()
                logger.exception(f"Load metrics update failed with the following error:\n{error}")
            elif self.cluster_metrics_failures < MAX_FAILURES_FOR_LOGGING:
                # short form
                logger.exception(f"Load metrics update failed with the following error:{str(e)}")

            if self.cluster_metrics_failures == MAX_FAILURES_FOR_LOGGING:
                logger.exception(f"The above error has been showed consecutively"
                                 f" for {self.cluster_metrics_failures} times. Stop showing.")

            self.cluster_metrics_failures += 1

    def _update_node_heartbeats(
            self, heartbeat_nodes: Dict[str, str]):
        cluster_heartbeat_state = self.scaling_state_client.get_cluster_heartbeat_state(timeout=60)
        for node_id, node_heartbeat_state in cluster_heartbeat_state.node_heartbeat_states.items():
            ip = node_heartbeat_state.node_ip
            last_heartbeat_time = node_heartbeat_state.last_heartbeat_time
            heartbeat_nodes[node_id] = ip
            self.cluster_metrics.update_heartbeat(ip, node_id, last_heartbeat_time)

    def _update_scaling_metrics(
            self, heartbeat_nodes: Dict[str, str]):
        """Fetches resource usage data from control state and updates load metrics."""
        scaling_state = self.scaling_state_client.get_scaling_state(timeout=60)
        self.cluster_metrics.update_autoscaling_instructions(
            scaling_state.autoscaling_instructions)

        # If there is no scaling metrics for nodes, we still need to make sure
        # to set node last used status so that the idle nodes can be killed
        # We will depend on the timeout of a previous node resource states
        # if there is no latest reporting from that node
        node_resource_states = scaling_state.node_resource_states
        if node_resource_states is None:
            node_resource_states = {}

        for node_id, node_resource_state in node_resource_states.items():
            ip = node_resource_state[NODE_STATE_NODE_IP]
            resource_time = node_resource_state["resource_time"]
            # Node resource state
            total_resources = node_resource_state["total_resources"]
            available_resources = node_resource_state["available_resources"]
            resource_load = node_resource_state["resource_load"]

            self.cluster_metrics.update_node_resources(
                ip, node_id, resource_time,
                total_resources, available_resources, resource_load)

        # All the nodes that shows in heartbeat but not in reported with node resources
        # We consider it is idle
        resource_time = time.time()
        for node_id, ip in heartbeat_nodes.items():
            if node_id in node_resource_states:
                continue
            total_resources = {}
            available_resources = {}
            resource_load = {}
            self.cluster_metrics.update_node_resources(
                ip, node_id, resource_time,
                total_resources, available_resources, resource_load)

    def _update_resource_requests(self):
        """Fetches resource requests from the internal KV and updates load."""
        if not kv_initialized():
            return
        data = kv_get(CLOUDTIK_RESOURCE_REQUESTS)
        if data:
            try:
                resource_requests = json.loads(data)
                request_resources = resource_requests.get("requests")
                self.cluster_metrics.set_resource_requests(
                    resource_requests["request_time"],
                    request_resources)
            except Exception:
                logger.exception("Error parsing resource requests")

    def _update_event_summary(self):
        """Report the current size of the cluster.

        To avoid log spam, only cluster size changes (CPU or GPU count change)
        are reported to the event summarizer. The event summarizer will report
        only the latest cluster size per batch.
        """
        avail_resources = self.cluster_metrics.resources_avail_summary()
        if avail_resources != self.last_avail_resources:
            self.event_summarizer.add(
                "Resized to {}.",  # e.g., Resized to 100 CPUs, 4 GPUs.
                quantity=avail_resources,
                aggregate=lambda old, new: new)
            self.last_avail_resources = avail_resources
