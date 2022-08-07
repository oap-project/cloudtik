import json
import logging
import traceback
from typing import Optional

from cloudtik.core._private.cluster.cluster_metrics import ClusterMetrics
from cloudtik.core._private.cluster.event_summarizer import EventSummarizer
from cloudtik.core._private.constants import CLOUDTIK_RESOURCE_REQUEST_CHANNEL
from cloudtik.core._private.state.kv_store import kv_initialized, kv_get
from cloudtik.core._private.state.resource_state import ResourceStateClient

logger = logging.getLogger(__name__)

MAX_FAILURES_FOR_LOGGING = 16


class ClusterMetricsUpdater:
    def __init__(self,
                 cluster_metrics: ClusterMetrics,
                 event_summarizer: Optional[EventSummarizer],
                 resource_state_client: ResourceStateClient):
        self.cluster_metrics = cluster_metrics
        self.event_summarizer = event_summarizer
        self.resource_state_client = resource_state_client
        self.last_avail_resources = None
        self.cluster_metrics_failures = 0

    def update(self):
        self._update_cluster_metrics()
        self._update_resource_requests()
        self._update_event_summary()

    def _update_cluster_metrics(self):
        try:
            self._do_update_cluster_metrics()
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

    def _do_update_cluster_metrics(self):
        self._update_node_heartbeats()
        self._update_resource_metrics()

    def _update_node_heartbeats(self):
        cluster_heartbeat_state = self.resource_state_client.get_cluster_heartbeat_state(timeout=60)
        for node_id, node_heartbeat_state in cluster_heartbeat_state.node_heartbeat_states.items():
            ip = node_heartbeat_state.node_ip
            last_heartbeat_time = node_heartbeat_state.last_heartbeat_time
            self.cluster_metrics.update_heartbeat(ip, node_id, last_heartbeat_time)

    def _update_resource_metrics(self):
        """Fetches resource usage data from control state and updates load metrics."""
        cluster_resource_state = self.resource_state_client.get_cluster_resource_state(timeout=60)
        self.cluster_metrics.update_autoscaling_instructions(
            cluster_resource_state.autoscaling_instructions)

        for node_resource_state in cluster_resource_state.node_resource_states:
            node_id = node_resource_state["node_id"]
            ip = node_resource_state["node_ip"]

            # Node resource state
            total_resources = node_resource_state["total_resources"]
            available_resources = node_resource_state["available_resources"]
            resource_load = node_resource_state["resource_load"]

            self.cluster_metrics.update_node_resources(
                ip, node_id, node_resource_state.resource_time,
                total_resources, available_resources, resource_load)

    def _update_resource_requests(self):
        """Fetches resource requests from the internal KV and updates load."""
        if not kv_initialized():
            return
        data = kv_get(CLOUDTIK_RESOURCE_REQUEST_CHANNEL)
        if data:
            try:
                resource_request = json.loads(data)
                self.cluster_metrics.set_resource_requests(resource_request)
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
