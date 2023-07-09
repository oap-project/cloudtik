from typing import Any, Optional, Dict
import copy
import logging
import operator
import threading
import traceback
import time

from cloudtik.core.tags import (CLOUDTIK_TAG_LAUNCH_CONFIG, CLOUDTIK_TAG_NODE_STATUS,
                                CLOUDTIK_TAG_NODE_KIND, CLOUDTIK_TAG_NODE_NAME,
                                CLOUDTIK_TAG_USER_NODE_TYPE, STATUS_UNINITIALIZED,
                                NODE_KIND_WORKER)
from cloudtik.core._private.prometheus_metrics import ClusterPrometheusMetrics
from cloudtik.core._private.utils import hash_launch_conf

logger = logging.getLogger(__name__)


class BaseNodeLauncher:
    """Launches nodes synchronously in the foreground."""

    def __init__(self,
                 provider,
                 pending,
                 event_summarizer,
                 session_name: Optional[str] = None,
                 prometheus_metrics=None,
                 node_types=None,
                 index=None,
                 *args,
                 **kwargs):
        self.pending = pending
        self.prometheus_metrics = prometheus_metrics or ClusterPrometheusMetrics(
            session_name=session_name)
        self.provider = provider
        self.node_types = node_types
        self.index = str(index) if index is not None else ""
        self.event_summarizer = event_summarizer

    def _launch_node(self, config: Dict[str, Any], count: int,
                     node_type: Optional[str]):
        if self.node_types:
            assert node_type, node_type

        launch_config = {}
        if node_type:
            launch_config.update(
                config["available_node_types"][node_type]["node_config"])
        resources = copy.deepcopy(
            config["available_node_types"][node_type]["resources"])
        launch_hash = hash_launch_conf(launch_config, config["auth"])
        self.log("Launching {} nodes, type {}.".format(count, node_type))
        node_config = {}
        node_tags = {
            CLOUDTIK_TAG_NODE_NAME: "cloudtik-{}-worker".format(config["cluster_name"]),
            CLOUDTIK_TAG_NODE_KIND: NODE_KIND_WORKER,
            CLOUDTIK_TAG_NODE_STATUS: STATUS_UNINITIALIZED,
            CLOUDTIK_TAG_LAUNCH_CONFIG: launch_hash,
        }
        # A custom node type is specified; set the tag in this case, and also
        # merge the configs. We merge the configs instead of overriding, so
        # that the bootstrapped per-cloud properties are preserved.
        # TODO: this logic is duplicated in cluster_operator.py (keep in sync)
        if node_type:
            node_tags[CLOUDTIK_TAG_USER_NODE_TYPE] = node_type
            node_config.update(launch_config)
        launch_start_time = time.time()
        self.provider.create_node_with_resources(node_config, node_tags, count,
                                                 resources)
        launch_time = time.time() - launch_start_time
        for _ in range(count):
            # Note: when launching multiple nodes we observe the time it
            # took all nodes to launch for each node. For example, if 4
            # nodes were created in 25 seconds, we would observe the 25
            # second create time 4 times.
            self.prometheus_metrics.worker_create_node_time.observe(launch_time)
        self.prometheus_metrics.started_nodes.inc(count)

    def launch_node(self, config: Dict[str, Any], count: int, node_type: Optional[str]):
        self.log("Got {} nodes to launch.".format(count))
        try:
            self._launch_node(config, count, node_type)
        except Exception:
            self.prometheus_metrics.node_launch_exceptions.inc()
            self.prometheus_metrics.failed_create_nodes.inc(count)
            self.event_summarizer.add(
                "Failed to launch {} nodes of type " + node_type + ".",
                quantity=count,
                aggregate=operator.add)
            # Log traceback from failed node creation only once per minute
            # to avoid spamming driver logs with tracebacks.
            self.event_summarizer.add_once_per_interval(
                message="Node creation failed. See the traceback below."
                        " See cluster scaler logs for further details.\n"
                        f"{traceback.format_exc()}",
                key="Failed to create node.",
                interval_s=60)
            logger.exception("Launch failed")
        finally:
            self.pending.dec(node_type, count)
            self.prometheus_metrics.pending_nodes.set(self.pending.value)

    def log(self, statement):
        # launcher_class is "BaseNodeLauncher", or "NodeLauncher" if called
        # from that subclass.
        launcher_class: str = type(self).__name__
        prefix = "{}{}:".format(launcher_class, self.index)
        logger.info(prefix + " {}".format(statement))


class NodeLauncher(BaseNodeLauncher, threading.Thread):
    """Launches nodes asynchronously in the background."""

    def __init__(self,
                 provider,
                 queue,
                 pending,
                 event_summarizer,
                 session_name: Optional[str] = None,
                 prometheus_metrics=None,
                 node_types=None,
                 index=None,
                 *thread_args,
                 **thread_kwargs):
        self.queue = queue
        BaseNodeLauncher.__init__(
            self,
            provider=provider,
            pending=pending,
            event_summarizer=event_summarizer,
            session_name=session_name,
            prometheus_metrics=prometheus_metrics,
            node_types=node_types,
            index=index,
        )
        threading.Thread.__init__(self, *thread_args, **thread_kwargs)

    def run(self):
        """Launches nodes in a background thread.

        Overrides threading.Thread.run().
        NodeLauncher.start() executes this loop in a background thread.
        """
        while True:
            config, count, node_type = self.queue.get()
            # launch_node is implemented in BaseNodeLauncher
            self.launch_node(config, count, node_type)
