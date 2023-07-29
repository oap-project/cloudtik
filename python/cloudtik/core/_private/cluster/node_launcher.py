from typing import Any, Optional, Dict
import copy
import logging
import operator
import threading
import traceback
import time

from cloudtik.core._private.cluster.node_availability_tracker import NodeAvailabilityTracker
from cloudtik.core.node_provider import NodeLaunchException
from cloudtik.core.tags import (CLOUDTIK_TAG_LAUNCH_CONFIG, CLOUDTIK_TAG_NODE_STATUS,
                                CLOUDTIK_TAG_NODE_KIND, CLOUDTIK_TAG_NODE_NAME,
                                CLOUDTIK_TAG_USER_NODE_TYPE, STATUS_UNINITIALIZED,
                                NODE_KIND_WORKER, CLOUDTIK_TAG_QUORUM_ID, CLOUDTIK_TAG_QUORUM_JOIN,
                                QUORUM_JOIN_STATUS_INIT)
from cloudtik.core._private.prometheus_metrics import ClusterPrometheusMetrics
from cloudtik.core._private.utils import hash_launch_conf

logger = logging.getLogger(__name__)


LAUNCH_ARGS_QUORUM_ID = "quorum_id"


class BaseNodeLauncher:
    """Launches nodes synchronously in the foreground."""

    def __init__(self,
                 provider,
                 pending,
                 event_summarizer,
                 node_availability_tracker: NodeAvailabilityTracker,
                 session_name: Optional[str] = None,
                 prometheus_metrics=None,
                 node_types=None,
                 index=None,
                 *args,
                 **kwargs):
        self.pending = pending
        self.event_summarizer = event_summarizer
        self.node_availability_tracker = node_availability_tracker
        self.prometheus_metrics = prometheus_metrics or ClusterPrometheusMetrics(
            session_name=session_name)
        self.provider = provider
        self.node_types = node_types
        self.index = str(index) if index is not None else ""

    def _launch_node(
            self, config: Dict[str, Any], count: int, node_type: str,
            launch_args: Dict[str, Any]):
        if self.node_types:
            assert node_type, node_type

        launch_config = {}
        if node_type:
            launch_config.update(
                config["available_node_types"][node_type]["node_config"])
        resources = copy.deepcopy(
            config["available_node_types"][node_type]["resources"])
        launch_hash = hash_launch_conf(launch_config, config["auth"])
        node_config = {}
        node_tags = {
            CLOUDTIK_TAG_NODE_NAME: "cloudtik-{}-worker".format(config["cluster_name"]),
            CLOUDTIK_TAG_NODE_KIND: NODE_KIND_WORKER,
            CLOUDTIK_TAG_NODE_STATUS: STATUS_UNINITIALIZED,
            CLOUDTIK_TAG_LAUNCH_CONFIG: launch_hash,
        }
        # if quorum_id is provided, it is joining an existing quorum
        quorum_id = launch_args.get(LAUNCH_ARGS_QUORUM_ID)
        if quorum_id:
            node_tags[CLOUDTIK_TAG_QUORUM_ID] = quorum_id
            node_tags[CLOUDTIK_TAG_QUORUM_JOIN] = QUORUM_JOIN_STATUS_INIT

        # A custom node type is specified; set the tag in this case, and also
        # merge the configs. We merge the configs instead of overriding, so
        # that the bootstrapped per-cloud properties are preserved.
        # TODO: this logic is duplicated in cluster_operator.py (keep in sync)
        if node_type:
            node_tags[CLOUDTIK_TAG_USER_NODE_TYPE] = node_type
            node_config.update(launch_config)

        node_launch_start_time = time.time()

        error_msg = None
        full_exception = None
        try:
            self.provider.create_node_with_resources(
                node_config, node_tags, count, resources
            )
        except NodeLaunchException as node_launch_exception:
            self.node_availability_tracker.update_node_availability(
                node_type, int(node_launch_start_time), node_launch_exception
            )

            if node_launch_exception.src_exc_info is not None:
                full_exception = "\n".join(
                    traceback.format_exception(*node_launch_exception.src_exc_info)
                )

            error_msg = (
                f"Failed to launch {{}} node(s) of type {node_type}. "
                f"({node_launch_exception.category}): "
                f"{node_launch_exception.description}"
            )
        except Exception:
            error_msg = f"Failed to launch {{}} node(s) of type {node_type}."
            full_exception = traceback.format_exc()
        else:
            # Record some metrics/observability information when a node is launched.
            launch_time = time.time() - node_launch_start_time
            for _ in range(count):
                # Note: when launching multiple nodes we observe the time it
                # took all nodes to launch for each node. For example, if 4
                # nodes were created in 25 seconds, we would observe the 25
                # second create time 4 times.
                self.prometheus_metrics.worker_create_node_time.observe(launch_time)
            self.prometheus_metrics.started_nodes.inc(count)
            self.node_availability_tracker.update_node_availability(
                node_type=node_type,
                timestamp=int(node_launch_start_time),
                node_launch_exception=None,
            )

        if error_msg is not None:
            self.event_summarizer.add(
                error_msg,
                quantity=count,
                aggregate=operator.add,
            )
            self.log(error_msg)
            self.prometheus_metrics.node_launch_exceptions.inc()
            self.prometheus_metrics.failed_create_nodes.inc(count)
        else:
            self.log("Launching {} nodes, type {}.".format(count, node_type))
            self.event_summarizer.add(
                "Adding {} node(s) of type " + str(node_type) + ".",
                quantity=count,
                aggregate=operator.add,
            )

        if full_exception is not None:
            self.log(full_exception)

    def launch_node(
            self, config: Dict[str, Any], count: int, node_type: str,
            launch_args: Dict[str, Any]):
        self.log("Got {} nodes to launch, type {}.".format(count, node_type))
        self._launch_node(config, count, node_type, launch_args)
        self.pending.dec(node_type, count)

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
                 node_availability_tracker: NodeAvailabilityTracker,
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
            node_availability_tracker=node_availability_tracker,
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
            config, count, node_type, launch_args = self.queue.get()
            # launch_node is implemented in BaseNodeLauncher
            self.launch_node(config, count, node_type, launch_args)
