import asyncio
import logging
import multiprocessing as mp
import os
import threading
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import kopf
import yaml

import cloudtik.core._private.service.cloudtik_cluster_controller as cluster_controller
from cloudtik.core._private import constants, services
from cloudtik.core._private.call_context import CallContext
from cloudtik.core._private.cluster import cluster_operator
from cloudtik.core._private.cluster.cluster_operator import _get_head_node_ip
from cloudtik.providers.kubernetes.cloudtik_operator import operator_utils
from cloudtik.providers.kubernetes.cloudtik_operator.operator_utils import (
    STATUS_RECOVERING,
    STATUS_RUNNING,
    STATUS_UPDATING,
)

logger = logging.getLogger(__name__)

# Queue to process cluster status updates.
cluster_status_q = mp.Queue()  # type: mp.Queue[Optional[Tuple[str, str, str]]]


class CloudTikCluster:
    """Manages a CloudTik cluster.

    Attributes:
        config: Cluster configuration dict.
        subprocess: The subprocess used to create, update, and control the cluster.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.config["cluster_name"]
        self.namespace = self.config["provider"]["namespace"]
        self.controller_config = None

        # Make directory for configs of clusters in the namespace,
        # if the directory doesn't exist already.
        namespace_dir = operator_utils.namespace_dir(self.namespace)
        os.makedirs(namespace_dir, exist_ok=True)

        self.config_path = operator_utils.cluster_config_path(
            cluster_namespace=self.namespace, cluster_name=self.name
        )
        self.controller_config_path = operator_utils.controller_config_path(
            cluster_namespace=self.namespace, cluster_name=self.name
        )

        # Controller subprocess
        # self.subprocess is non-null iff there's an active controller subprocess
        # or a finished controller subprocess in need of cleanup.
        self.subprocess = None  # type: Optional[mp.Process]
        # Controller logs for this cluster will be prefixed by the controller
        # subprocess name:
        self.subprocess_name = ",".join([self.name, self.namespace])
        self.controller_stop_event = mp.Event()
        self.setup_logging()
        self.call_context = CallContext()
        self.call_context.set_call_from_api(True)
        # We need to restore this in the future to avoid the errors from the lack of TTY
        # self.call_context.set_allow_interactive(False)

    def create_or_update(self, restart_head: bool = False) -> None:
        """Create/update the Cluster and run the controller loop, all in a
        subprocess.

        The main function of the Operator is managing the
        subprocesses started by this method.

        Args:
            restart_head: If True, restarts head to recover from failure.
        """
        self.do_in_subprocess(self._create_or_update, args=(restart_head,))

    def _create_or_update(self, restart_head: bool = False) -> None:
        try:
            self.start_head(restart_head=restart_head)
            self.start_controller()
        except Exception:
            # Report failed cluster controller status to trigger cluster restart.
            cluster_status_q.put(
                (self.name, self.namespace, STATUS_RECOVERING)
            )
            # `status_handling_loop` will increment the
            # `status.scalerRetries` of the CR. A restart will trigger
            # at the subsequent "MODIFIED" event.
            raise

    def start_head(self, restart_head: bool = False) -> None:
        self.write_config(self.config, self.config_path)
        # Don't restart on head unless recovering from failure.
        no_restart = not restart_head
        # Create or update cluster head and record config side effects.

        self.controller_config = cluster_operator.create_or_update_cluster(
            self.config_path,
            call_context=self.call_context,
            override_min_workers=None,
            override_max_workers=None,
            no_restart=no_restart,
            restart_only=False,
            yes=True,
            no_config_cache=True,
            use_login_shells=True,
            no_controller_on_head=True,
        )
        # Write the resulting config for use by the cluster controller:
        self.write_config(self.controller_config, self.controller_config_path)

    def start_controller(self) -> None:
        """Runs the cluster controller in operator instead of on head."""
        head_pod_ip = _get_head_node_ip(self.controller_config)
        port = operator_utils.infer_head_port(self.controller_config)
        address = services.address(head_pod_ip, port)
        controller = cluster_controller.ClusterController(
            address,
            cluster_scaling_config=self.controller_config_path,
            redis_password=constants.CLOUDTIK_REDIS_DEFAULT_PASSWORD,
            prefix_cluster_info=True,
            stop_event=self.controller_stop_event,
            retry_on_failure=False,
        )
        controller.run()

    def teardown(self) -> None:
        """Attempt orderly tear-down of cluster processes before CloudTikCluster
        resource deletion."""
        self.do_in_subprocess(self._teardown, args=(), block=True)

    def _teardown(self) -> None:
        cluster_operator.teardown_cluster(
            self.config_path,
            yes=True,
            workers_only=False,
            override_cluster_name=None,
            keep_min_workers=False,
        )

    def do_in_subprocess(
        self, f: Callable[[], None], args: Tuple = (), block: bool = False
    ) -> None:
        # First stop the subprocess if it's alive
        self.clean_up_subprocess()
        # Reinstantiate process with f as target and start.
        self.subprocess = mp.Process(
            name=self.subprocess_name, target=f, args=args, daemon=True
        )
        self.subprocess.start()
        if block:
            self.subprocess.join()

    def clean_up_subprocess(self):
        """
        Clean up the monitor process.

        Executed when CR for this cluster is "DELETED".
        Executed when Autoscaling monitor is restarted.
        """

        if self.subprocess is None:
            # Nothing to clean.
            return

        # Triggers graceful stop of the monitor loop.
        self.controller_stop_event.set()
        self.subprocess.join()
        # Clears the event for subsequent runs of the monitor.
        self.controller_stop_event.clear()
        # Signal completed cleanup.
        self.subprocess = None

    def clean_up(self) -> None:
        """Executed when the CR for this cluster is "DELETED".

        The key thing is to end the monitoring subprocess.
        """
        self.teardown()
        self.clean_up_subprocess()
        self.clean_up_logging()
        self.delete_config()

    def setup_logging(self) -> None:
        """Add a log handler which appends the name and namespace of this
        cluster to the cluster's monitor logs.
        """
        self.handler = logging.StreamHandler()
        # Filter by subprocess name to get this cluster's monitor logs.
        self.handler.addFilter(lambda rec: rec.processName == self.subprocess_name)
        # Lines start with "<cluster name>,<cluster namespace>:"
        logging_format = ":".join([self.subprocess_name, constants.LOGGER_FORMAT])
        self.handler.setFormatter(logging.Formatter(logging_format))
        operator_utils.root_logger.addHandler(self.handler)

    def clean_up_logging(self) -> None:
        operator_utils.root_logger.removeHandler(self.handler)

    def set_config(self, config: Dict[str, Any]) -> None:
        self.config = config

    @staticmethod
    def write_config(config, config_path) -> None:
        """Write config to disk for use by the autoscaling monitor."""
        # Make sure to create the file to owner only rw permissions.
        with open(config_path, "w", opener=partial(os.open, mode=0o600)) as f:
            yaml.dump(config, f)

    def delete_config(self) -> None:
        self.delete_config_file(self.config_path)
        self.delete_config_file(self.controller_config_path)

    def delete_config_file(self, config_path) -> None:
        try:
            os.remove(config_path)
        except OSError:
            log_prefix = ",".join([self.name, self.namespace])
            logger.warning(
                f"{log_prefix}: config path does not exist {config_path}"
            )


@kopf.on.startup()
def start_background_worker(memo: kopf.Memo, **_):
    memo.status_handler = threading.Thread(
        target=status_handling_loop, args=(cluster_status_q,)
    )
    memo.status_handler.start()


@kopf.on.cleanup()
def stop_background_worker(memo: kopf.Memo, **_):
    cluster_status_q.put(None)
    memo.status_handler.join()


def status_handling_loop(queue: mp.Queue):
    # TODO: Status will not be set if Operator restarts after `queue.put`
    # but before `set_status`.
    while True:
        item = queue.get()
        if item is None:
            break

        cluster_name, cluster_namespace, phase = item
        try:
            operator_utils.set_status(cluster_name, cluster_namespace, phase)
        except Exception:
            log_prefix = ",".join([cluster_name, cluster_namespace])
            logger.exception(f"{log_prefix}: Error setting CloudTikCluster status.")


@kopf.on.create("cloudtikclusters")
@kopf.on.update("cloudtikclusters")
@kopf.on.resume("cloudtikclusters")
def create_or_update_cluster(body, name, namespace, logger, memo: kopf.Memo, **kwargs):
    """
    1. On creation of a CloudTikCluster resource, create the cluster.
    2. On update of a CloudTikCluster resource, update the cluster
        without restarting processes,
        unless the head's config is modified.
    3. On operator restart ("resume"), rebuild operator memo state and restart
        the cluster's controller process, without restarting head processes.
    """
    _create_or_update_cluster(body, name, namespace, memo, restart_head=False)


@kopf.on.field("cloudtikclusters", field="status.scalerRetries")
def restart_cluster(body, status, name, namespace, memo: kopf.Memo, **kwargs):
    """On increment of status.scalerRetries, restart cluster processes.

    Increment of scalerRetries happens when cluster's controller fails,
    for example due to head failure.
    """
    # Don't act on initialization of status.scalerRetries from nil to 0.
    if status.get("scalerRetries"):
        # Restart the cluster:
        _create_or_update_cluster(body, name, namespace, memo, restart_head=True)


def _create_or_update_cluster(
    custom_resource_body, name, namespace, memo, restart_head=False
):
    """Create, update, or restart the cluster described by a CloudTikCluster
    resource.

    Args:
        custom_resource_body: The body of the K8s CloudTikCluster resources describing
            a CloudTik cluster.
        name: The name of the cluster.
        namespace: The K8s namespace in which the cluster runs.
        memo: kopf memo state for this cluster.
        restart_head: Only restart cluster processes if this is true.
    """
    # Convert the CloudTikCluster custom resource to a cluster config.
    cluster_config = operator_utils.custom_resource_to_config(custom_resource_body)

    # Fetch or create the CloudTikCluster python object encapsulating cluster state.
    cloudtik_cluster = memo.get("cloudtik_cluster")
    if cloudtik_cluster is None:
        cloudtik_cluster = CloudTikCluster(cluster_config)
        memo.cloudtik_cluster = cloudtik_cluster

    # Indicate in status.phase that a "create-or-update" is in progress.
    cluster_status_q.put((name, namespace, STATUS_UPDATING))

    # Store the cluster config for use by the cluster controller.
    cloudtik_cluster.set_config(cluster_config)

    # Launch the cluster by SSHing into the pod and running
    # the initialization commands. This will not restart the cluster
    # unless there was a failure.
    cloudtik_cluster.create_or_update(restart_head=restart_head)

    # Indicate in status.phase that the head is up and the monitor is running.
    cluster_status_q.put((name, namespace, STATUS_RUNNING))


@kopf.on.delete("cloudtikclusters")
def delete_fn(memo: kopf.Memo, **kwargs):
    cloudtik_cluster = memo.get("cloudtik_cluster")
    if cloudtik_cluster is None:
        return

    cloudtik_cluster.clean_up()


def main():
    if operator_utils.NAMESPACED_OPERATOR:
        kwargs = {"namespaces": [operator_utils.OPERATOR_NAMESPACE]}
    else:
        kwargs = {"clusterwide": True}

    asyncio.run(kopf.operator(**kwargs))


if __name__ == "__main__":
    main()
