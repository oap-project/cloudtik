"""IMPORTANT: this is an experimental interface and not currently stable."""

from typing import Any, Callable, Dict, List, Optional, Union
import os

from cloudtik.core._private.call_context import CallContext
from cloudtik.core._private.cluster import cluster_operator
from cloudtik.core._private.event_system import (
    global_event_system)
from cloudtik.core._private.cli_logger import cli_logger


class Cluster:
    def __init__(self, cluster_config: Union[dict, str]) -> None:
        """Create a cluster object to operate on with this API.

        Args:
            cluster_config (Union[str, dict]): Either the config dict of the
                cluster, or a path pointing to a file containing the config.
        """
        self.cluster_config = cluster_config
        if isinstance(cluster_config, dict):
            self.config = \
                cluster_operator._bootstrap_config(cluster_config, no_config_cache=True)
        else:
            if not os.path.exists(cluster_config):
                raise ValueError("Cluster config file not found: {}".format(cluster_config))
            self.config = \
                cluster_operator._load_cluster_config(cluster_config, no_config_cache=True)

        # TODO: Each call may need its own call context
        self.call_context = CallContext()

    def start(self,
              no_restart: bool = False,
              restart_only: bool = False) -> None:
        """Create or updates an autoscaling cluster.

        Args:
            no_restart (bool): Whether to skip restarting services during the
                update. This avoids interrupting running jobs and can be used to
                dynamically adjust cluster configuration.
            restart_only (bool): Whether to skip running setup commands and only
                restart. This cannot be used with 'no-restart'.
        """
        return cluster_operator._create_or_update_cluster(
            config=self.config,
            call_context=self.call_context,
            no_restart=no_restart,
            restart_only=restart_only,
            yes=True,
            redirect_command_output=None,
            use_login_shells=True)

    def stop(self,
             workers_only: bool = False,
             keep_min_workers: bool = False) -> None:
        """Destroys all nodes of a cluster.

        Args:
            workers_only (bool): Whether to keep the head node running and only
                teardown worker nodes.
            keep_min_workers (bool): Whether to keep min_workers (as specified
                in the YAML) still running.
        """
        return cluster_operator._teardown_cluster(
            config=self.config,
            call_context=self.call_context,
            workers_only=workers_only,
            keep_min_workers=keep_min_workers)

    def exec(self,
             *,
             cmd: Optional[str] = None,
             node_ip: str = None,
             all_nodes: bool = False,
             run_env: str = "auto",
             tmux: bool = False,
             stop: bool = False,
             port_forward: Optional[cluster_operator.Port_forward] = None,
             with_output: bool = False,
             parallel: bool = True) -> Optional[str]:
        """Runs a command on the specified cluster.

        Args:
            cmd (str): the command to run, or None for a no-op command.
            node_ip (str): node ip on which to run the command
            all_nodes (bool): whether to run the command on all nodes
            run_env (str): whether to run the command on the host or in a
                container. Select between "auto", "host" and "docker".
            tmux (bool): whether to run in a tmux session
            stop (bool): whether to stop the cluster after command run
            port_forward ( (int,int) or list[(int,int)]): port(s) to forward.
            with_output (bool): Whether to capture command output.
            parallel (bool): Whether to run the commands on nodes in parallel

        Returns:
            The output of the command as a string.
        """
        return cluster_operator.exec_on_nodes(
            config=self.config,
            call_context=self.call_context,
            node_ip=node_ip,
            all_nodes=all_nodes,
            cmd=cmd,
            run_env=run_env,
            screen=False,
            tmux=tmux,
            stop=stop,
            start=False,
            port_forward=port_forward,
            with_output=with_output,
            parallel=parallel)

    def submit(self,
               script_file: str,
               script_args,
               tmux: bool = False,
               stop: bool = False,
               port_forward: Optional[cluster_operator.Port_forward] = None) -> Optional[str]:
        """Submit a script file to cluster and run.

        Args:
            script_file (str): The script file to submit and run.
            script_args (array): An array of arguments for the script file.
            tmux (bool): whether to run in a tmux session
            stop (bool): whether to stop the cluster after command run
            port_forward ( (int,int) or list[(int,int)]): port(s) to forward.

        Returns:
            The output of the command as a string.
        """
        return cluster_operator.submit_and_exec(
            config=self.config,
            call_context=self.call_context,
            script=script_file,
            script_args=script_args,
            tmux=tmux,
            stop=stop,
            port_forward=port_forward)

    def rsync(self,
              *,
              source: Optional[str],
              target: Optional[str],
              down: bool,
              node_ip: str = None,
              all_nodes: bool = False,
              use_internal_ip: bool = False):
        """Rsyncs files to or from the cluster.

        Args:
            source (str): rsync source argument.
            target (str): rsync target argument.
            down (bool): whether we're syncing remote -> local.
            node_ip (str): Address of node to rsync
            all_nodes (bool): For rsync-up, whether to rsync uup to all nodes
            use_internal_ip (bool): Whether the provided ip_address is
                public or private.

        Raises:
            RuntimeError if the cluster head node is not found.
        """
        return cluster_operator._rsync(
            config=self.config,
            call_context=self.call_context,
            source=source,
            target=target,
            down=down,
            node_ip=node_ip,
            all_nodes=all_nodes,
            use_internal_ip=use_internal_ip)

    def scale(self, num_cpus: Optional[int] = None,
              bundles: Optional[List[dict]] = None) -> None:
        """Reqeust to scale to accommodate the specified requests.

        The cluster will immediately attempt to scale to accommodate the requested
        resources, bypassing normal upscaling speed constraints. This takes into
        account existing resource usage.

        For example, suppose you call ``request_resources(num_cpus=100)`` and
        there are 45 currently running tasks, each requiring 1 CPU. Then, enough
        nodes will be added so up to 100 tasks can run concurrently. It does
        **not** add enough nodes so that 145 tasks can run.

        This call is only a hint. The actual resulting cluster
        size may be slightly larger or smaller than expected depending on the
        internal bin packing algorithm and max worker count restrictions.

        Args:
            num_cpus (int): Scale the cluster to ensure this number of CPUs are
                available. This request is persistent until another call to
                request_resources() is made to override.
            bundles (List[ResourceDict]): Scale the cluster to ensure this set of
                resource shapes can fit. This request is persistent until another
                call to request_resources() is made to override.

        Examples:
            >>> # Request 1000 CPUs.
            >>> scale(num_cpus=1000)
            >>> # Request 64 CPUs and also fit a 1-GPU/4-CPU task.
            >>> scale(num_cpus=64, bundles=[{"GPU": 1, "CPU": 4}])
            >>> # Same as requesting num_cpus=3.
            >>> scale(bundles=[{"CPU": 1}, {"CPU": 1}, {"CPU": 1}])
        """
        return cluster_operator._scale_cluster(
            config=self.config,
            call_context=self.call_context,
            cpus=num_cpus)

    def start_node(self,
                   node_ip: str = None,
                   all_nodes: bool = False,
                   parallel: bool = True) -> None:
        """Start services on a node.
        Args:
            node_ip (str): The node_ip to run on
            all_nodes(bool): Run on all nodes
            parallel (bool): Run the command in parallel if there are more than one node
        """
        return cluster_operator._start_node_from_head(
            config=self.config,
            call_context=self.call_context,
            node_ip=node_ip,
            all_nodes=all_nodes,
            parallel=parallel
            )

    def stop_node(self,
                  node_ip: str = None,
                  all_nodes: bool = False,
                  parallel: bool = True) -> None:
        """Run stop commands on a node.
        Args:
            node_ip (str): The node_ip to run on
            all_nodes(bool): Run on all nodes
            parallel (bool): Run the command in parallel if there are more than one node
        """
        return cluster_operator._stop_node_from_head(
            config=self.config,
            call_context=self.call_context,
            node_ip=node_ip,
            all_nodes=all_nodes,
            parallel=parallel
            )

    def kill_node(self,
                  node_ip: str = None,
                  hard: bool = False) -> str:
        """Kill a node or a random node
        Args:
            node_ip (str): The node_ip to run on
            hard(bool): Terminate the node by force
        """
        return cluster_operator._kill_node_from_head(
            config=self.config,
            call_context=self.call_context,
            node_ip=node_ip,
            hard=hard)

    def get_head_node_ip(self) -> str:
        """Returns head node IP for given configuration file if exists.

        Returns:
            The ip address of the cluster head node.

        Raises:
            RuntimeError if the cluster is not found.
        """
        return cluster_operator._get_head_node_ip(config=self.config)

    def get_worker_node_ips(self) -> List[str]:
        """Returns worker node IPs for given configuration file.
        Returns:
            List of worker node ip addresses.

        Raises:
            RuntimeError if the cluster is not found.
        """
        return cluster_operator._get_worker_node_ips(config=self.config)

    def get_nodes(self) -> List[Dict[str, Any]]:
        """Returns a list of info for each cluster node
        Returns:
            A list of Dict object for each node with the informaition
        """
        return cluster_operator._get_cluster_nodes_info(config=self.config)

    def get_info(self) -> Dict[str, Any]:
        """Returns the general information of the cluster
        Returns:
            A Dict object for cluster properties
        """
        return cluster_operator._get_cluster_info(config=self.config)

    def wait_for_ready(self, min_workers: int = None,
                       timeout: int = None) -> None:
        """Wait for to the min_workers to be ready.
        Args:
            min_workers (int): If min_workers is not specified, the min_workers of cluster will be used.
            timeout (int): The maximum time to wait
        """
        return cluster_operator._wait_for_ready(config=self.config,
                                                min_workers=min_workers,
                                                timeout=timeout)


def configure_logging(log_style: Optional[str] = None,
                      color_mode: Optional[str] = None,
                      verbosity: Optional[int] = None):
    """Configures logging for cluster command calls.

    Args:
        log_style (str): If 'pretty', outputs with formatting and color.
            If 'record', outputs record-style without formatting.
            'auto' defaults to 'pretty', and disables pretty logging
            if stdin is *not* a TTY. Defaults to "auto".
        color_mode (str):
            Can be "true", "false", or "auto".

            Enables or disables `colorful`.

            If `color_mode` is "auto", is set to `not stdout.isatty()`
        verbosity (int):
            Output verbosity (0, 1, 2, 3).

            Low verbosity will disable `verbose` and `very_verbose` messages.

    """
    cli_logger.configure(
        log_style=log_style, color_mode=color_mode, verbosity=verbosity)


def register_callback_handler(
        cluster_uri:str,
        event_name: str,
        callback: Union[Callable[[Dict], None], List[Callable[[Dict], None]]],
) -> None:
    """Registers a callback handler for scaling  events.

    Args:
        cluster_uri (str): The unique identifier of a cluster which is
                a combination of provider:cluster_name
        event_name (str): Event that callback should be called on. See
            CreateClusterEvent for details on the events available to be
            registered against.
        callback (Callable): Callable object that is invoked
            when specified event occurs.
    """
    global_event_system.add_callback_handler(cluster_uri, event_name, callback)


def get_docker_host_mount_location(cluster_name: str) -> str:
    """Return host path that Docker mounts attach to."""
    docker_mount_prefix = "/tmp/cloudtik_tmp_mount/{cluster_name}"
    return docker_mount_prefix.format(cluster_name=cluster_name)
