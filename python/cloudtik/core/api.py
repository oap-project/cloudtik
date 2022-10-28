"""IMPORTANT: this is an experimental interface and not currently stable."""

from typing import Any, Callable, Dict, List, Optional, Union
import os

from cloudtik.core._private.call_context import CallContext
from cloudtik.core._private.cluster.cluster_config import _bootstrap_config, _load_cluster_config
from cloudtik.core._private.utils import verify_runtime_list, load_head_cluster_config
from cloudtik.core._private.workspace import workspace_operator
from cloudtik.core._private.cluster import cluster_operator
from cloudtik.core._private.event_system import (
    global_event_system)
from cloudtik.core._private.cli_logger import cli_logger
from cloudtik.core._private import utils
from cloudtik.core.workspace_provider import Existence


class Workspace:
    def __init__(self, workspace_config: Union[dict, str]) -> None:
        """Create a workspace object to operator on workspace.

        Args:
            workspace_config (Union[str, dict]): Either the config dict of the
                workspace, or a path pointing to a file containing the config.
        """
        self.workspace_config = workspace_config
        if isinstance(workspace_config, dict):
            self.config = workspace_operator._bootstrap_workspace_config(
                workspace_config, no_config_cache=True)
        else:
            if not os.path.exists(workspace_config):
                raise ValueError("Workspace config file not found: {}".format(workspace_config))
            self.config =workspace_operator._load_workspace_config(
                workspace_config, no_config_cache=True)

    def create(self) -> None:
        """Create and provision the workspace resources."""
        workspace_operator._create_workspace(
            self.config, yes=True)

    def delete(self, delete_managed_storage: bool = False) -> None:
        """Delete the workspace and corresponding resources."""
        workspace_operator._delete_workspace(
            self.config, yes=True, delete_managed_storage=delete_managed_storage)

    def update_firewalls(self) -> None:
        """Update the firewall rules for the workspace."""
        workspace_operator._update_workspace_firewalls(self.config)

    def get_status(self) -> Existence:
        """Return the existence status of the workspace."""
        return workspace_operator._get_workspace_status(self.config)

    def list_clusters(self) -> Optional[Dict[str, Any]]:
        """Get a list of cluster information running in the workspace"""
        return workspace_operator._list_workspace_clusters(self.config)


class Cluster:
    def __init__(self, cluster_config: Union[dict, str], should_bootstrap: bool = True, no_config_cache: bool = True) -> None:
        """Create a cluster object to operate on with this API.

        Args:
            cluster_config (Union[str, dict]): Either the config dict of the
                cluster, or a path pointing to a file containing the config.
        """
        self.cluster_config = cluster_config
        if isinstance(cluster_config, dict):
            if should_bootstrap:
                self.config = _bootstrap_config(
                    cluster_config, no_config_cache=no_config_cache)
            else:
                self.config = cluster_config
        else:
            if not os.path.exists(cluster_config):
                raise ValueError("Cluster config file not found: {}".format(cluster_config))
            self.config = _load_cluster_config(
                cluster_config, should_bootstrap=should_bootstrap, no_config_cache=no_config_cache)

        # TODO: Each call may need its own call context
        self.call_context = CallContext()
        self.call_context.set_call_from_api(True)

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
             keep_min_workers: bool = False,
             hard: bool = False) -> None:
        """Destroys all nodes of a cluster.

        Args:
            workers_only (bool): Whether to keep the head node running and only
                teardown worker nodes.
            keep_min_workers (bool): Whether to keep min_workers (as specified
                in the YAML) still running.
            hard (bool): Stop the cluster nodes by without running stop commands.
        """
        return cluster_operator._teardown_cluster(
            config=self.config,
            call_context=self.call_context,
            workers_only=workers_only,
            keep_min_workers=keep_min_workers,
            hard=hard)

    def exec(self,
             cmd: str,
             *,
             node_ip: str = None,
             all_nodes: bool = False,
             run_env: str = "auto",
             screen: bool = False,
             tmux: bool = False,
             stop: bool = False,
             start: bool = False,
             force_update: bool = False,
             wait_for_workers: bool = False,
             min_workers: Optional[int] = None,
             wait_timeout: Optional[int] = None,
             port_forward: Optional[cluster_operator.Port_forward] = None,
             with_output: bool = False,
             parallel: bool = True,
             job_waiter: Optional[str] = None) -> Optional[str]:
        """Runs a command on the specified cluster.

        Args:
            cmd (str): the command to run
            node_ip (str): node ip on which to run the command
            all_nodes (bool): whether to run the command on all nodes
            run_env (str): whether to run the command on the host or in a
                container. Select between "auto", "host" and "docker".
            screen (bool): whether to run in a screen session
            tmux (bool): whether to run in a tmux session
            stop (bool): whether to stop the cluster after command run
            start (bool): whether to start the cluster if not started
            force_update (bool): if already started, whether force update the configuration if start is true
            wait_for_workers (bool): whether wait for minimum number of ready workers
            min_workers (int): The number of workers to wait for ready
            wait_timeout (int): The timeout for wait for ready
            port_forward ( (int,int) or list[(int,int)]): port(s) to forward.
            with_output (bool): Whether to capture command output.
            parallel (bool): Whether to run the commands on nodes in parallel.
            job_waiter (str): The job waiter to use for waiting an async job to complete.
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
            screen=screen,
            tmux=tmux,
            stop=stop,
            start=start,
            force_update=force_update,
            wait_for_workers=wait_for_workers,
            min_workers=min_workers,
            wait_timeout=wait_timeout,
            port_forward=port_forward,
            with_output=with_output,
            parallel=parallel,
            yes=True,
            job_waiter_name=job_waiter)

    def submit(self,
               script_file: str,
               script_args: Optional[List[str]] = None,
               screen: bool = False,
               tmux: bool = False,
               stop: bool = False,
               start: bool = False,
               force_update: bool = False,
               wait_for_workers: bool = False,
               min_workers: Optional[int] = None,
               wait_timeout: Optional[int] = None,
               port_forward: Optional[cluster_operator.Port_forward] = None,
               with_output: bool = False,
               job_waiter: Optional[str] = None) -> Optional[str]:
        """Submit a script file to cluster and run.

        Args:
            script_file (str): The script file to submit and run.
            script_args (list): An array of arguments for the script file.
            screen (bool): whether to run in a screen session
            tmux (bool): whether to run in a tmux session
            stop (bool): whether to stop the cluster after command run
            start (bool): whether to start the cluster if not started
            force_update (bool): if already started, whether force update the configuration if start is true
            wait_for_workers (bool): whether wait for minimum number of ready workers
            min_workers (int): The number of workers to wait for ready
            wait_timeout (int): The timeout for wait for ready
            port_forward ( (int,int) or list[(int,int)]): port(s) to forward.
            with_output (bool): Whether to capture command output.
            job_waiter (str): The job waiter to use for waiting an async job to complete.
        Returns:
            The output of the command as a string.
        """
        return cluster_operator.submit_and_exec(
            config=self.config,
            call_context=self.call_context,
            script=script_file,
            script_args=script_args,
            screen=screen,
            tmux=tmux,
            stop=stop,
            start=start,
            force_update=force_update,
            wait_for_workers=wait_for_workers,
            min_workers=min_workers,
            wait_timeout=wait_timeout,
            port_forward=port_forward,
            with_output=with_output,
            yes=True,
            job_waiter_name=job_waiter)

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
            all_nodes (bool): For rsync-up, whether to rsync up to all nodes
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

    def scale(self, num_cpus: Optional[int] = None, workers: Optional[int] = None,
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
            workers (int): Scale to number of workers.
            bundles (List[ResourceDict]): Scale the cluster to ensure this set of
                resource shapes can fit. This request is persistent until another
                call to request_resources() is made to override.
        """
        return cluster_operator._scale_cluster(
            config=self.config,
            call_context=self.call_context,
            cpus=num_cpus,
            workers=workers)

    def start_node(self,
                   node_ip: str = None,
                   all_nodes: bool = False,
                   runtimes: Optional[List[str]] = None,
                   parallel: bool = True) -> None:
        """Start services on a node.
        Args:
            node_ip (str): The node_ip to run on
            all_nodes (bool): Run on all nodes
            runtimes (Optional[List[str]]): Optional list of runtime services to start
            parallel (bool): Run the command in parallel if there are more than one node
        """
        verify_runtime_list(self.config, runtimes)
        return cluster_operator._start_node_from_head(
            config=self.config,
            call_context=self.call_context,
            node_ip=node_ip,
            all_nodes=all_nodes,
            runtimes=runtimes,
            parallel=parallel
            )

    def stop_node(self,
                  node_ip: str = None,
                  all_nodes: bool = False,
                  runtimes: Optional[List[str]] = None,
                  parallel: bool = True) -> None:
        """Run stop commands on a node.
        Args:
            node_ip (str): The node_ip to run on
            all_nodes(bool): Run on all nodes
            runtimes (Optional[List[str]]): Optional list of runtime services to start
            parallel (bool): Run the command in parallel if there are more than one node
        """
        verify_runtime_list(self.config, runtimes)
        return cluster_operator._stop_node_from_head(
            config=self.config,
            call_context=self.call_context,
            node_ip=node_ip,
            all_nodes=all_nodes,
            runtimes=runtimes,
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

    def get_head_node_ip(self, public: bool = False) -> str:
        """Returns head node IP for given configuration file if exists.
        Args:
            public (bool): Whether to return the public ip if there is one
        Returns:
            The ip address of the cluster head node.

        Raises:
            RuntimeError if the cluster is not found.
        """
        return cluster_operator._get_head_node_ip(config=self.config, public=public)

    def get_worker_node_ips(self,
                            runtime: str = None) -> List[str]:
        """Returns worker node IPs for given configuration file.
        Returns:
            List of worker node ip addresses.

        Raises:
            RuntimeError if the cluster is not found.
        """
        return cluster_operator._get_worker_node_ips(config=self.config, runtime=runtime)

    def get_nodes(self) -> List[Dict[str, Any]]:
        """Returns a list of info for each cluster node
        Returns:
            A list of Dict object for each node with the information
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
                                                call_context=self.call_context,
                                                min_workers=min_workers,
                                                timeout=timeout)

    def get_default_cloud_storage(self):
        """Get the managed cloud storage information."""
        return cluster_operator.get_default_cloud_storage(config=self.config)


    def register_callback(self,
                          event_name: str,
                          callback: Union[Callable[[Dict], None], List[Callable[[Dict], None]]]) -> None:
        """Registers a callback handler for scaling events.

        Args:
            event_name (str): Event that callback should be called on. See
                CreateClusterEvent for details on the events available to be
                registered against.
            callback (Callable): Callable object that is invoked
                when specified event occurs.
        """
        cluster_uri = utils.get_cluster_uri(self.config)
        global_event_system.add_callback_handler(cluster_uri, event_name, callback)


class ThisCluster:
    def __init__(self) -> None:
        """Create a cluster object to operate on from head with this API."""
        self.config = load_head_cluster_config()

        # TODO: Each call may need its own call context
        self.call_context = CallContext()
        self.call_context.set_call_from_api(True)

    def exec(self,
             cmd: str,
             *,
             node_ip: str = None,
             all_nodes: bool = False,
             run_env: str = "auto",
             screen: bool = False,
             tmux: bool = False,
             wait_for_workers: bool = False,
             min_workers: Optional[int] = None,
             wait_timeout: Optional[int] = None,
             port_forward: Optional[cluster_operator.Port_forward] = None,
             with_output: bool = False,
             parallel: bool = True,
             job_waiter: Optional[str] = None) -> Optional[str]:
        """Runs a command on the specified cluster.

        Args:
            cmd (str): the command to run
            node_ip (str): node ip on which to run the command
            all_nodes (bool): whether to run the command on all nodes
            run_env (str): whether to run the command on the host or in a
                container. Select between "auto", "host" and "docker".
            screen (bool): whether to run in a screen session
            tmux (bool): whether to run in a tmux session
            wait_for_workers (bool): whether wait for minimum number of ready workers
            min_workers (int): The number of workers to wait for ready
            wait_timeout (int): The timeout for wait for ready
            port_forward ( (int,int) or list[(int,int)]): port(s) to forward.
            with_output (bool): Whether to capture command output.
            parallel (bool): Whether to run the commands on nodes in parallel.
            job_waiter (str): The job waiter to use for waiting an async job to complete.
        Returns:
            The output of the command as a string.
        """
        return cluster_operator._exec_node_on_head(
            config=self.config,
            call_context=self.call_context,
            node_ip=node_ip,
            all_nodes=all_nodes,
            cmd=cmd,
            run_env=run_env,
            screen=screen,
            tmux=tmux,
            wait_for_workers=wait_for_workers,
            min_workers=min_workers,
            wait_timeout=wait_timeout,
            port_forward=port_forward,
            with_output=with_output,
            parallel=parallel,
            job_waiter_name=job_waiter)

    def rsync(self,
              *,
              source: Optional[str],
              target: Optional[str],
              down: bool,
              node_ip: str = None,
              all_workers: bool = False):
        """Rsyncs files to or from the cluster.

        Args:
            source (str): rsync source argument.
            target (str): rsync target argument.
            down (bool): whether we're syncing remote -> local.
            node_ip (str): Address of node to rsync
            all_workers (bool): For rsync-up, whether to rsync up to all workers
        Raises:
            RuntimeError if the cluster head node is not found.
        """
        return cluster_operator.rsync_node_on_head(
            config=self.config,
            call_context=self.call_context,
            source=source,
            target=target,
            down=down,
            node_ip=node_ip,
            all_workers=all_workers)

    def scale(self, num_cpus: Optional[int] = None, workers: Optional[int] = None,
              bundles: Optional[List[dict]] = None) -> None:
        """Reqeust to scale to accommodate the specified requests."""
        return cluster_operator._scale_cluster_on_head(
            config=self.config,
            call_context=self.call_context,
            cpus=num_cpus,
            workers=workers)

    def start_node(self,
                   node_ip: str = None,
                   all_nodes: bool = False,
                   runtimes: Optional[List[str]] = None,
                   parallel: bool = True) -> None:
        """Start services on a node.
        Args:
            node_ip (str): The node_ip to run on
            all_nodes (bool): Run on all nodes
            runtimes (Optional[List[str]]): Optional list of runtime services to start
            parallel (bool): Run the command in parallel if there are more than one node
        """
        verify_runtime_list(self.config, runtimes)
        return cluster_operator._start_node_on_head(
            config=self.config,
            call_context=self.call_context,
            node_ip=node_ip,
            all_nodes=all_nodes,
            runtimes=runtimes,
            parallel=parallel
            )

    def stop_node(self,
                  node_ip: str = None,
                  all_nodes: bool = False,
                  runtimes: Optional[List[str]] = None,
                  parallel: bool = True) -> None:
        """Run stop commands on a node.
        Args:
            node_ip (str): The node_ip to run on
            all_nodes(bool): Stop on all nodes
            runtimes (Optional[List[str]]): Optional list of runtime services to start
            parallel (bool): Run the command in parallel if there are more than one node
        """
        verify_runtime_list(self.config, runtimes)
        return cluster_operator._stop_node_on_head(
            config=self.config,
            call_context=self.call_context,
            node_ip=node_ip,
            all_nodes=all_nodes,
            runtimes=runtimes,
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
        return cluster_operator._kill_node(
            config=self.config,
            call_context=self.call_context,
            node_ip=node_ip,
            hard=hard)

    def get_head_node_ip(self, public: bool = False) -> str:
        """Returns head node IP for given configuration file if exists.
        Args:
            public (bool): Whether to return the public ip if there is one
        Returns:
            The ip address of the cluster head node.

        Raises:
            RuntimeError if the cluster is not found.
        """
        return cluster_operator._get_head_node_ip(config=self.config, public=public)

    def get_worker_node_ips(self,
                            runtime: str = None) -> List[str]:
        """Returns worker node IPs for given configuration file.
        Returns:
            List of worker node ip addresses.

        Raises:
            RuntimeError if the cluster is not found.
        """
        return cluster_operator._get_worker_node_ips(config=self.config, runtime=runtime)

    def get_nodes(self) -> List[Dict[str, Any]]:
        """Returns a list of info for each cluster node
        Returns:
            A list of Dict object for each node with the information
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
                                                call_context=self.call_context,
                                                min_workers=min_workers,
                                                timeout=timeout)

    def get_default_cloud_storage(self):
        """Get the default cloud storage information."""
        return cluster_operator.get_default_cloud_storage(config=self.config)


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


def get_docker_host_mount_location(cluster_name: str) -> str:
    """Return host path that Docker mounts attach to."""
    docker_mount_prefix = "/tmp/cloudtik_tmp_mount/{cluster_name}"
    return docker_mount_prefix.format(cluster_name=cluster_name)
