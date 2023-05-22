import time
from typing import Any, List, Tuple, Dict, Optional

from cloudtik.core._private.call_context import CallContext

COMMAND_RUN_DEFAULT_NUMBER_OF_RETRIES = 30
COMMAND_RUN_DEFAULT_RETRY_DELAY_S = 5

MAX_COMMAND_LENGTH_TO_PRINT = 48


def get_cmd_to_print(cmd, verbose=False):
    if not verbose and len(cmd) > MAX_COMMAND_LENGTH_TO_PRINT:
        cmd_to_print = cmd[:MAX_COMMAND_LENGTH_TO_PRINT] + "..."
    else:
        cmd_to_print = cmd
    return cmd_to_print


class CommandExecutor:
    """Interface to run commands on a remote cluster node.

    **Important**: This is an INTERNAL API that is only exposed for the purpose
    of implementing custom node providers. It is not allowed to call into
    CommandRunner methods from any package outside, only to
    define new implementations for use with the "external" node provider
    option.

    Command executor instances are returned by provider.get_command_executor()."""
    def __init__(self, call_context: CallContext) -> None:
        self.call_context = call_context

    @property
    def cli_logger(self):
        return self.call_context.cli_logger

    def run(
            self,
            cmd: str = None,
            timeout: int = 120,
            exit_on_fail: bool = False,
            port_forward: List[Tuple[int, int]] = None,
            with_output: bool = False,
            environment_variables: Dict[str, object] = None,
            run_env: str = "auto",
            ssh_options_override_ssh_key: str = "",
            shutdown_after_run: bool = False,
            cmd_to_print: str = None,
            silent: bool = False
    ) -> str:
        """Run the given command on the cluster node and optionally get output.

        WARNING: the cloudgateway needs arguments of "run" function to be json
            dumpable to send them over HTTP requests.

        Args:
            cmd (str): The command to run.
            timeout (int): The command timeout in seconds.
            exit_on_fail (bool): Whether to sys exit on failure.
            port_forward (list): List of (local, remote) ports to forward, or
                a single tuple.
            with_output (bool): Whether to return output.
            environment_variables (Dict[str, str | int | Dict[str, str]):
                Environment variables that `cmd` should be run with.
            run_env (str): Options: docker/host/auto. Used in
                DockerCommandRunner to determine the run environment.
            ssh_options_override_ssh_key (str): if provided, overwrites
                SSHOptions class with SSHOptions(ssh_options_override_ssh_key).
            shutdown_after_run (bool): if provided, shutdowns down the machine
            after executing the command with `sudo shutdown -h now`.
            cmd_to_print (str): The command to print instead of print the original cmd.
            silent (bool): Whether run this command in silent mode without output
        """
        raise NotImplementedError

    def run_with_retry(
            self,
            cmd: str = None,
            timeout: int = 120,
            exit_on_fail: bool = False,
            port_forward: List[Tuple[int, int]] = None,
            with_output: bool = False,
            environment_variables: Dict[str, object] = None,
            run_env: str = "auto",
            ssh_options_override_ssh_key: str = "",
            shutdown_after_run: bool = False,
            cmd_to_print: str = None,
            silent: bool = False,
            number_of_retries: Optional[int] = None,
            retry_interval: Optional[int] = None
    ) -> str:
        retries = number_of_retries if number_of_retries is not None else COMMAND_RUN_DEFAULT_NUMBER_OF_RETRIES
        interval = retry_interval if retry_interval is not None else COMMAND_RUN_DEFAULT_RETRY_DELAY_S
        while retries > 0:
            try:
                return self.run(cmd,
                                timeout=timeout,
                                exit_on_fail=exit_on_fail,
                                port_forward=port_forward,
                                with_output=with_output,
                                environment_variables=environment_variables,
                                run_env=run_env,
                                ssh_options_override_ssh_key=ssh_options_override_ssh_key,
                                shutdown_after_run=shutdown_after_run,
                                cmd_to_print=cmd_to_print,
                                silent=silent)
            except Exception as e:
                retries -= 1
                if retries > 0:
                    cmd_to_print = cmd if cmd_to_print is None else cmd_to_print
                    verbose = False if self.cli_logger.verbosity == 0 else True
                    cmd_to_print = get_cmd_to_print(cmd_to_print, verbose)
                    self.cli_logger.warning(
                        "Error running command: {}. Retrying in {} seconds.",
                        cmd_to_print,
                        interval
                    )
                    time.sleep(interval)
                else:
                    raise e

    def run_rsync_up(self,
                     source: str,
                     target: str,
                     options: Optional[Dict[str, Any]] = None) -> None:
        """Rsync files up to the cluster node.

        Args:
            source (str): The (local) source directory or file.
            target (str): The (remote) destination path.
        """
        raise NotImplementedError

    def run_rsync_down(self,
                       source: str,
                       target: str,
                       options: Optional[Dict[str, Any]] = None) -> None:
        """Rsync files down from the cluster node.

        Args:
            source (str): The (remote) source directory or file.
            target (str): The (local) destination path.
        """
        raise NotImplementedError

    def remote_shell_command_str(self) -> str:
        """Return the command the user can use to open a shell."""
        raise NotImplementedError

    def run_init(self, *, as_head: bool, file_mounts: Dict[str, str],
                 shared_memory_ratio: float, sync_run_yet: bool) -> Optional[bool]:
        """Used to run extra initialization commands.

        Args:
            as_head (bool): Run as head image or worker.
            file_mounts (dict): Files to copy to the head and worker nodes.
            shared_memory_ratio(float): The share memory ratio for the node
            sync_run_yet (bool): Whether sync has been run yet.

        Returns:
            optional (bool): Whether initialization is necessary.
        """
        pass

    def run_terminate(self):
        """Used to run extra terminate commands."""
        pass

    def bootstrap_data_disks(self) -> None:
        """Used to format and mount data disks on host."""
        pass
