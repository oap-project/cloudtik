import copy
import click
import subprocess
import sys
from typing import Dict
import logging

from cloudtik.core._private.command_executor.command_executor \
    import _with_shutdown, _with_environment_variables, _with_interactive
from cloudtik.core._private.command_executor.host_command_executor import HostCommandExecutor
from cloudtik.core._private.cli_logger import cf
from cloudtik.core._private.subprocess_output_util import ProcessRunnerError

logger = logging.getLogger(__name__)


def _with_local_interactive(cmd):
    return (
        f"true && "
        f"export PYTHONWARNINGS=ignore && ({cmd})")


class LocalCommandExecutor(HostCommandExecutor):
    def __init__(self, call_context, log_prefix, auth_config,
                 cluster_name, process_runner, use_internal_ip,
                 provider, node_id):
        HostCommandExecutor.__init__(
            self, call_context, log_prefix, auth_config,
            cluster_name, process_runner, use_internal_ip,
            provider, node_id)

    def _run_local_shell(self, cmd: str,
                         with_output=False,
                         exit_on_fail=False,
                         cmd_to_print=None):
        try:
            if with_output:
                return self.process_runner.check_output(
                    cmd, shell=True)
            else:
                self.process_runner.check_call(
                    cmd, shell=True)
        except subprocess.CalledProcessError as e:
            cmd_to_print = cmd if cmd_to_print is None else cmd_to_print
            if (not self.call_context.is_using_login_shells()) or (
                    self.call_context.is_call_from_api()):
                raise ProcessRunnerError(
                    "Command failed",
                    "command_failed",
                    code=e.returncode,
                    command=cmd_to_print,
                    output=e.output)

            if exit_on_fail:
                msg = "Command failed"
                if self.cli_logger.verbosity > 0:
                    msg += ":\n\n  {}\n".format(cmd_to_print)
                else:
                    msg += ". Use -v for more details.".format(cmd_to_print)
                raise click.ClickException(
                    msg) from None
            else:
                fail_msg = "Command failed."
                if self.call_context.is_output_redirected():
                    fail_msg += " See above for the output from the failure."
                raise click.ClickException(fail_msg) from None
        finally:
            # Do our best to flush output to terminal.
            sys.stdout.flush()
            sys.stderr.flush()

    def run(
            self,
            cmd=None,
            timeout=120,
            exit_on_fail=False,
            port_forward=None,  # Unused argument.
            with_output=False,
            environment_variables: Dict[str, object] = None,
            run_env="auto",  # Unused argument.
            ssh_options_override_ssh_key="",  # Unused argument.
            shutdown_after_run=False,
            cmd_to_print=None,
            silent=False,  # Unused argument
    ):
        if shutdown_after_run:
            cmd, cmd_to_print = _with_shutdown(cmd, cmd_to_print)

        final_cmd = ""
        final_cmd_to_print = None
        if cmd:
            if environment_variables:
                cmd, cmd_to_print = _with_environment_variables(
                    cmd, environment_variables, cmd_to_print=cmd_to_print)
            if cmd_to_print:
                final_cmd_to_print = copy.deepcopy(final_cmd)
            if self.call_context.is_using_login_shells():
                final_cmd = _with_local_interactive(cmd)
                if cmd_to_print:
                    final_cmd_to_print = _with_local_interactive(cmd_to_print)
            else:
                final_cmd = cmd
                if cmd_to_print:
                    final_cmd_to_print = cmd_to_print
        else:
            # We do this because `-o ControlMaster` causes the `-N` flag to
            # still create an interactive shell in some ssh versions.
            final_cmd = "true"

        self.cli_logger.verbose(
            "Running `{}`", cf.bold(cmd if cmd_to_print is None else cmd_to_print))
        with self.cli_logger.indented():
            self.cli_logger.verbose(
                "Full command is `{}`",
                cf.bold(final_cmd if final_cmd_to_print is None else final_cmd_to_print))

        if self.cli_logger.verbosity > 0:
            with self.cli_logger.indented():
                return self._run_local_shell(
                    final_cmd, with_output=with_output,
                    exit_on_fail=exit_on_fail,
                    cmd_to_print=final_cmd_to_print)
        else:
            return self._run_local_shell(
                final_cmd, with_output=with_output,
                exit_on_fail=exit_on_fail,
                cmd_to_print=final_cmd_to_print)

    def run_rsync_up(self, source, target, options=None):
        self._run_rsync(source, target, options)

    def run_rsync_down(self, source, target, options=None):
        self._run_rsync(source, target, options)

    def _run_rsync(self, source, target, options=None):
        # this is on the same host, check whether source target is the same
        if source == target:
            return

        command = ["rsync"]
        command += ["-avz"]
        command += self._create_rsync_filter_args(options=options)
        command += [source, target]
        self.cli_logger.verbose("Running `{}`", cf.bold(" ".join(command)))
        self._run_helper(command, silent=self.call_context.is_rsync_silent())

    def remote_shell_command_str(self):
        return "bash"
