import copy
from shlex import quote
from typing import Dict
import logging
import os
import subprocess
import sys
import time
import warnings

from cloudtik.core._private.command_executor.command_executor import _with_shutdown, _with_environment_variables, \
    _with_login_shell
from cloudtik.core.command_executor import CommandExecutor

from cloudtik.core._private.cli_logger import cf
from cloudtik.core._private.debug import log_once

logger = logging.getLogger(__name__)

KUBECTL_RSYNC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "providers/_private/_kubernetes/kubectl-rsync.sh")
MAX_HOME_RETRIES = 3
HOME_RETRY_DELAY_S = 5


class KubernetesCommandExecutor(CommandExecutor):
    def __init__(self, call_context, log_prefix, namespace, node_id, auth_config,
                 process_runner):
        CommandExecutor.__init__(self, call_context)
        self.log_prefix = log_prefix
        self.process_runner = process_runner
        self.node_id = str(node_id)
        self.namespace = namespace
        self.kubectl = ["kubectl", "-n", self.namespace]
        self._home_cached = None

    def run(
            self,
            cmd=None,
            timeout=120,
            exit_on_fail=False,
            port_forward=None,
            with_output=False,
            environment_variables: Dict[str, object] = None,
            run_env="auto",  # Unused argument.
            ssh_options_override_ssh_key="",  # Unused argument.
            shutdown_after_run=False,
            cmd_to_print=None,
            silent=False,  # Unused argument.
    ):
        if shutdown_after_run:
            cmd, cmd_to_print = _with_shutdown(cmd, cmd_to_print)
        if cmd and port_forward:
            raise Exception(
                "exec with Kubernetes can't forward ports and execute"
                "commands together.")

        if port_forward:
            if not isinstance(port_forward, list):
                port_forward = [port_forward]
            port_forward_cmd = self.kubectl + [
                "port-forward",
                self.node_id,
            ] + [
                "{}:{}".format(local, remote) for local, remote in port_forward
            ]
            logger.info("Port forwarding with: {}".format(
                " ".join(port_forward_cmd)))
            port_forward_process = subprocess.Popen(port_forward_cmd)
            port_forward_process.wait()
            # We should never get here, this indicates that port forwarding
            # failed, likely because we couldn't bind to a port.
            pout, perr = port_forward_process.communicate()
            exception_str = " ".join(
                port_forward_cmd) + " failed with error: " + perr.decode()
            raise Exception(exception_str)
        else:
            if environment_variables:
                cmd, cmd_to_print = _with_environment_variables(
                    cmd, environment_variables, cmd_to_print=cmd_to_print)
            if self.call_context.is_using_login_shells():
                cmd = _with_login_shell(
                    cmd, interactive=self.call_context.does_allow_interactive())
                cmd_to_print = _with_login_shell(
                    cmd_to_print, interactive=self.call_context.does_allow_interactive()) if cmd_to_print else None
            else:
                # Originally, cmd and cmd_to_print is a string
                # Hence it was converted to a array by _with_interactive
                cmd = [cmd]
                if cmd_to_print:
                    cmd_to_print = [cmd_to_print]

            final_cmd = self._with_kubectl_exec()
            cmd_prefix = " ".join(final_cmd)

            final_cmd_to_print = None
            if cmd_to_print:
                final_cmd_to_print = copy.deepcopy(final_cmd)

            final_cmd += cmd
            if cmd_to_print:
                final_cmd_to_print += cmd_to_print
            # `kubectl exec` + subprocess w/ list of args has unexpected
            # side-effects.
            final_cmd = " ".join(final_cmd)
            final_cmd_to_print = " ".join(final_cmd_to_print) if final_cmd_to_print else None

            self.cli_logger.verbose("Running `{}`", cf.bold(" ".join(cmd if cmd_to_print is None else cmd_to_print)))
            with self.cli_logger.indented():
                self.cli_logger.verbose(
                    "Full command is `{}`",
                    cf.bold(final_cmd if final_cmd_to_print is None else final_cmd_to_print))
            try:
                if with_output:
                    return self.process_runner.check_output(
                        final_cmd, shell=True)
                else:
                    self.process_runner.check_call(final_cmd, shell=True)
            except subprocess.CalledProcessError:
                if (not self.call_context.is_using_login_shells()) or (
                        self.call_context.is_call_from_api()):
                    raise

                if exit_on_fail:
                    quoted_cmd = cmd_prefix + quote(" ".join(cmd if cmd_to_print is None else cmd_to_print))
                    msg = self.log_prefix + "Command failed"
                    if self.cli_logger.verbosity > 0:
                        msg += ": \n\n  {}\n".format(quoted_cmd)
                    else:
                        msg += ". Use -v for more details."
                    logger.error(
                        msg)
                    sys.exit(1)
                else:
                    raise

    def run_rsync_up(self, source, target, options=None):
        options = options or {}
        if options.get("rsync_exclude"):
            if log_once("scaler_k8s_rsync_exclude"):
                logger.warning("'rsync_exclude' detected but is currently "
                               "unsupported for k8s.")
        if options.get("rsync_filter"):
            if log_once("scaler_k8s_rsync_filter"):
                logger.warning("'rsync_filter' detected but is currently "
                               "unsupported for k8s.")
        if target.startswith("~"):
            target = self._home + target[1:]

        try:
            flags = "-aqz" if self.call_context.is_rsync_silent() else "-avz"
            self.process_runner.check_call([
                KUBECTL_RSYNC,
                flags,
                source,
                "{}@{}:{}".format(self.node_id, self.namespace, target),
            ])
        except Exception as e:
            warnings.warn(
                self.log_prefix +
                "rsync failed: '{}'. Falling back to 'kubectl cp'".format(e),
                UserWarning)

            self.process_runner.check_call(self.kubectl + [
                "cp", source, "{}/{}:{}".format(self.namespace, self.node_id,
                                                target)
            ])

    def run_rsync_down(self, source, target, options=None):
        if source.startswith("~"):
            source = self._home + source[1:]

        try:
            flags = "-aqz" if self.call_context.is_rsync_silent() else "-avz"
            self.process_runner.check_call([
                KUBECTL_RSYNC,
                flags,
                "{}@{}:{}".format(self.node_id, self.namespace, source),
                target,
            ])
        except Exception as e:
            warnings.warn(
                self.log_prefix +
                "rsync failed: '{}'. Falling back to 'kubectl cp'".format(e),
                UserWarning)

            self.process_runner.check_call(self.kubectl + [
                "cp", "{}/{}:{}".format(self.namespace, self.node_id, source),
                target
            ])

    def remote_shell_command_str(self):
        return "{} exec -it {} -- bash".format(" ".join(self.kubectl),
                                               self.node_id)

    @property
    def _home(self):
        if self._home_cached is not None:
            return self._home_cached
        for _ in range(MAX_HOME_RETRIES - 1):
            try:
                self._home_cached = self._try_to_get_home()
                return self._home_cached
            except Exception:
                # TODO (Dmitri): Identify the exception we're trying to avoid.
                logger.info("Error reading container's home directory. "
                            f"Retrying in {HOME_RETRY_DELAY_S} seconds.")
                time.sleep(HOME_RETRY_DELAY_S)
        # Last try
        self._home_cached = self._try_to_get_home()
        return self._home_cached

    def _try_to_get_home(self):
        # TODO: Think about how to use the node's HOME variable
        # without making an extra kubectl exec call.
        cmd = self._with_kubectl_exec(["printenv", "HOME"])
        joined_cmd = " ".join(cmd)
        raw_out = self.process_runner.check_output(joined_cmd, shell=True)
        home = raw_out.decode().strip("\n\r")
        return home

    def _with_kubectl_exec(self, cmd=None):
        exec_cmd = self.kubectl + ["exec"]
        if self.call_context.does_allow_interactive():
            exec_cmd += ["-it"]
        exec_cmd += [
            self.node_id,
            "--",
        ]
        if cmd:
            exec_cmd += cmd
        return exec_cmd
