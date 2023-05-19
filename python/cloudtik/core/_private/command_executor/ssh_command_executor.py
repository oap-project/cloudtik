import copy
from getpass import getuser
from typing import Dict
import hashlib
import logging
import os
import subprocess
import time

from cloudtik.core._private.command_executor.command_executor \
    import _with_shutdown, _with_environment_variables, _with_interactive
from cloudtik.core._private.command_executor.host_command_executor import HostCommandExecutor
from cloudtik.core._private.constants import \
    CLOUDTIK_NODE_START_WAIT_S
from cloudtik.core._private.log_timer import LogTimer

from cloudtik.core._private.cli_logger import cf

logger = logging.getLogger(__name__)

HASH_MAX_LENGTH = 10


class SSHOptions:
    def __init__(self, call_context, ssh_key, control_path=None, **kwargs):
        self.call_context = call_context
        self.ssh_key = ssh_key
        self.arg_dict = {
            # Supresses initial fingerprint verification.
            "StrictHostKeyChecking": "no",
            # SSH IP and fingerprint pairs no longer added to known_hosts.
            # This is to remove a "REMOTE HOST IDENTIFICATION HAS CHANGED"
            # warning if a new node has the same IP as a previously
            # deleted node, because the fingerprints will not match in
            # that case.
            "UserKnownHostsFile": os.devnull,
            # Try fewer extraneous key pairs.
            "IdentitiesOnly": "yes",
            # Abort if port forwarding fails (instead of just printing to
            # stderr).
            "ExitOnForwardFailure": "yes",
            # Quickly kill the connection if network connection breaks (as
            # opposed to hanging/blocking).
            "ServerAliveInterval": 5,
            "ServerAliveCountMax": 10
        }
        if self.call_context.cli_logger.verbosity == 0:
            self.arg_dict["LogLevel"] = "ERROR"
        if control_path:
            self.arg_dict.update({
                "ControlMaster": "auto",
                "ControlPath": "{}/%C".format(control_path),
                "ControlPersist": "10s",
            })
        self.arg_dict.update(kwargs)

    def to_ssh_options_list(self, *, timeout=60):
        self.arg_dict["ConnectTimeout"] = "{}s".format(timeout)
        ssh_key_option = ["-i", self.ssh_key] if self.ssh_key else []
        return ssh_key_option + [
            x for y in (["-o", "{}={}".format(k, v)]
                        for k, v in self.arg_dict.items()
                        if v is not None) for x in y
        ]


class SSHCommandExecutor(HostCommandExecutor):
    def __init__(self, call_context, log_prefix, node_id, provider, auth_config,
                 cluster_name, process_runner, use_internal_ip):
        HostCommandExecutor.__init__(
            self, call_context, log_prefix, node_id, provider, auth_config,
            cluster_name, process_runner, use_internal_ip)

        ssh_control_hash = hashlib.md5(cluster_name.encode()).hexdigest()
        ssh_user_hash = hashlib.md5(getuser().encode()).hexdigest()
        ssh_control_path = "/tmp/cloudtik_ssh_{}/{}".format(
            ssh_user_hash[:HASH_MAX_LENGTH],
            ssh_control_hash[:HASH_MAX_LENGTH])
        self.ssh_private_key = auth_config.get("ssh_private_key")
        self.ssh_control_path = ssh_control_path
        self.ssh_ip = None
        self.ssh_proxy_command = auth_config.get("ssh_proxy_command", None)
        self.ssh_options = SSHOptions(
            self.call_context,
            self.ssh_private_key,
            self.ssh_control_path,
            ProxyCommand=self.ssh_proxy_command)

    def _set_ssh_ip_if_required(self):
        if self.ssh_ip is not None:
            return

        # We assume that this never changes.
        #   I think that's reasonable.
        deadline = time.time() + CLOUDTIK_NODE_START_WAIT_S
        with LogTimer(self.log_prefix + "Got IP"):
            ip = self._wait_for_ip(deadline)

            self.cli_logger.doassert(ip is not None,
                                     "Could not get node IP.")  # todo: msg
            assert ip is not None, "Unable to find IP of node"

        self.ssh_ip = ip

        # This should run before any SSH commands and therefore ensure that
        #   the ControlPath directory exists, allowing SSH to maintain
        #   persistent sessions later on.
        try:
            os.makedirs(self.ssh_control_path, mode=0o700, exist_ok=True)
        except OSError as e:
            self.cli_logger.warning("{}", str(e))  # todo: msg

    def run(
            self,
            cmd=None,
            timeout=120,
            exit_on_fail=False,
            port_forward=None,
            with_output=False,
            environment_variables: Dict[str, object] = None,
            run_env="auto",  # Unused argument.
            ssh_options_override_ssh_key="",
            shutdown_after_run=False,
            cmd_to_print=None,
            silent=False):
        if shutdown_after_run:
            cmd, cmd_to_print = _with_shutdown(cmd, cmd_to_print)
        if ssh_options_override_ssh_key:
            ssh_options = SSHOptions(
                self.call_context, ssh_options_override_ssh_key,
                ProxyCommand=self.ssh_proxy_command)
        else:
            ssh_options = self.ssh_options

        assert isinstance(
            ssh_options, SSHOptions
        ), "ssh_options must be of type SSHOptions, got {}".format(
            type(ssh_options))

        self._set_ssh_ip_if_required()

        if self.call_context.is_using_login_shells():
            ssh = ["ssh", "-tt"]
        else:
            ssh = ["ssh"]

        if port_forward:
            with self.cli_logger.group("Forwarding ports"):
                if not isinstance(port_forward, list):
                    port_forward = [port_forward]
                for local, remote in port_forward:
                    self.cli_logger.verbose(
                        "Forwarding port {} to port {} on localhost.",
                        cf.bold(local), cf.bold(remote))  # todo: msg
                    ssh += ["-L", "{}:localhost:{}".format(remote, local)]

        final_cmd = ssh + ssh_options.to_ssh_options_list(timeout=timeout) + [
            "{}@{}".format(self.ssh_user, self.ssh_ip)
        ]
        final_cmd_to_print = None
        if cmd:
            if environment_variables:
                cmd, cmd_to_print = _with_environment_variables(
                    cmd, environment_variables, cmd_to_print=cmd_to_print)
            if cmd_to_print:
                final_cmd_to_print = copy.deepcopy(final_cmd)
            if self.call_context.is_using_login_shells():
                final_cmd += _with_interactive(cmd)
                if cmd_to_print:
                    final_cmd_to_print += _with_interactive(cmd_to_print)
            else:
                final_cmd += [cmd]
                if cmd_to_print:
                    final_cmd_to_print += [cmd_to_print]
        else:
            # We do this because `-o ControlMaster` causes the `-N` flag to
            # still create an interactive shell in some ssh versions.
            final_cmd.append("while true; do sleep 86400; done")

        self.cli_logger.verbose(
            "Running `{}`", cf.bold(cmd if cmd_to_print is None else cmd_to_print))
        with self.cli_logger.indented():
            self.cli_logger.verbose(
                "Full command is `{}`",
                cf.bold(" ".join(final_cmd if final_cmd_to_print is None else final_cmd_to_print)))

        if self.cli_logger.verbosity > 0:
            with self.cli_logger.indented():
                return self._run_helper(
                    final_cmd, with_output, exit_on_fail,
                    silent=silent, cmd_to_print=final_cmd_to_print)
        else:
            return self._run_helper(
                final_cmd, with_output, exit_on_fail,
                silent=silent, cmd_to_print=final_cmd_to_print)

    def run_rsync_up(self, source, target, options=None):
        self._set_ssh_ip_if_required()
        options = options or {}

        command = ["rsync"]
        command += [
            "--rsh",
            subprocess.list2cmdline(
                ["ssh"] + self.ssh_options.to_ssh_options_list(timeout=120))
        ]
        command += ["-avz"]
        command += self._create_rsync_filter_args(options=options)
        command += [
            source, "{}@{}:{}".format(self.ssh_user, self.ssh_ip, target)
        ]
        self.cli_logger.verbose("Running `{}`", cf.bold(" ".join(command)))
        self._run_helper(command, silent=self.call_context.is_rsync_silent())

    def run_rsync_down(self, source, target, options=None):
        self._set_ssh_ip_if_required()

        command = ["rsync"]
        command += [
            "--rsh",
            subprocess.list2cmdline(
                ["ssh"] + self.ssh_options.to_ssh_options_list(timeout=120))
        ]
        command += ["-avz"]
        command += self._create_rsync_filter_args(options=options)
        command += [
            "{}@{}:{}".format(self.ssh_user, self.ssh_ip, source), target
        ]
        self.cli_logger.verbose("Running `{}`", cf.bold(" ".join(command)))
        self._run_helper(command, silent=self.call_context.is_rsync_silent())

    def remote_shell_command_str(self):
        self._set_ssh_ip_if_required()
        command = "ssh -o IdentitiesOnly=yes"
        if self.ssh_private_key:
            command += " -i {}".format(self.ssh_private_key)
        if self.ssh_proxy_command:
            command += " -o ProxyCommand='{}'".format(self.ssh_proxy_command)
        return command + " {}@{}\n".format(
            self.ssh_user, self.ssh_ip)
