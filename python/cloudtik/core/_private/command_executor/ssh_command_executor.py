import copy
from getpass import getuser
from typing import Dict, List
import click
import hashlib
import json
import logging
import os
import subprocess
import time

from cloudtik.core._private.command_executor.command_executor import _with_shutdown, _with_environment_variables, \
    _with_interactive
from cloudtik.core.command_executor import CommandExecutor
from cloudtik.core._private.constants import \
    CLOUDTIK_NODE_SSH_INTERVAL_S, \
    CLOUDTIK_NODE_START_WAIT_S, \
    CLOUDTIK_DATA_DISK_MOUNT_POINT
from cloudtik.core._private.log_timer import LogTimer

from cloudtik.core._private.subprocess_output_util import (
    run_cmd_redirected, ProcessRunnerError)

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


class SSHCommandExecutor(CommandExecutor):
    def __init__(self, call_context, log_prefix, node_id, provider, auth_config,
                 cluster_name, process_runner, use_internal_ip):
        CommandExecutor.__init__(self, call_context)
        ssh_control_hash = hashlib.md5(cluster_name.encode()).hexdigest()
        ssh_user_hash = hashlib.md5(getuser().encode()).hexdigest()
        ssh_control_path = "/tmp/cloudtik_ssh_{}/{}".format(
            ssh_user_hash[:HASH_MAX_LENGTH],
            ssh_control_hash[:HASH_MAX_LENGTH])

        self.cluster_name = cluster_name
        self.log_prefix = log_prefix
        self.process_runner = process_runner
        self.node_id = node_id
        self.use_internal_ip = use_internal_ip
        self.provider = provider
        self.ssh_private_key = auth_config.get("ssh_private_key")
        self.ssh_user = auth_config["ssh_user"]
        self.ssh_control_path = ssh_control_path
        self.ssh_ip = None
        self.ssh_proxy_command = auth_config.get("ssh_proxy_command", None)
        self.ssh_options = SSHOptions(
            self.call_context,
            self.ssh_private_key,
            self.ssh_control_path,
            ProxyCommand=self.ssh_proxy_command)

    def _get_node_ip(self):
        if self.use_internal_ip:
            return self.provider.internal_ip(self.node_id)
        else:
            return self.provider.external_ip(self.node_id)

    def _wait_for_ip(self, deadline):
        # if we have IP do not print waiting info
        ip = self._get_node_ip()
        if ip is not None:
            if self.cli_logger.verbosity > 0:
                self.cli_logger.labeled_value("Fetched IP", ip)
            return ip

        interval = CLOUDTIK_NODE_SSH_INTERVAL_S
        with self.cli_logger.group("Waiting for IP"):
            while time.time() < deadline and \
                    not self.provider.is_terminated(self.node_id):
                ip = self._get_node_ip()
                if ip is not None:
                    self.cli_logger.labeled_value("Received", ip)
                    return ip
                self.cli_logger.print("Not yet available, retrying in {} seconds",
                                 cf.bold(str(interval)))
                time.sleep(interval)

        return None

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

    def _run_helper(self,
                    final_cmd,
                    with_output=False,
                    exit_on_fail=False,
                    silent=False,
                    cmd_to_print=None):
        """Run a command that was already setup with SSH and `bash` settings.

        Args:
            final_cmd (List[str]):
                Full command to run. Should include SSH options and other
                processing that we do.
            with_output (bool):
                If `with_output` is `True`, command stdout and stderr
                will be captured and returned.
            exit_on_fail (bool):
                If `exit_on_fail` is `True`, the process will exit
                if the command fails (exits with a code other than 0).
            cmd_to_print (Optional[List[str]]):
                Command to print out for any error cases.
        Raises:
            ProcessRunnerError if using new log style and disabled
                login shells.
            click.ClickException if using login shells.
        """
        try:
            # For now, if the output is needed we just skip the new logic.
            # In the future we could update the new logic to support
            # capturing output, but it is probably not needed.
            if not with_output:
                return run_cmd_redirected(
                    final_cmd,
                    process_runner=self.process_runner,
                    silent=silent,
                    use_login_shells=self.call_context.is_using_login_shells(),
                    allow_interactive=self.call_context.does_allow_interactive(),
                    output_redirected=self.call_context.is_output_redirected(),
                    cmd_to_print=cmd_to_print
                )
            else:
                return self.process_runner.check_output(final_cmd)
        except subprocess.CalledProcessError as e:
            joined_cmd = " ".join(final_cmd if cmd_to_print is None else cmd_to_print)
            if (not self.call_context.is_using_login_shells()) or (
                    self.call_context.is_call_from_api()):
                raise ProcessRunnerError(
                    "Command failed",
                    "ssh_command_failed",
                    code=e.returncode,
                    command=joined_cmd,
                    output=e.output)

            if exit_on_fail:
                msg = "Command failed"
                if self.cli_logger.verbosity > 0:
                    msg += ":\n\n  {}\n".format(joined_cmd)
                else:
                    msg += ". Use -v for more details.".format(joined_cmd)
                raise click.ClickException(
                    msg) from None
            else:
                fail_msg = "SSH command failed."
                if self.call_context.is_output_redirected():
                    fail_msg += " See above for the output from the failure."
                raise click.ClickException(fail_msg) from None

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
                self.call_context, ssh_options_override_ssh_key, ProxyCommand=self.ssh_proxy_command)
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

        self.cli_logger.verbose("Running `{}`", cf.bold(cmd if cmd_to_print is None else cmd_to_print))
        with self.cli_logger.indented():
            self.cli_logger.verbose("Full command is `{}`",
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

    def _create_rsync_filter_args(self, options):
        rsync_excludes = options.get("rsync_exclude") or []
        rsync_filters = options.get("rsync_filter") or []

        exclude_args = [["--exclude", rsync_exclude]
                        for rsync_exclude in rsync_excludes]
        filter_args = [["--filter", "dir-merge,- {}".format(rsync_filter)]
                       for rsync_filter in rsync_filters]

        # Combine and flatten the two lists
        return [
            arg for args_list in exclude_args + filter_args
            for arg in args_list
        ]

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

    def bootstrap_data_disks(self) -> None:
        """Used to format and mount data disks on host."""
        block_devices = self._get_raw_block_devices()
        data_disk_index = 1
        for block_device in block_devices:
            self._format_and_mount(block_device, data_disk_index)
            data_disk_index += 1

    def _is_raw_block_device(self, block_device):
        # Read only or not a disk
        if (not isinstance(block_device["ro"], bool) and  block_device["ro"] != '0') \
                or block_device["ro"] is True or block_device["type"] != "disk":
            return False
        mount_point = block_device.get("mountpoint", None)
        if mount_point is not None and mount_point != "":
            # Mounted
            return False
        # Has children
        device_children = block_device.get("children", [])
        if len(device_children) > 0:
            return False

        return True

    def _get_raw_block_devices(self):
        self.run_with_retry("touch ~/.sudo_as_admin_successful")
        lsblk_output = self.run_with_retry(
            "lsblk -o name,ro,type,size,mountpoint -p --json || true",
            with_output=True).decode().strip()
        self.cli_logger.verbose("List of all block devices:\n{}", lsblk_output)

        block_devices_doc = json.loads(lsblk_output)
        block_devices = block_devices_doc.get("blockdevices", [])
        raw_block_devices = []
        for block_device in block_devices:
            if not self._is_raw_block_device(block_device):
                continue

            self.cli_logger.verbose("Found raw block devices {}", block_device["name"])
            raw_block_devices += [block_device]

        return raw_block_devices

    def _format_and_mount(self, block_device, data_disk_index):
        device_name = block_device["name"]
        mount_point = CLOUDTIK_DATA_DISK_MOUNT_POINT
        mount_path = f"{mount_point}/data_disk_{data_disk_index}"

        self.run_with_retry(
            "which mkfs.xfs > /dev/null || "
            "(sudo apt-get -qq update -y && "
            "sudo apt-get -qq install -y xfsprogs > /dev/null)")
        self.cli_logger.print("Formatting device {} and mount to {}...", device_name, mount_path)
        # Execute the format commands on the block device
        self.run_with_retry(f"sudo mkfs -t xfs -f {device_name}")
        self.run_with_retry(f"sudo mkdir -p {mount_path}")
        self.run_with_retry(f"sudo mount {device_name} {mount_path}")
        self.run_with_retry(f"sudo chmod a+w {mount_path}")
