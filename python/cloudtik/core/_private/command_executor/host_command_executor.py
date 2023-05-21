from typing import List
import click
import json
import logging
import subprocess
import time

from cloudtik.core.command_executor import CommandExecutor
from cloudtik.core._private.constants import \
    CLOUDTIK_NODE_SSH_INTERVAL_S, \
    CLOUDTIK_DATA_DISK_MOUNT_POINT

from cloudtik.core._private.subprocess_output_util import (
    run_cmd_redirected, ProcessRunnerError)

from cloudtik.core._private.cli_logger import cf

logger = logging.getLogger(__name__)


class HostCommandExecutor(CommandExecutor):
    def __init__(self, call_context, log_prefix, auth_config,
                 cluster_name, process_runner, use_internal_ip,
                 provider, node_id):
        CommandExecutor.__init__(self, call_context)
        self.cluster_name = cluster_name
        self.log_prefix = log_prefix
        self.process_runner = process_runner
        self.use_internal_ip = use_internal_ip
        self.ssh_user = auth_config["ssh_user"]
        self.provider = provider
        self.node_id = node_id

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
