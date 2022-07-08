import copy
from getpass import getuser
from shlex import quote
from typing import Dict, List
import click
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import warnings

from cloudtik.core.command_executor import CommandExecutor
from cloudtik.core._private.constants import \
                                     CLOUDTIK_NODE_SSH_INTERVAL_S, \
                                     CLOUDTIK_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES, \
                                     CLOUDTIK_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION, \
                                     CLOUDTIK_NODE_START_WAIT_S,\
                                     CLOUDTIK_DATA_DISK_MOUNT_POINT
from cloudtik.core._private.docker import check_bind_mounts_cmd, \
                                  check_docker_running_cmd, \
                                  check_docker_image, \
                                  docker_start_cmds, \
                                  with_docker_exec
from cloudtik.core._private.log_timer import LogTimer

from cloudtik.core._private.subprocess_output_util import (
    run_cmd_redirected, ProcessRunnerError)

from cloudtik.core._private.cli_logger import cf
from cloudtik.core._private.debug import log_once

logger = logging.getLogger(__name__)

# How long to wait for a node to start, in seconds
HASH_MAX_LENGTH = 10
KUBECTL_RSYNC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "providers/_private/_kubernetes/kubectl-rsync.sh")
MAX_HOME_RETRIES = 3
HOME_RETRY_DELAY_S = 5

PRIVACY_KEYWORDS = ["PASSWORD", "ACCOUNT", "SECRET", "ACCESS_KEY", "PRIVATE_KEY", "PROJECT_ID"]

PRIVACY_REPLACEMENT = "VALUE-PROTECTED"
PRIVACY_REPLACEMENT_TEMPLATE = "VALUE-{}PROTECTED"


def is_key_with_privacy(key: str):
    for keyword in PRIVACY_KEYWORDS:
        if keyword in key:
            return True
    return False


def _with_environment_variables(cmd: str,
                                environment_variables: Dict[str, object],
                                cmd_to_print: str = None):
    """Prepend environment variables to a shell command.

    Args:
        cmd (str): The base command.
        environment_variables (Dict[str, object]): The set of environment
            variables. If an environment variable value is a dict, it will
            automatically be converted to a one line yaml string.
        cmd_to_print (str): The command to print for base command if there is one
    """

    as_strings = []
    as_strings_to_print = []
    with_privacy = False
    for key, val in environment_variables.items():
        # json.dumps will add an extra quote to string value
        # since we use quote to make sure value is safe for shell, we don't need the quote for string
        escaped_val = json.dumps(val, separators=(",", ":"))
        if isinstance(val, str):
            escaped_val = escaped_val.strip("\"\'")

        s = "export {}={};".format(key, quote(escaped_val))
        as_strings.append(s)

        if is_key_with_privacy(key):
            with_privacy = True
            val_len = len(escaped_val)
            replacement_len = len(PRIVACY_REPLACEMENT)
            if val_len > replacement_len:
                escaped_val = PRIVACY_REPLACEMENT_TEMPLATE.format("-" * (val_len - replacement_len))
            else:
                escaped_val = PRIVACY_REPLACEMENT
            s = "export {}={};".format(key, quote(escaped_val))

        as_strings_to_print.append(s)

    all_vars = "".join(as_strings)
    cmd_with_vars = all_vars + cmd

    cmd_with_vars_to_print = None
    if cmd_to_print or with_privacy:
        all_vars_to_print = "".join(as_strings_to_print)
        cmd_with_vars_to_print = all_vars_to_print + (cmd if cmd_to_print is None else cmd_to_print)
    return cmd_with_vars, cmd_with_vars_to_print


def _with_shutdown(cmd, cmd_to_print=None):
    cmd += "; sudo shutdown -h now"
    if cmd_to_print:
        cmd_to_print += "; sudo shutdown -h now"
    return cmd, cmd_to_print


def _with_interactive(cmd):
    force_interactive = (
        f"true && source ~/.bashrc && "
        f"export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore && ({cmd})")
    return ["bash", "--login", "-c", "-i", quote(force_interactive)]


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
                port_forward_cmd) + " failed with error: " + perr
            raise Exception(exception_str)
        else:
            final_cmd = self.kubectl + ["exec", "-it"]
            final_cmd += [
                self.node_id,
                "--",
            ]
            if environment_variables:
                cmd, cmd_to_print = _with_environment_variables(
                    cmd, environment_variables, cmd_to_print=cmd_to_print)
            cmd = _with_interactive(cmd)
            cmd_to_print = _with_interactive(cmd_to_print) if cmd_to_print else None
            cmd_prefix = " ".join(final_cmd)
            final_cmd += cmd
            final_cmd_to_print = None
            if cmd_to_print:
                final_cmd_to_print = copy.deepcopy(final_cmd)
                final_cmd_to_print += cmd_to_print
            # `kubectl exec` + subprocess w/ list of args has unexpected
            # side-effects.
            final_cmd = " ".join(final_cmd)
            final_cmd_to_print = " ".join(final_cmd_to_print) if final_cmd_to_print else None

            self.cli_logger.verbose("Running `{}`", cf.bold(" ".join(cmd if cmd_to_print is None else cmd_to_print)))
            with self.cli_logger.indented():
                self.cli_logger.verbose("Full command is `{}`",
                                   cf.bold(final_cmd if final_cmd_to_print is None else final_cmd_to_print))
            try:
                if with_output:
                    return self.process_runner.check_output(
                        final_cmd, shell=True)
                else:
                    self.process_runner.check_call(final_cmd, shell=True)
            except subprocess.CalledProcessError:
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
            if target.startswith("~"):
                target = self._home + target[1:]

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
            if target.startswith("~"):
                target = self._home + target[1:]

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
        # TODO (Dmitri): Think about how to use the node's HOME variable
        # without making an extra kubectl exec call.
        cmd = self.kubectl + [
            "exec", "-it", self.node_id, "--", "printenv", "HOME"
        ]
        joined_cmd = " ".join(cmd)
        raw_out = self.process_runner.check_output(joined_cmd, shell=True)
        home = raw_out.decode().strip("\n\r")
        return home


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
            if with_output:
                return self.process_runner.check_output(final_cmd)
            else:
                return self.process_runner.check_call(final_cmd)
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
        self.run("touch ~/.sudo_as_admin_successful")
        lsblk_output = self.run(
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

        self.cli_logger.print("Formatting device {} and mount to {}...", device_name, mount_path)

        # Execute the format commands on the block device
        self.run(f"sudo mkfs -t xfs -f {device_name}")
        self.run(f"sudo mkdir -p {mount_path}")
        self.run(f"sudo mount {device_name} {mount_path}")
        self.run(f"sudo chmod a+w {mount_path}")


class DockerCommandExecutor(CommandExecutor):
    def __init__(self, call_context, docker_config, **common_args):
        CommandExecutor.__init__(self, call_context)
        self.ssh_command_executor = SSHCommandExecutor(call_context, **common_args)
        self.container_name = docker_config["container_name"]
        self.docker_config = docker_config
        self.home_dir = None
        self.initialized = False
        # Optionally use 'podman' instead of 'docker'
        use_podman = docker_config.get("use_podman", False)
        self.docker_cmd = "podman" if use_podman else "docker"

    def run(
            self,
            cmd=None,
            timeout=120,
            exit_on_fail=False,
            port_forward=None,
            with_output=False,
            environment_variables: Dict[str, object] = None,
            run_env="auto",
            ssh_options_override_ssh_key="",
            shutdown_after_run=False,
            cmd_to_print=None,
            silent=False
    ):
        if run_env == "auto":
            run_env = "host" if (not bool(cmd) or cmd.find(
                self.docker_cmd) == 0) else self.docker_cmd

        if environment_variables:
            cmd, cmd_to_print = _with_environment_variables(
                cmd, environment_variables, cmd_to_print=cmd_to_print)

        if run_env == "docker":
            cmd = self._docker_expand_user(cmd, any_char=True)
            cmd_to_print = self._docker_expand_user(cmd_to_print, any_char=True) if cmd_to_print else None
            if self.call_context.is_using_login_shells():
                cmd = " ".join(_with_interactive(cmd))
                cmd_to_print = " ".join(_with_interactive(cmd_to_print)) if cmd_to_print else None
            cmd, cmd_to_print = self._with_docker_exec(cmd, cmd_to_print)

        if shutdown_after_run:
            # shutdown should run after `with_docker_exec` command above
            cmd, cmd_to_print = _with_shutdown(cmd, cmd_to_print)

        # Do not pass shutdown_after_run argument to ssh_command_runner.run()
        # since it is handled above.
        return self.ssh_command_executor.run(
            cmd,
            timeout=timeout,
            exit_on_fail=exit_on_fail,
            port_forward=port_forward,
            with_output=with_output,
            ssh_options_override_ssh_key=ssh_options_override_ssh_key,
            cmd_to_print=cmd_to_print,
            silent=silent)

    def run_rsync_up(self, source, target, options=None):
        options = options or {}
        host_destination = os.path.join(
            self._get_docker_host_mount_location(
                self.ssh_command_executor.cluster_name), target.lstrip("/"))

        host_mount_location = os.path.dirname(host_destination.rstrip("/"))
        self.ssh_command_executor.run(
            f"mkdir -p {host_mount_location} && chown -R "
            f"{self.ssh_command_executor.ssh_user} {host_mount_location}",
            silent=self.call_context.is_rsync_silent())

        self.ssh_command_executor.run_rsync_up(
            source, host_destination, options=options)
        if self._check_container_status() and not options.get(
                "docker_mount_if_possible", False):
            if os.path.isdir(source):
                # Adding a "." means that docker copies the *contents*
                # Without it, docker copies the source *into* the target
                host_destination += "/."

            # This path may not exist inside the container. This ensures
            # that the path is created!
            prefix = with_docker_exec(
                [
                    "mkdir -p {}".format(
                        os.path.dirname(self._docker_expand_user(target)))
                ],
                container_name=self.container_name,
                with_interactive=self.call_context.is_using_login_shells(),
                docker_cmd=self.docker_cmd)[0]

            self.ssh_command_executor.run(
                "{} && rsync -e '{} exec -i' -avz {} {}:{}".format(
                    prefix, self.docker_cmd, host_destination,
                    self.container_name, self._docker_expand_user(target)),
                silent=self.call_context.is_rsync_silent())

    def run_rsync_down(self, source, target, options=None):
        options = options or {}
        host_source = os.path.join(
            self._get_docker_host_mount_location(
                self.ssh_command_executor.cluster_name), source.lstrip("/"))
        host_mount_location = os.path.dirname(host_source.rstrip("/"))
        self.ssh_command_executor.run(
            f"mkdir -p {host_mount_location} && chown -R "
            f"{self.ssh_command_executor.ssh_user} {host_mount_location}",
            silent=self.call_context.is_rsync_silent())
        if source[-1] == "/":
            source += "."
            # Adding a "." means that docker copies the *contents*
            # Without it, docker copies the source *into* the target
        if not options.get("docker_mount_if_possible", False):
            # NOTE: `--delete` is okay here because the container is the source
            # of truth.
            self.ssh_command_executor.run(
                "rsync -e '{} exec -i' -avz --delete {}:{} {}".format(
                    self.docker_cmd, self.container_name,
                    self._docker_expand_user(source), host_source),
                silent=self.call_context.is_rsync_silent())
        self.ssh_command_executor.run_rsync_down(
            host_source, target, options=options)

    def remote_shell_command_str(self):
        inner_str = self.ssh_command_executor.remote_shell_command_str().replace(
            "ssh", "ssh -tt", 1).strip("\n")
        return inner_str + " {} exec -it {} /bin/bash\n".format(
            self.docker_cmd, self.container_name)

    def _check_docker_installed(self):
        no_exist = "NoExist"
        output = self.ssh_command_executor.run(
            f"command -v {self.docker_cmd} || echo '{no_exist}'",
            with_output=True)
        cleaned_output = output.decode().strip()
        if no_exist in cleaned_output or "docker" not in cleaned_output:
            if self.docker_cmd == "docker":
                install_commands = [
                    "curl -fsSL https://get.docker.com -o get-docker.sh",
                    "sudo sh get-docker.sh", "sudo usermod -aG docker $USER",
                    "sudo systemctl restart docker -f"
                ]
            else:
                install_commands = [
                    "sudo apt-get update", "sudo apt-get -y install podman"
                ]

            logger.error(
                f"{self.docker_cmd.capitalize()} not installed. You can "
                f"install {self.docker_cmd.capitalize()} by adding the "
                "following commands to 'initialization_commands':\n" +
                "\n".join(install_commands))

    def _check_container_status(self):
        if self.initialized:
            return True
        output = self.ssh_command_executor.run(
            check_docker_running_cmd(self.container_name, self.docker_cmd),
            with_output=True).decode("utf-8").strip()
        # Checks for the false positive where "true" is in the container name
        return ("true" in output.lower()
                and "no such object" not in output.lower())

    def _docker_expand_user(self, string, any_char=False):
        user_pos = string.find("~")
        if user_pos > -1:
            if self.home_dir is None:
                self.home_dir = self.ssh_command_executor.run(
                    f"{self.docker_cmd} exec {self.container_name} "
                    "printenv HOME",
                    with_output=True).decode("utf-8").strip()

            if any_char:
                return string.replace("~/", self.home_dir + "/")

            elif not any_char and user_pos == 0:
                return string.replace("~", self.home_dir, 1)

        return string

    def _check_if_container_restart_is_needed(
            self, image: str, cleaned_bind_mounts: Dict[str, str]) -> bool:
        re_init_required = False
        running_image = self.run(
            check_docker_image(self.container_name, self.docker_cmd),
            with_output=True,
            run_env="host").decode("utf-8").strip()
        if running_image != image:
            self.cli_logger.error(
                "A container with name {} is running image {} instead " +
                "of {} (which was provided in the YAML)", self.container_name,
                running_image, image)
        mounts = self.run(
            check_bind_mounts_cmd(self.container_name, self.docker_cmd),
            with_output=True,
            run_env="host").decode("utf-8").strip()
        try:
            active_mounts = json.loads(mounts)
            active_remote_mounts = {
                mnt["Destination"].strip("/")
                for mnt in active_mounts
            }
            # Ignore bootstrap files.
            requested_remote_mounts = {
                self._docker_expand_user(remote).strip("/")
                for remote in cleaned_bind_mounts.keys()
            }
            unfulfilled_mounts = (
                requested_remote_mounts - active_remote_mounts)
            if unfulfilled_mounts:
                re_init_required = True
                self.cli_logger.warning(
                    "This Docker Container is already running. "
                    "Restarting the Docker container on "
                    "this node to pick up the following file_mounts {}",
                    unfulfilled_mounts)
        except json.JSONDecodeError:
            self.cli_logger.verbose(
                "Unable to check if file_mounts specified in the YAML "
                "differ from those on the running container.")
        return re_init_required

    def run_init(self, *, as_head: bool, file_mounts: Dict[str, str],
                 sync_run_yet: bool):
        BOOTSTRAP_MOUNTS = [
            "~/cloudtik_bootstrap_config.yaml", "~/cloudtik_bootstrap_key.pem"
        ]

        specific_image = self.docker_config.get(
            f"{'head' if as_head else 'worker'}_image",
            self.docker_config.get("image"))

        self._check_docker_installed()
        if self.docker_config.get("pull_before_run", True):
            assert specific_image, "Image must be included in config if " + \
                "pull_before_run is specified"
            self.run(
                "{} pull {}".format(self.docker_cmd, specific_image),
                run_env="host")
        else:

            self.run(f"{self.docker_cmd} image inspect {specific_image} "
                     "1> /dev/null  2>&1 || "
                     f"{self.docker_cmd} pull {specific_image}")

        # Bootstrap files cannot be bind mounted because docker opens the
        # underlying inode. When the file is switched, docker becomes outdated.
        cleaned_bind_mounts = file_mounts.copy()
        for mnt in BOOTSTRAP_MOUNTS:
            cleaned_bind_mounts.pop(mnt, None)

        docker_run_executed = False

        container_running = self._check_container_status()
        requires_re_init = False
        if container_running:
            requires_re_init = self._check_if_container_restart_is_needed(
                specific_image, cleaned_bind_mounts)
            if requires_re_init:
                self.run(
                    f"{self.docker_cmd} stop {self.container_name}",
                    run_env="host")

        if (not container_running) or requires_re_init:
            if not sync_run_yet:
                # Do not start the actual image as we need to run file_sync
                # first to ensure that all folders are created with the
                # correct ownership. Docker will create the folders with
                # `root` as the owner.
                return True
            # Get home directory
            image_env = self.ssh_command_executor.run(
                f"{self.docker_cmd} " + "inspect -f '{{json .Config.Env}}' " +
                specific_image,
                with_output=True).decode().strip()
            home_directory = "/root"
            try:
                for env_var in json.loads(image_env):
                    if env_var.startswith("HOME="):
                        home_directory = env_var.split("HOME=")[1]
                        break
            except json.JSONDecodeError as e:
                self.cli_logger.error(
                    "Unable to deserialize `image_env` to Python object. "
                    f"The `image_env` is:\n{image_env}"
                )
                raise e

            host_data_disks = self._get_host_data_disks()
            user_docker_run_options = self.docker_config.get(
                "run_options", []) + self.docker_config.get(
                    f"{'head' if as_head else 'worker'}_run_options", [])
            start_command = docker_start_cmds(
                self.ssh_command_executor.ssh_user, specific_image,
                cleaned_bind_mounts, host_data_disks, self.container_name,
                self._configure_runtime(
                    self._auto_configure_shm(user_docker_run_options)),
                self.ssh_command_executor.cluster_name, home_directory,
                self.docker_cmd)
            self.run(start_command, run_env="host")
            docker_run_executed = True

        # Explicitly copy in bootstrap files.
        for mount in BOOTSTRAP_MOUNTS:
            if mount in file_mounts:
                if not sync_run_yet:
                    # NOTE(ilr) This rsync is needed because when starting from
                    #  a stopped instance,  /tmp may be deleted and `run_init`
                    # is called before the first `file_sync` happens
                    self.run_rsync_up(file_mounts[mount], mount)
                self.ssh_command_executor.run(
                    "rsync -e '{cmd} exec -i' -avz {src} {container}:{dst}".
                    format(
                        cmd=self.docker_cmd,
                        src=os.path.join(
                            self._get_docker_host_mount_location(
                                self.ssh_command_executor.cluster_name), mount),
                        container=self.container_name,
                        dst=self._docker_expand_user(mount)))
                try:
                    # Check if the current user has read permission.
                    # If they do not, try to change ownership!
                    self.run(f"cat {mount} >/dev/null 2>&1 || "
                             f"sudo chown $(id -u):$(id -g) {mount}")
                except Exception:
                    lsl_string = self.run(
                        f"ls -l {mount}",
                        with_output=True).decode("utf-8").strip()
                    # The string is of format <Permission> <Links>
                    # <Owner> <Group> <Size> <Date> <Name>
                    permissions = lsl_string.split(" ")[0]
                    owner = lsl_string.split(" ")[2]
                    group = lsl_string.split(" ")[3]
                    current_user = self.run(
                        "whoami", with_output=True).decode("utf-8").strip()
                    self.cli_logger.warning(
                        f"File ({mount}) is owned by user:{owner} and group:"
                        f"{group} with permissions ({permissions}). The "
                        f"current user ({current_user}) does not have "
                        "permission to read these files, and we may not be "
                        "able to scale. This can be resolved by "
                        "installing `sudo` in your container, or adding a "
                        f"command like 'chown {current_user} {mount}' to "
                        "your `setup_commands`.")
        self.initialized = True
        return docker_run_executed

    def bootstrap_data_disks(self) -> None:
        """Used to format and mount data disks on host."""
        # For docker command executor, call directly on host command executor
        self.ssh_command_executor.bootstrap_data_disks()

    def _configure_runtime(self, run_options: List[str]) -> List[str]:
        if self.docker_config.get("disable_automatic_runtime_detection"):
            return run_options

        runtime_output = self.ssh_command_executor.run(
            f"{self.docker_cmd} " + "info -f '{{.Runtimes}}' ",
            with_output=True).decode().strip()
        if "nvidia-container-runtime" in runtime_output:
            try:
                self.ssh_command_executor.run("nvidia-smi", with_output=False)
                return run_options + ["--runtime=nvidia"]
            except Exception as e:
                logger.warning(
                    "Nvidia Container Runtime is present, but no GPUs found.")
                logger.debug(f"nvidia-smi error: {e}")
                return run_options

        return run_options

    def _auto_configure_shm(self, run_options: List[str]) -> List[str]:
        if self.docker_config.get("disable_shm_size_detection"):
            return run_options
        for run_opt in run_options:
            if "--shm-size" in run_opt:
                logger.info("Bypassing automatic SHM-Detection because of "
                            f"`run_option`: {run_opt}")
                return run_options
        try:
            shm_output = self.ssh_command_executor.run(
                "cat /proc/meminfo || true",
                with_output=True).decode().strip()
            available_memory = int([
                ln for ln in shm_output.split("\n") if "MemAvailable" in ln
            ][0].split()[1])
            available_memory_bytes = available_memory * 1024
            # Overestimate SHM size by 10%
            shm_size = int(min((available_memory_bytes *
                            CLOUDTIK_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION * 1.1),
                           CLOUDTIK_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES))
            if shm_size <= 0:
                return run_options

            return run_options + [f"--shm-size='{shm_size}b'"]
        except Exception as e:
            logger.warning(
                f"Received error while trying to auto-compute SHM size {e}")
            return run_options

    def _get_docker_host_mount_location(self, cluster_name: str) -> str:
        """Return the docker host mount directory location."""
        # Imported here due to circular dependency in imports.
        from cloudtik.core.api import get_docker_host_mount_location
        return get_docker_host_mount_location(cluster_name)

    def _get_host_data_disks(self):
        mount_point = CLOUDTIK_DATA_DISK_MOUNT_POINT
        data_disks_string = self.run(
            "([ -d {} ] && ls --color=no {}) || true".format(mount_point, mount_point),
            with_output=True,
            run_env="host").decode("utf-8").strip()
        if data_disks_string is None or data_disks_string == "":
            return []

        data_disks = data_disks_string.split()
        return ["{}/{}".format(mount_point, data_disks) for data_disks in data_disks]

    def _with_docker_exec(self, cmd, cmd_to_print=None):
        cmd = with_docker_exec(
            [cmd],
            container_name=self.container_name,
            with_interactive=self.call_context.is_using_login_shells(),
            docker_cmd=self.docker_cmd)[0]
        cmd_to_print = with_docker_exec(
            [cmd_to_print],
            container_name=self.container_name,
            with_interactive=self.call_context.is_using_login_shells(),
            docker_cmd=self.docker_cmd)[0] if cmd_to_print else None
        return cmd, cmd_to_print
