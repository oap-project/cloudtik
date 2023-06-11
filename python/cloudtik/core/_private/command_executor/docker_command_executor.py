from typing import Dict, List
import json
import logging
import os

from cloudtik.core._private.command_executor.command_executor import _with_environment_variables, _with_interactive, \
    _with_shutdown
from cloudtik.core._private.command_executor.local_command_executor import LocalCommandExecutor
from cloudtik.core._private.command_executor.ssh_command_executor import SSHCommandExecutor
from cloudtik.core.command_executor import CommandExecutor
from cloudtik.core._private.constants import \
    CLOUDTIK_DEFAULT_SHARED_MEMORY_MAX_BYTES, \
    CLOUDTIK_DATA_DISK_MOUNT_POINT
from cloudtik.core._private.docker import check_bind_mounts_cmd, \
    check_docker_running_cmd, \
    check_docker_image, \
    docker_start_cmds, \
    with_docker_exec, get_configured_docker_image, get_docker_cmd, with_docker_cmd, \
    get_docker_host_mount_location_for_object

logger = logging.getLogger(__name__)

# How long to wait for a node to start, in seconds
CHECK_DOCKER_RUNTIME_NUMBER_OF_RETRIES = 5


class DockerCommandExecutor(CommandExecutor):
    def __init__(self, call_context, docker_config, remote_host: bool = True, **common_args):
        CommandExecutor.__init__(self, call_context)
        self.host_command_executor = SSHCommandExecutor(
            call_context, **common_args) if remote_host else LocalCommandExecutor(
            call_context, **common_args)
        self.container_name = docker_config["container_name"]
        self.docker_config = docker_config
        self.home_dir = None
        self.initialized = False
        # Optionally use 'podman' instead of 'docker'
        use_podman = docker_config.get("use_podman", False)
        self.docker_cmd = "podman" if use_podman else "docker"
        # flag set at bootstrap
        self.docker_with_sudo = docker_config.get("docker_with_sudo", False)

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
                self.docker_cmd) == 0) else "docker"

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
        return self.host_command_executor.run(
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
        do_with_rsync = self._check_container_status() and not options.get(
                "docker_mount_if_possible", False)
        identical = False if do_with_rsync else True
        host_destination = get_docker_host_mount_location_for_object(
            self.host_command_executor.cluster_name, target,
            identical=identical)

        host_mount_location = os.path.dirname(host_destination.rstrip("/"))
        self.host_command_executor.run(
            f"mkdir -p {host_mount_location} && chown -R "
            f"{self.host_command_executor.ssh_user} {host_mount_location}",
            silent=self.call_context.is_rsync_silent())

        self.host_command_executor.run_rsync_up(
            source, host_destination, options=options)
        if do_with_rsync:
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
                docker_cmd=self.get_docker_cmd())[0]

            self.host_command_executor.run(
                "{} && rsync -e '{} exec -i' -avz {} {}:{}".format(
                    prefix, self.get_docker_cmd(), host_destination,
                    self.container_name, self._docker_expand_user(target)),
                silent=self.call_context.is_rsync_silent())

    def run_rsync_down(self, source, target, options=None):
        options = options or {}
        do_with_rsync = not options.get("docker_mount_if_possible", False)
        identical = False if do_with_rsync else True
        host_source = get_docker_host_mount_location_for_object(
            self.host_command_executor.cluster_name, source,
            identical=identical)
        host_mount_location = os.path.dirname(host_source.rstrip("/"))
        self.host_command_executor.run(
            f"mkdir -p {host_mount_location} && chown -R "
            f"{self.host_command_executor.ssh_user} {host_mount_location}",
            silent=self.call_context.is_rsync_silent())
        if source[-1] == "/":
            source += "."
            # Adding a "." means that docker copies the *contents*
            # Without it, docker copies the source *into* the target
        if do_with_rsync:
            # NOTE: `--delete` is okay here because the container is the source
            # of truth.
            self.host_command_executor.run(
                "rsync -e '{} exec -i' -avz --delete {}:{} {}".format(
                    self.get_docker_cmd(), self.container_name,
                    self._docker_expand_user(source), host_source),
                silent=self.call_context.is_rsync_silent())
        self.host_command_executor.run_rsync_down(
            host_source, target, options=options)

    def remote_shell_command_str(self):
        inner_str = self.host_command_executor.remote_shell_command_str().replace(
            "ssh", "ssh -tt", 1).strip("\n")
        return inner_str + " {} exec -it {} /bin/bash\n".format(
            self.get_docker_cmd(), self.container_name)

    def get_docker_cmd(self):
        return get_docker_cmd(self.docker_cmd, self.docker_with_sudo)

    def with_docker_cmd(self, cmd):
        return with_docker_cmd(cmd, self.docker_cmd, self.docker_with_sudo)

    def _check_docker_installed(self):
        no_exist = "NoExist"
        output = self.host_command_executor.run_with_retry(
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
        return self._is_container_running(self.container_name)

    def _is_container_running(self, container_name):
        output = self.host_command_executor.run_with_retry(
            check_docker_running_cmd(container_name, self.get_docker_cmd()),
            with_output=True).decode("utf-8").strip()
        # Checks for the false positive where "true" is in the container name
        return ("true" in output.lower()
                and "no such object" not in output.lower())

    def _docker_expand_user(self, string, any_char=False):
        user_pos = string.find("~")
        if user_pos > -1:
            if self.home_dir is None:
                self.home_dir = self.host_command_executor.run_with_retry(
                    "{} exec {} printenv HOME".format(
                        self.get_docker_cmd(), self.container_name),
                    with_output=True).decode("utf-8").strip()

            if any_char:
                return string.replace("~/", self.home_dir + "/")

            elif not any_char and user_pos == 0:
                return string.replace("~", self.home_dir, 1)

        return string

    def _check_if_container_restart_is_needed(
            self, image: str, cleaned_bind_mounts: Dict[str, str]) -> bool:
        re_init_required = False
        running_image = self.run_with_retry(
            check_docker_image(self.container_name, self.get_docker_cmd()),
            with_output=True,
            run_env="host").decode("utf-8").strip()
        if running_image != image:
            self.cli_logger.error(
                "A container with name {} is running image {} instead " +
                "of {} (which was provided in the YAML)", self.container_name,
                running_image, image)
        mounts = self.run_with_retry(
            check_bind_mounts_cmd(self.container_name, self.get_docker_cmd()),
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
                 shared_memory_ratio: float, sync_run_yet: bool):
        bootstrap_mounts = [
            "~/cloudtik_bootstrap_config.yaml", "~/cloudtik_bootstrap_key.pem"
        ]

        specific_image = get_configured_docker_image(self.docker_config, as_head)

        self._check_docker_installed()
        if self.docker_config.get("pull_before_run", True):
            assert specific_image, "Image must be included in config if " + \
                "pull_before_run is specified"
            self.run_with_retry(
                "{} pull {}".format(self.get_docker_cmd(), specific_image),
                run_env="host")
        else:
            self.run_with_retry(
                "{docker_cmd} image inspect {specific_image} "
                "1> /dev/null  2>&1 || "
                "{docker_cmd} pull {specific_image}".format(
                    docker_cmd=self.get_docker_cmd(), specific_image=specific_image))

        # Bootstrap files cannot be bind mounted because docker opens the
        # underlying inode. When the file is switched, docker becomes outdated.
        cleaned_bind_mounts = file_mounts.copy()
        for mnt in bootstrap_mounts:
            cleaned_bind_mounts.pop(mnt, None)

        docker_run_executed = False

        container_running = self._check_container_status()
        requires_re_init = False
        if container_running:
            requires_re_init = self._check_if_container_restart_is_needed(
                specific_image, cleaned_bind_mounts)
            if requires_re_init:
                self.run_with_retry(
                    "{} stop {} > /dev/null".format(
                        self.get_docker_cmd(), self.container_name),
                    run_env="host")

        if (not container_running) or requires_re_init:
            if not sync_run_yet:
                # Do not start the actual image as we need to run file_sync
                # first to ensure that all folders are created with the
                # correct ownership. Docker will create the folders with
                # `root` as the owner.
                return True
            # Get home directory
            image_env = self.host_command_executor.run_with_retry(
                "{} ".format(self.get_docker_cmd()) + "inspect -f '{{json .Config.Env}}' " +
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
                    "Unable to deserialize `image_env` to Python object. The `image_env` is:\n{}",
                    image_env
                )
                raise e

            host_data_disks = self._get_host_data_disks()
            user_docker_run_options = self.docker_config.get(
                "run_options", []) + self.docker_config.get(
                    f"{'head' if as_head else 'worker'}_run_options", [])
            start_command = docker_start_cmds(
                self.host_command_executor.ssh_user, specific_image,
                cleaned_bind_mounts, host_data_disks, self.container_name,
                self._configure_runtime(
                    self._auto_configure_shm(user_docker_run_options,
                                             shared_memory_ratio),
                    as_head),
                self.host_command_executor.cluster_name, home_directory,
                self.get_docker_cmd(),
                network=self.docker_config.get("network"),
                cpus=self.docker_config.get("cpus"),
                memory=self.docker_config.get("memory"),
                labels=self.docker_config.get("labels"),
                port_mappings=self.docker_config.get("port_mappings"),
                mounts_mapping=self.docker_config.get("mounts_mapping", True),
            )
            self.run_with_retry(
                start_command, run_env="host")
            docker_run_executed = True

        # Explicitly copy in bootstrap files.
        for mount in bootstrap_mounts:
            if mount in file_mounts:
                if not sync_run_yet:
                    # NOTE(ilr) This rsync is needed because when starting from
                    #  a stopped instance,  /tmp may be deleted and `run_init`
                    # is called before the first `file_sync` happens
                    self.run_rsync_up(file_mounts[mount], mount)
                self.host_command_executor.run_with_retry(
                    "rsync -e '{cmd} exec -i' -avz {src} {container}:{dst}".
                    format(
                        cmd=self.get_docker_cmd(),
                        src=get_docker_host_mount_location_for_object(
                                self.host_command_executor.cluster_name, mount),
                        container=self.container_name,
                        dst=self._docker_expand_user(mount)))
                try:
                    # Check if the current user has read permission.
                    # If they do not, try to change ownership!
                    self.run_with_retry(
                        f"cat {mount} >/dev/null 2>&1 || "
                        f"sudo chown $(id -u):$(id -g) {mount}")
                except Exception:
                    lsl_string = self.run_with_retry(
                        f"ls -l {mount}",
                        with_output=True).decode("utf-8").strip()
                    # The string is of format <Permission> <Links>
                    # <Owner> <Group> <Size> <Date> <Name>
                    permissions = lsl_string.split(" ")[0]
                    owner = lsl_string.split(" ")[2]
                    group = lsl_string.split(" ")[3]
                    current_user = self.run_with_retry(
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

    def run_terminate(self):
        """Stop the container if it is running"""
        container_running = self._check_container_status()
        if container_running:
            self.run_with_retry(
                "{} stop {} > /dev/null".format(
                    self.get_docker_cmd(), self.container_name),
                run_env="host")
        self.initialized = False

    def bootstrap_data_disks(self) -> None:
        """Used to format and mount data disks on host."""
        # For docker command executor, call directly on host command executor
        self.host_command_executor.bootstrap_data_disks()

    def _configure_runtime(self, run_options: List[str], as_head: bool) -> List[str]:
        if self.docker_config.get("disable_automatic_runtime_detection") or (
                as_head and self.docker_config.get("disable_head_automatic_runtime_detection", True)
        ):
            return run_options

        runtime_output = self.host_command_executor.run_with_retry(
            "{} ".format(self.get_docker_cmd()) + "info -f '{{.Runtimes}}' ",
            with_output=True).decode().strip()
        if "nvidia-container-runtime" in runtime_output:
            try:
                self.host_command_executor.run_with_retry(
                    "nvidia-smi", with_output=False,
                    number_of_retries=CHECK_DOCKER_RUNTIME_NUMBER_OF_RETRIES
                )
                return run_options + ["--runtime=nvidia"]
            except Exception as e:
                logger.warning(
                    "Nvidia Container Runtime is present, but no GPUs found.")
                logger.debug(f"nvidia-smi error: {e}")
                return run_options

        return run_options

    def _auto_configure_shm(self, run_options: List[str],
                            shared_memory_ratio: float) -> List[str]:
        if self.docker_config.get("disable_shm_size_detection"):
            return run_options
        for run_opt in run_options:
            if "--shm-size" in run_opt:
                logger.info("Bypassing automatic SHM-Detection because of "
                            f"`run_option`: {run_opt}")
                return run_options

        if shared_memory_ratio == 0:
            return run_options
        try:
            shm_output = self.host_command_executor.run_with_retry(
                "cat /proc/meminfo || true",
                with_output=True).decode().strip()
            available_memory = int([
                ln for ln in shm_output.split("\n") if "MemAvailable" in ln
            ][0].split()[1])
            available_memory_bytes = available_memory * 1024
            # Overestimate SHM size by 10%
            shm_size = int(min((available_memory_bytes *
                                shared_memory_ratio * 1.1),
                               CLOUDTIK_DEFAULT_SHARED_MEMORY_MAX_BYTES))
            if shm_size <= 0:
                return run_options

            return run_options + [f"--shm-size='{shm_size}b'"]
        except Exception as e:
            logger.warning(
                f"Received error while trying to auto-compute SHM size {e}")
            return run_options

    def _get_host_data_disks(self):
        mount_point = CLOUDTIK_DATA_DISK_MOUNT_POINT
        data_disks_string = self.run_with_retry(
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
            docker_cmd=self.get_docker_cmd())[0]
        cmd_to_print = with_docker_exec(
            [cmd_to_print],
            container_name=self.container_name,
            with_interactive=self.call_context.is_using_login_shells(),
            docker_cmd=self.get_docker_cmd())[0] if cmd_to_print else None
        return cmd, cmd_to_print
