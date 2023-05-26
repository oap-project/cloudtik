import logging
from typing import Dict, Optional

from cloudtik.core._private.command_executor.docker_command_executor import DockerCommandExecutor

logger = logging.getLogger(__name__)


class VirtualDockerCommandExecutor(DockerCommandExecutor):
    def __init__(
            self, call_context, docker_config, remote_host: bool = True, **common_args):
        super().__init__(call_context, docker_config, remote_host, **common_args)

    def run_rsync_up(self, source, target, options=None):
        # since docker has been started as part of node creation
        # we need to reset docker_mount_if_possible rsync all the mounts
        if options is not None and options.get(
                "docker_mount_if_possible", False):
            options["docker_mount_if_possible"] = False
        super().run_rsync_up(source, target, options)

    def run_init(self, *, as_head: bool, file_mounts: Dict[str, str],
                 shared_memory_ratio: float, sync_run_yet: bool) -> Optional[bool]:
        pass

    def run_terminate(self):
        pass

    def start_container(
            self, as_head: bool, file_mounts: Dict[str, str],
            shared_memory_ratio: float) -> Optional[bool]:
        return super().run_init(
            as_head=as_head,
            file_mounts=file_mounts,
            shared_memory_ratio=shared_memory_ratio,
            sync_run_yet=True)

    def stop_container(self):
        return super().run_terminate()

    def run_docker_cmd(self, cmd, with_output=True):
        final_cmd = self.with_docker_cmd(cmd)
        output = self.run_with_retry(
            final_cmd,
            run_env="host",
            with_output=with_output)
        if output is None:
            return None
        return output.decode("utf-8").strip()
