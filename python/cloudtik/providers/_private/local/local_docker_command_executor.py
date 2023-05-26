import copy
import logging
from typing import Dict, Optional

from cloudtik.core._private.command_executor.docker_command_executor import DockerCommandExecutor

logger = logging.getLogger(__name__)


class LocalDockerCommandExecutor(DockerCommandExecutor):
    def __init__(
            self, call_context, docker_config, remote_host: bool = True,
            local_file_mounts=None, **common_args):
        super().__init__(call_context, docker_config, remote_host, **common_args)
        self.local_file_mounts = local_file_mounts

    def run_init(self, *args, as_head: bool, file_mounts: Dict[str, str],
                 shared_memory_ratio: float, sync_run_yet: bool) -> Optional[bool]:
        # add additional file mounts for local node container start
        local_file_mounts = self.local_file_mounts
        if local_file_mounts:
            if file_mounts:
                file_mounts = copy.deepcopy(file_mounts)
                file_mounts.update(local_file_mounts)
            else:
                file_mounts = local_file_mounts
        return super().run_init(
            as_head=as_head,
            file_mounts=file_mounts,
            shared_memory_ratio=shared_memory_ratio,
            sync_run_yet=sync_run_yet)
