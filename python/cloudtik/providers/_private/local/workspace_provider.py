import logging
from typing import Any, Dict

from cloudtik.core.workspace_provider import WorkspaceProvider

logger = logging.getLogger(__name__)


class LocalWorkspaceProvider(WorkspaceProvider):
    def __init__(self, provider_config, workspace_name):
        WorkspaceProvider.__init__(self, provider_config, workspace_name)
