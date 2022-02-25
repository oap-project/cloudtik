import logging
from typing import Any, Dict

from cloudtik.core.workspace_provider import WorkspaceProvider

logger = logging.getLogger(__name__)


class AzureWorkspaceProvider(WorkspaceProvider):
    def __init__(self, provider_config, workspace_name):
        WorkspaceProvider.__init__(self, provider_config, workspace_name)

    @staticmethod
    def validate_config(
            provider_config: Dict[str, Any]) -> None:
        return None
