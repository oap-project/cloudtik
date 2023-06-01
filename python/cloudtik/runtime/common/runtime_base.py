import inspect
import logging
import os
from typing import Any, Dict

from cloudtik.core._private.utils import _get_runtime_config_object, merge_rooted_config_hierarchy
from cloudtik.core.runtime import Runtime

logger = logging.getLogger(__name__)


class RuntimeBase(Runtime):
    """A Runtime base class which by default get runtime commands and defaults config
    from the config folder in the same directory of the Runtime class.
    """

    def __init__(self, runtime_config: Dict[str, Any]) -> None:
        super().__init__(runtime_config)

    def get_runtime_commands(self, cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a copy of runtime commands to run at different stages"""
        return self._get_config_object_default(cluster_config, "commands")

    def get_defaults_config(self, cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a copy of runtime config"""
        return self._get_config_object_default(cluster_config, "defaults")

    def _get_config_object_default(
            self, cluster_config: Dict[str, Any], object_name: str) -> Dict[str, Any]:
        runtime_module = inspect.getmodule(self.__class__)
        runtime_home = os.path.dirname(runtime_module.__file__)
        config_root = os.path.join(runtime_home, "config")
        runtime_commands = _get_runtime_config_object(
            config_root, cluster_config["provider"], object_name)
        return merge_rooted_config_hierarchy(config_root, runtime_commands, object_name)
