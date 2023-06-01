import logging
from typing import Any, Dict

from cloudtik.runtime.common.runtime_base import RuntimeBase
from cloudtik.runtime.ganglia.utils import _get_runtime_processes, \
    _get_runtime_services, _get_runtime_service_ports

logger = logging.getLogger(__name__)


class GangliaRuntime(RuntimeBase):
    """Implementation for Ganglia Runtime"""

    def __init__(self, runtime_config: Dict[str, Any]) -> None:
        super().__init__(runtime_config)

    def get_runtime_services(self, cluster_head_ip: str):
        return _get_runtime_services(cluster_head_ip)

    def get_runtime_service_ports(self) -> Dict[str, Any]:
        return _get_runtime_service_ports(self.runtime_config)

    @staticmethod
    def get_processes():
        """Return a list of processes for this runtime.
        Format:
        #1 Keyword to filter,
        #2 filter by command (True)/filter by args (False)
        #3 The third element is the process name.
        #4 The forth element, if node, the process should on all nodes, if head, the process should on head node.
        For example
        ["cloudtik_cluster_controller.py", False, "ClusterController", "head"],
        """
        return _get_runtime_processes()
