import logging
from typing import Any, Dict

from cloudtik.runtime.common.runtime_base import RuntimeBase
from cloudtik.runtime.nginx.utils import _get_runtime_processes, \
    _get_runtime_endpoints, _get_head_service_ports, _get_runtime_services

logger = logging.getLogger(__name__)


class NGINXRuntime(RuntimeBase):
    """Implementation for NGINX Runtime for Load Balancer"""

    def __init__(self, runtime_config: Dict[str, Any]) -> None:
        super().__init__(runtime_config)

    def get_runtime_endpoints(self, cluster_head_ip: str):
        return _get_runtime_endpoints(self.runtime_config, cluster_head_ip)

    def get_head_service_ports(self) -> Dict[str, Any]:
        return _get_head_service_ports(self.runtime_config)

    def get_runtime_services(self, cluster_name: str):
        return _get_runtime_services(self.runtime_config, cluster_name)

    @staticmethod
    def get_processes():
        return _get_runtime_processes()
