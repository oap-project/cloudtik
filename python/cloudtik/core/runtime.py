import logging
from typing import Any, Dict, Optional, List, Tuple

from cloudtik.core.job_waiter import JobWaiter
from cloudtik.core.node_provider import NodeProvider
from cloudtik.core.scaling_policy import ScalingPolicy

logger = logging.getLogger(__name__)


class Runtime:
    """Interface for runtime abstraction.

    **Important**: This is an INTERNAL API that is only exposed for the purpose
    of implementing custom runtime. It is not allowed to call into
    RuntimeProvider methods from any package outside, only to
    define new implementations of RuntimeProvider for use with the "external" runtime
    provider option.
    """

    def __init__(self, runtime_config: Dict[str, Any]) -> None:
        self.runtime_config = runtime_config

    def prepare_config(self, cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare runtime specific configurations
        This happens after provider post_prepare is done.
        """
        return cluster_config

    def validate_config(self, cluster_config: Dict[str, Any], provider: NodeProvider):
        """Validate cluster configuration from runtime perspective.
        This happens after runtime prepare_config is done and before provider bootstrap_config
        """
        pass

    def verify_config(self, cluster_config: Dict[str, Any], provider: NodeProvider):
        """Verify cluster configuration at the last stage of bootstrap.
        This happens after provider bootstrap_config is done.
        The verification may mean a slow process to check with a server"""
        pass

    def with_environment_variables(
            self, config: Dict[str, Any], provider: NodeProvider,
            node_id: str) -> Dict[str, Any]:
        """Export necessary runtime environment variables for running node commands.
        For example: {"ENV_NAME": value}
        """
        return {}

    def get_runtime_shared_memory_ratio(
            self, config: Dict[str, Any], provider: NodeProvider,
            node_id: str) -> float:
        """Return the shared memory ratio for /dev/shm if needed.
        """
        return 0.0

    def cluster_booting_completed(
            self, cluster_config: Dict[str, Any], head_node_id: str) -> None:
        """This method is called after the cluster head is completed all the setup steps.
        """
        pass

    def get_runnable_command(self, target: str, runtime_options: Optional[List[str]]):
        """Return the runnable command for the target script.
        For example: ["bash", target]
        """
        return None

    def get_runtime_commands(self, cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a copy of runtime commands to run at different stages"""
        return None

    def get_defaults_config(self, cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a copy of runtime config"""
        return None

    def get_runtime_services(self, cluster_head_ip: str):
        """Return the user service information provided by this runtime
        It's a map of dictionary of service properties
        For example:
        {
            "app-web": {
                "name": "app web",
                "url": "http://localhost/app",
                "info": "additional information"
            },
        }
        """
        return None

    def get_runtime_service_ports(self) -> Dict[str, Any]:
        """Return a dictionary of service port with name as the key.
        For example:
            {
                "service-port-name": {
                    "protocol": "TCP",
                    "port": "1234",
                },
            }
        """
        return None

    def require_minimal_nodes(self, cluster_config: Dict[str, Any]) -> Tuple[int, int]:
        """Whether the runtime nodes need minimal nodes launch before going to setup.
        Usually this is because the setup of the nodes need to know each other.
        Return a tuple (whether to require minimal nodes, whether to fix the number of minimal nodes)
        """
        return False, False

    def minimal_nodes_reached(self, cluster_config: Dict[str, Any], nodes_info: Dict[str, Any]):
        """If the require_minimal_nodes_before_setup method returns True and runtime will be notified on head
        When the minimal nodes are reached. Please note this may call multiple times (for example server down and up)
        """
        pass

    def get_scaling_policy(self, cluster_config: Dict[str, Any], head_ip: str) -> Optional[ScalingPolicy]:
        """
        If the runtime has a resource management and configured to act resource scaling source
        return a scaling policy object to use by the cluster scaler.
        """
        return None

    def get_job_waiter(self, cluster_config: Dict[str, Any]) -> Optional[JobWaiter]:
        """
        If the runtime has job waiter for checking job completion, return a job waiter object.
        """
        return None

    @staticmethod
    def get_logs() -> Dict[str, str]:
        """Return a dictionary of name to log paths.
        For example {"server-a": "/tmp/server-a/logs"}
        """
        return {}

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
        return []

    @staticmethod
    def get_dependencies():
        """Return a list of runtimes which can be used by this runtime.
        This is for the purposes of reorder the installing, configuring and starting of the runtimes.
        If there is no such information, the runtime will installed and started in the user order
        """
        return []
