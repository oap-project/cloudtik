import logging
import os
import time
from typing import Any, Dict, Optional

from cloudtik.core._private.core_utils import get_address_string
from cloudtik.core._private.services import address_to_ip
from cloudtik.core._private.state.state_utils import NODE_STATE_NODE_ID, NODE_STATE_NODE_IP, NODE_STATE_TIME
from cloudtik.core._private.utils import make_node_id, RUNTIME_CONFIG_KEY
from cloudtik.core.scaling_policy import ScalingPolicy, ScalingState

logger = logging.getLogger(__name__)

# The maximum allowed resource demand vector size to guarantee the resource
# demand scheduler bin packing algorithm takes a reasonable amount of time
# to run.
MAX_RESOURCE_DEMAND_VECTOR_SIZE = 1000


def _address_to_ip(address):
    try:
        return address_to_ip(address)
    except Exception:
        return None


def parse_resource_demands(resource_load_by_shape):
    """Handle the message.resource_load_by_shape protobuf for the demand
    based autoscaling. Catch and log all exceptions so this doesn't
    interfere with the utilization based autoscaler until we're confident
    this is stable. Worker queue backlogs are added to the appropriate
    resource demand vector.

    Args:
        resource_load_by_shape (pb2.gcs.ResourceLoad): The resource demands
            in protobuf form or None.

    Returns:
        List[ResourceDict]: Waiting bundles (ready and feasible).
        List[ResourceDict]: Infeasible bundles.
    """
    waiting_bundles, infeasible_bundles = [], []
    try:
        for resource_demand_pb in list(resource_load_by_shape.resource_demands):
            request_shape = dict(resource_demand_pb.shape)
            for _ in range(resource_demand_pb.num_ready_requests_queued):
                waiting_bundles.append(request_shape)
            for _ in range(resource_demand_pb.num_infeasible_requests_queued):
                infeasible_bundles.append(request_shape)

            # Infeasible and ready states for tasks are (logically)
            # mutually exclusive.
            if resource_demand_pb.num_infeasible_requests_queued > 0:
                backlog_queue = infeasible_bundles
            else:
                backlog_queue = waiting_bundles
            for _ in range(resource_demand_pb.backlog_size):
                backlog_queue.append(request_shape)
            if (
                    len(waiting_bundles + infeasible_bundles)
                    > MAX_RESOURCE_DEMAND_VECTOR_SIZE
            ):
                break
    except Exception:
        logger.exception("Failed to parse resource demands.")

    return waiting_bundles, infeasible_bundles


def get_resource_demands(waiting_bundles, infeasible_bundles, clip=True):
    if clip:
        # Bound the total number of bundles to
        # 2xMAX_RESOURCE_DEMAND_VECTOR_SIZE. This guarantees the resource
        # demand scheduler bin packing algorithm takes a reasonable amount
        # of time to run.
        return (
            waiting_bundles[:MAX_RESOURCE_DEMAND_VECTOR_SIZE]
            + infeasible_bundles[:MAX_RESOURCE_DEMAND_VECTOR_SIZE]
        )
    else:
        return waiting_bundles + infeasible_bundles


class RayScalingPolicy(ScalingPolicy):
    def __init__(self,
                 config: Dict[str, Any],
                 head_ip: str,
                 ray_port) -> None:
        ScalingPolicy.__init__(self, config, head_ip)

        # scaling parameters
        self.scaling_config = {}
        self.auto_scaling = False
        self._reset_ray_config()

        self.ray_port = ray_port
        self.last_state_time = 0

        self._init_gcs_client()

    def name(self):
        return "scaling-with-ray"

    def reset(self, config):
        super().reset(config)
        self._reset_ray_config()

    def _reset_ray_config(self):
        ray_config = self.config.get(RUNTIME_CONFIG_KEY, {}).get("ray", {})
        self.scaling_config = ray_config.get("scaling", {})
        # Update the scaling parameters
        self.auto_scaling = ray_config.get("auto_scaling", False)

    def get_scaling_state(self) -> Optional[ScalingState]:
        self.last_state_time = time.time()

        all_resource_usage = self._get_all_resource_usage()
        autoscaling_instructions = self._get_autoscaling_instructions(all_resource_usage)
        node_resource_states, lost_nodes = self._get_node_resource_states(all_resource_usage)

        scaling_state = ScalingState()
        scaling_state.set_autoscaling_instructions(autoscaling_instructions)
        scaling_state.set_node_resource_states(node_resource_states)
        scaling_state.set_lost_nodes(lost_nodes)
        return scaling_state

    def _get_autoscaling_instructions(self, all_resource_usage):
        if not self.auto_scaling:
            return None

        resources_batch_data = all_resource_usage.resource_usage_data
        waiting_bundles, infeasible_bundles = parse_resource_demands(
            resources_batch_data.resource_load_by_shape
        )
        resource_demands = get_resource_demands(
            waiting_bundles, infeasible_bundles)

        autoscaling_instructions = {
            "scaling_time": self.last_state_time,
            "resource_demands": resource_demands
        }
        if len(resource_demands) > 0:
            logger.debug("Resource demands: {}".format(resource_demands))
        return autoscaling_instructions

    def _get_node_resource_states(self, all_resource_usage):
        node_resource_states = {}
        lost_nodes = {}

        resources_batch_data = all_resource_usage.resource_usage_data
        for resource_message in resources_batch_data.batch:
            # ray_id = resource_message.node_id
            resource_load = dict(resource_message.resource_load)
            total_resources = dict(resource_message.resources_total)
            available_resources = dict(resource_message.resources_available)
            node_ip = resource_message.node_manager_address
            node_id = make_node_id(node_ip)

            node_resource_state = {
                NODE_STATE_NODE_ID: node_id,
                NODE_STATE_NODE_IP: node_ip,
                NODE_STATE_TIME: self.last_state_time,
                "total_resources": total_resources,
                "available_resources": available_resources,
                "resource_load": resource_load
            }
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Node resources: {}".format(node_resource_state))
            node_resource_states[node_id] = node_resource_state

        # if the lost nodes appears in RUNNING, exclude it
        lost_nodes = {
            node_id: lost_nodes[node_id] for node_id in lost_nodes if node_id not in node_resource_states
        }

        return node_resource_states, lost_nodes

    def _init_gcs_client(self):
        import ray
        import ray._private.ray_constants as ray_constants
        from ray.core.generated import gcs_pb2, gcs_service_pb2, gcs_service_pb2_grpc

        gcs_address = get_address_string(self.head_ip, self.ray_port)
        options = ray_constants.GLOBAL_GRPC_OPTIONS
        gcs_channel = ray._private.utils.init_grpc_channel(gcs_address, options)
        self.gcs_node_resources_stub = (
            gcs_service_pb2_grpc.NodeResourceInfoGcsServiceStub(gcs_channel)
        )

    def _get_all_resource_usage(self):
        from ray.core.generated import gcs_pb2, gcs_service_pb2, gcs_service_pb2_grpc

        def log_resource_batch_data_if_desired(
                resources_batch_data: gcs_pb2.ResourceUsageBatchData,
        ) -> None:
            if os.getenv("RAY_LOG_RESOURCE_BATCH_DATA") == "1":
                logger.info("Logging raw resource message pulled from GCS.")
                logger.info(resources_batch_data)
                logger.info("Done logging raw resource message.")

        request = gcs_service_pb2.GetAllResourceUsageRequest()
        response = self.gcs_node_resources_stub.GetAllResourceUsage(request, timeout=60)
        resources_batch_data = response.resource_usage_data

        log_resource_batch_data_if_desired(resources_batch_data)
        return response
