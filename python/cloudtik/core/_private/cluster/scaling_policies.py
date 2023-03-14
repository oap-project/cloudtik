import json
import logging
from typing import Any, Dict, Optional
import time
import math

from cloudtik.core._private import constants
from cloudtik.core._private.state.control_state import ControlState
from cloudtik.core._private.utils import get_resource_demands_for_cpu, RUNTIME_CONFIG_KEY, \
    convert_nodes_to_cpus, get_resource_demands_for_memory, convert_nodes_to_memory
from cloudtik.core.scaling_policy import ScalingPolicy, ScalingState

logger = logging.getLogger(__name__)

SCALING_WITH_LOAD = "scaling-with-load"
SCALING_WITH_TIME = "scaling-with-time"

SCALING_WITH_LOAD_RESOURCE_CPU = constants.CLOUDTIK_RESOURCE_CPU
SCALING_WITH_LOAD_RESOURCE_MEMORY = constants.CLOUDTIK_RESOURCE_MEMORY

SCALING_WITH_LOAD_STEP_DEFAULT = 1
SCALING_WITH_LOAD_CPU_LOAD_THRESHOLD_DEFAULT = 0.85
SCALING_WITH_LOAD_MEMORY_LOAD_THRESHOLD_DEFAULT = 0.85
SCALING_WITH_LOAD_IN_USE_CPU_LOAD_THRESHOLD_DEFAULT = 0.10


class ScalingWithLoad(ScalingPolicy):
    def __init__(self,
                 config: Dict[str, Any],
                 head_ip: str) -> None:
        ScalingPolicy.__init__(self, config, head_ip)
        self.scaling_config = {}

        # scaling parameters
        self.scaling_step = SCALING_WITH_LOAD_STEP_DEFAULT
        self.scaling_resource = SCALING_WITH_LOAD_RESOURCE_CPU
        self.cpu_load_threshold = SCALING_WITH_LOAD_CPU_LOAD_THRESHOLD_DEFAULT
        self.memory_load_threshold = SCALING_WITH_LOAD_MEMORY_LOAD_THRESHOLD_DEFAULT
        self.in_use_cpu_load_threshold = SCALING_WITH_LOAD_IN_USE_CPU_LOAD_THRESHOLD_DEFAULT

        self.reset(config)

        self.last_state_time = 0
        self.last_resource_demands_time = 0
        self.last_resource_state_snapshot = None

        self.control_state = ControlState()
        self.control_state.initialize_control_state(
            head_ip, constants.CLOUDTIK_DEFAULT_PORT, constants.CLOUDTIK_REDIS_DEFAULT_PASSWORD)

    def reset(self, config):
        self.config = config
        runtime_config = config.get(RUNTIME_CONFIG_KEY, {})
        self.scaling_config = runtime_config.get("scaling", {})

        self.scaling_step = self.scaling_config.get(
            "scaling_step", SCALING_WITH_LOAD_STEP_DEFAULT)
        self.scaling_resource = self.scaling_config.get(
            "scaling_resource", SCALING_WITH_LOAD_RESOURCE_CPU)
        self.cpu_load_threshold = self.scaling_config.get(
            "cpu_load_threshold", SCALING_WITH_LOAD_CPU_LOAD_THRESHOLD_DEFAULT)
        self.memory_load_threshold = self.scaling_config.get(
            "memory_load_threshold", SCALING_WITH_LOAD_MEMORY_LOAD_THRESHOLD_DEFAULT)
        self.in_use_cpu_load_threshold = self.scaling_config.get(
            "in_use_cpu_load_threshold", SCALING_WITH_LOAD_IN_USE_CPU_LOAD_THRESHOLD_DEFAULT)

    def get_scaling_state(self) -> Optional[ScalingState]:
        self.last_state_time = time.time()

        node_metrics_table = self.control_state.get_node_metrics_table()
        all_node_metrics = self._get_all_node_metrics(node_metrics_table)

        autoscaling_instructions = self._get_autoscaling_instructions(
            all_node_metrics)
        node_resource_states, lost_nodes = self._get_node_resource_states(
            all_node_metrics)

        scaling_state = ScalingState()
        scaling_state.set_autoscaling_instructions(autoscaling_instructions)
        scaling_state.set_node_resource_states(node_resource_states)
        scaling_state.set_lost_nodes(lost_nodes)
        return scaling_state

    def _need_more_cores(self, cluster_metrics):
        num_cores = 0
        # check whether we need more cores based on the current CPU load
        cpu_load = cluster_metrics["cpu_load"]
        if cpu_load > self.cpu_load_threshold:
            num_cores = self.get_number_of_cores_to_scale(self.scaling_step)
        return num_cores

    def _need_more_memory(self, cluster_metrics):
        memory_to_scale = 0
        # check whether we need more cores based on the current memory load
        memory_load = cluster_metrics["memory_load"]
        if memory_load > self.memory_load_threshold:
            memory_to_scale = self.get_memory_to_scale(self.scaling_step)
        return memory_to_scale

    def _need_more_resources(self, cluster_metrics):
        requesting_resources = {}
        if self.scaling_resource == SCALING_WITH_LOAD_RESOURCE_CPU:
            requesting_cores = self._need_more_cores(cluster_metrics)
            requesting_resources[constants.CLOUDTIK_RESOURCE_CPU] = requesting_cores
        else:
            requesting_memory = self._need_more_memory(cluster_metrics)
            requesting_resources[constants.CLOUDTIK_RESOURCE_MEMORY] = requesting_memory

        return requesting_resources

    def get_number_of_cores_to_scale(self, scaling_step):
        return convert_nodes_to_cpus(self.config, scaling_step)

    def get_memory_to_scale(self, scaling_step):
        return convert_nodes_to_memory(self.config, scaling_step)

    def _get_autoscaling_instructions(self, all_node_metrics):
        autoscaling_instructions = {}
        resource_demands = []

        # Use the following information to make the decisions
        cluster_metrics = self._get_cluster_metrics(all_node_metrics)
        if cluster_metrics is not None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Cluster metrics: {}".format(cluster_metrics))

            resource_requesting = self._need_more_resources(cluster_metrics)
            if resource_requesting is not None:
                # We request more resources with a configured step of up scaling speed
                requesting_cores = resource_requesting.get(constants.CLOUDTIK_RESOURCE_CPU, 0)
                requesting_memory = resource_requesting.get(constants.CLOUDTIK_RESOURCE_MEMORY, 0)
                if requesting_cores > 0 or requesting_memory > 0:
                    if requesting_cores > 0:
                        resource_demands_for_cpu = get_resource_demands_for_cpu(
                            requesting_cores, self.config)
                        resource_demands += resource_demands_for_cpu

                        logger.info(
                            "Scaling event: utilization reaches to {} on total {} cores. "
                            "Requesting {} more cpus...".format(
                                cluster_metrics["cpu_load"],
                                cluster_metrics["total_cpus"],
                                requesting_cores))
                    elif requesting_memory > 0:
                        resource_demands_for_memory = get_resource_demands_for_memory(
                            requesting_memory, self.config)
                        resource_demands += resource_demands_for_memory

                        logger.info(
                            "Scaling event: utilization reaches to {} on total {} memory. "
                            "Requesting {} more memory...".format(
                                cluster_metrics["memory_load"],
                                cluster_metrics["total_memory"],
                                requesting_memory))

                    self.last_resource_demands_time = self.last_state_time
                    self.last_resource_state_snapshot = {
                        "total_cpus": cluster_metrics["total_cpus"],
                        "cpu_load": cluster_metrics["cpu_load"],
                        "total_memory": cluster_metrics["total_memory"],
                        "memory_load": cluster_metrics["memory_load"],
                        "resource_requesting": resource_requesting,
                    }

        autoscaling_instructions["demanding_time"] = self.last_state_time
        autoscaling_instructions["resource_demands"] = resource_demands
        if len(resource_demands) > 0 and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Resource demands: {}".format(resource_demands))

        return autoscaling_instructions

    def _get_node_resource_states(self, all_node_metrics):
        node_resource_states = {}
        lost_nodes = {}

        for node_metrics in all_node_metrics:
            node_id = node_metrics["node_id"]
            node_ip = node_metrics["node_ip"]
            if not node_id or not node_ip:
                continue

            # Filter out the stale record in the node table
            last_metrics_time = node_metrics.get("metrics_time", 0)
            delta = time.time() - last_metrics_time
            if delta >= constants.CLOUDTIK_HEARTBEAT_TIMEOUT_S:
                lost_nodes[node_id] = node_ip
                continue

            metrics = node_metrics.get("metrics")
            if not metrics:
                continue

            cpu_counts = metrics.get("cpus")
            cpu_percent = metrics.get("cpu")
            total_cpus = cpu_counts[0]

            load_avg = metrics.get("load_avg")
            load_avg_per_cpu = load_avg[1]
            load_avg_per_cpu_1 = load_avg_per_cpu[0]

            memory = metrics.get("mem")
            (total_memory, available_memory, percent_memory, used_memory) = memory

            total_resources = {
                constants.CLOUDTIK_RESOURCE_CPU: total_cpus,
                constants.CLOUDTIK_RESOURCE_MEMORY: total_memory
            }
            free_resources = {
                constants.CLOUDTIK_RESOURCE_CPU: int(total_cpus * (100 - cpu_percent)),
                constants.CLOUDTIK_RESOURCE_MEMORY: available_memory
            }

            resource_load = {
                "utilization": {
                    constants.CLOUDTIK_RESOURCE_CPU: load_avg_per_cpu_1,
                    constants.CLOUDTIK_RESOURCE_MEMORY: percent_memory,
                },
                "in_use": True if load_avg_per_cpu_1 > self.in_use_cpu_load_threshold else False
            }
            node_resource_state = {
                "node_id": node_id,
                "node_ip": node_ip,
                "resource_time": last_metrics_time,
                "total_resources": total_resources,
                "available_resources": free_resources,
                "resource_load": resource_load
            }
            logger.debug("Node metrics: {}".format(node_metrics))
            logger.debug("Node resources: {}".format(node_resource_state))
            node_resource_states[node_id] = node_resource_state

        return node_resource_states, lost_nodes

    def _get_all_node_metrics(self, node_metrics_table):
        node_metrics_rows = node_metrics_table.get_all().values()
        all_node_metrics = []
        for node_metrics_as_json in node_metrics_rows:
            node_metrics = json.loads(node_metrics_as_json)
            # filter out the head node
            if node_metrics["node_type"] == "head":
                continue

            all_node_metrics.append(node_metrics)
        return all_node_metrics

    def _get_cluster_metrics(self, all_node_metrics):
        cluster_total_cpus = 0
        cluster_total_memory = 0
        cluster_used_memory = 0
        cluster_load_avg_all_1 = 0.0
        for node_metrics in all_node_metrics:
            # Filter out the stale record in the node table
            last_metrics_time = node_metrics.get("metrics_time", 0)
            delta = time.time() - last_metrics_time
            if delta >= constants.CLOUDTIK_HEARTBEAT_TIMEOUT_S:
                continue

            metrics = node_metrics.get("metrics")
            if not metrics:
                continue

            cpu_counts = metrics.get("cpus")
            total_cpus = cpu_counts[0]

            load_avg = metrics.get("load_avg")
            load_avg_all = load_avg[1]
            load_avg_all_1 = load_avg_all[0]

            memory = metrics.get("mem")
            (total_memory, available_memory, percent_memory, used_memory) = memory

            cluster_total_cpus += total_cpus
            cluster_load_avg_all_1 += load_avg_all_1
            cluster_total_memory += total_memory
            cluster_used_memory += used_memory

        cluster_used_cpus = math.ceil(cluster_load_avg_all_1)
        cluster_cpu_load_1 = 0.0
        if cluster_total_cpus > 0:
            cluster_cpu_load_1 = round(cluster_load_avg_all_1 / cluster_total_cpus, 2)

        cluster_memory_load = 0.0
        if cluster_total_memory > 0:
            cluster_memory_load = round(cluster_used_memory / cluster_total_memory, 2)
        return {
            "total_cpus": cluster_total_cpus,
            "used_cpus": cluster_used_cpus,
            "available_cpus": min(0, cluster_total_cpus - cluster_used_cpus),
            "cpu_load": cluster_cpu_load_1,
            "total_memory": cluster_total_memory,
            "used_memory": cluster_used_memory,
            "available_memory": min(0, cluster_total_memory - cluster_used_memory),
            "memory_load": cluster_memory_load,
        }


def _create_scaling_policy(scaling_policy_name: str, config, head_ip):
    if SCALING_WITH_LOAD == scaling_policy_name:
        return ScalingWithLoad(config, head_ip)
    return None
