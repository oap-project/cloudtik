import json
import logging
from typing import Any, Dict, Optional
import time
import math

from cloudtik.core import tags
from cloudtik.core._private import constants
from cloudtik.core._private.state.control_state import ControlState
from cloudtik.core._private.utils import get_resource_demands_for_cpu, RUNTIME_CONFIG_KEY, \
    convert_nodes_to_cpus, get_resource_demands_for_memory, convert_nodes_to_memory, get_resource_requests_for_cpu, \
    _sum_min_workers
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

SCALING_WITH_TIME_MATH_ON_MIN_WORKERS = "on-min-workers"
SCALING_WITH_TIME_MATH_ON_PREVIOUS_TIME = "on-previous-time"

SCALING_WITH_TIME_PERIODIC_DAILY = "daily"
SCALING_WITH_TIME_PERIODIC_WEEKLY = "weekly"
SCALING_WITH_TIME_PERIODIC_MONTHLY = "monthly"


class ScalingWithResources(ScalingPolicy):
    def __init__(self,
                 config: Dict[str, Any],
                 head_ip: str) -> None:
        ScalingPolicy.__init__(self, config, head_ip)
        self.last_state_time = 0
        self.control_state = ControlState()
        self.control_state.initialize_control_state(
            head_ip, constants.CLOUDTIK_DEFAULT_PORT, constants.CLOUDTIK_REDIS_DEFAULT_PASSWORD)

        self.scaling_config = {}
        self.in_use_cpu_load_threshold = SCALING_WITH_LOAD_IN_USE_CPU_LOAD_THRESHOLD_DEFAULT
        self._reset_resources_config()

    def name(self):
        return "scaling-with-resources"

    def reset(self, config):
        super().reset(config)
        self._reset_resources_config()

    def _reset_resources_config(self):
        runtime_config = self.config.get(RUNTIME_CONFIG_KEY, {})
        self.scaling_config = runtime_config.get("scaling", {})
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

    def _get_autoscaling_instructions(self, all_node_metrics):
        return None

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
            total_cpus = cpu_counts[0]

            load_avg = metrics.get("load_avg")
            load_avg_per_cpu = load_avg[1]
            load_avg_per_cpu_1 = load_avg_per_cpu[0]
            load_avg_all_1 = load_avg[0][0]
            used_cpus = min(math.ceil(load_avg_all_1), total_cpus)

            memory = metrics.get("mem")
            (total_memory, available_memory, percent_memory, used_memory) = memory

            total_resources = {
                constants.CLOUDTIK_RESOURCE_CPU: total_cpus,
                constants.CLOUDTIK_RESOURCE_MEMORY: total_memory
            }
            free_resources = {
                constants.CLOUDTIK_RESOURCE_CPU: max(0, total_cpus - used_cpus),
                constants.CLOUDTIK_RESOURCE_MEMORY: available_memory
            }

            resource_load = {
                "load": {
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

            if logger.isEnabledFor(logging.DEBUG):
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
            if node_metrics["node_type"] == tags.NODE_KIND_HEAD:
                continue

            all_node_metrics.append(node_metrics)
        return all_node_metrics


class ScalingWithLoad(ScalingWithResources):
    def __init__(self,
                 config: Dict[str, Any],
                 head_ip: str) -> None:
        ScalingWithResources.__init__(self, config, head_ip)

        self.last_resource_demands_time = 0
        self.last_resource_state_snapshot = None

        # scaling parameters
        self.scaling_step = SCALING_WITH_LOAD_STEP_DEFAULT
        self.scaling_resource = SCALING_WITH_LOAD_RESOURCE_CPU
        self.cpu_load_threshold = SCALING_WITH_LOAD_CPU_LOAD_THRESHOLD_DEFAULT
        self.memory_load_threshold = SCALING_WITH_LOAD_MEMORY_LOAD_THRESHOLD_DEFAULT
        self._reset_load_config()

    def name(self):
        return "scaling-with-load"

    def reset(self, config):
        super().reset(config)
        self._reset_load_config()

    def _reset_load_config(self):
        self.scaling_step = self.scaling_config.get(
            "scaling_step", SCALING_WITH_LOAD_STEP_DEFAULT)
        self.scaling_resource = self.scaling_config.get(
            "scaling_resource", SCALING_WITH_LOAD_RESOURCE_CPU)
        self.cpu_load_threshold = self.scaling_config.get(
            "cpu_load_threshold", SCALING_WITH_LOAD_CPU_LOAD_THRESHOLD_DEFAULT)
        self.memory_load_threshold = self.scaling_config.get(
            "memory_load_threshold", SCALING_WITH_LOAD_MEMORY_LOAD_THRESHOLD_DEFAULT)

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

        autoscaling_instructions["scaling_time"] = self.last_state_time
        autoscaling_instructions["resource_demands"] = resource_demands
        if len(resource_demands) > 0 and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Resource demands: {}".format(resource_demands))

        return autoscaling_instructions

    def _get_cluster_metrics(self, all_node_metrics):
        cluster_total_cpus = 0
        cluster_used_cpus = 0
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
            load_avg_all = load_avg[0]
            load_avg_all_1 = load_avg_all[0]

            memory = metrics.get("mem")
            (total_memory, available_memory, percent_memory, used_memory) = memory

            cluster_total_cpus += total_cpus
            cluster_used_cpus += min(math.ceil(load_avg_all_1), total_cpus)
            cluster_load_avg_all_1 += load_avg_all_1
            cluster_total_memory += total_memory
            cluster_used_memory += used_memory

        cluster_cpu_load_1 = 0.0
        if cluster_total_cpus > 0:
            cluster_cpu_load_1 = round(cluster_load_avg_all_1 / cluster_total_cpus, 2)

        cluster_memory_load = 0.0
        if cluster_total_memory > 0:
            cluster_memory_load = round(cluster_used_memory / cluster_total_memory, 2)
        return {
            "total_cpus": cluster_total_cpus,
            "used_cpus": cluster_used_cpus,
            "available_cpus": max(0, cluster_total_cpus - cluster_used_cpus),
            "cpu_load": cluster_cpu_load_1,
            "total_memory": cluster_total_memory,
            "used_memory": cluster_used_memory,
            "available_memory": max(0, cluster_total_memory - cluster_used_memory),
            "memory_load": cluster_memory_load,
        }


class ScalingWithTime(ScalingWithResources):
    def __init__(self,
                 config: Dict[str, Any],
                 head_ip: str) -> None:
        ScalingWithResources.__init__(self, config, head_ip)

        # scaling parameters
        self.min_workers = 0
        self.scaling_periodic = SCALING_WITH_TIME_PERIODIC_DAILY
        self.scaling_math_base = SCALING_WITH_TIME_MATH_ON_MIN_WORKERS
        self.scaling_time_table = []
        self._reset_time_config()

    def name(self):
        return "scaling-with-time"

    def reset(self, config):
        super().reset(config)
        self._reset_time_config()

    def _reset_time_config(self):
        self.min_workers = self._get_min_workers()
        self.scaling_periodic = self.scaling_config.get(
            "scaling_periodic", SCALING_WITH_TIME_PERIODIC_DAILY)
        self.scaling_math_base = self.scaling_config.get(
            "scaling_math_base", SCALING_WITH_TIME_MATH_ON_MIN_WORKERS)
        self.scaling_time_table = self._formalize_time_table(
            self.scaling_config.get("scaling_time_table", {}))

    def _get_resource_requests_for(self, number_of_nodes):
        requested_cores = self.get_number_of_cores_to_scale(number_of_nodes)
        return get_resource_requests_for_cpu(requested_cores, self.config)

    def get_number_of_cores_to_scale(self, nodes):
        return convert_nodes_to_cpus(self.config, nodes)

    def get_memory_to_scale(self, nodes):
        return convert_nodes_to_memory(self.config, nodes)

    def _get_min_workers(self):
        return _sum_min_workers(self.config)

    def _get_autoscaling_instructions(self, all_node_metrics):
        # Use the time table to make the decisions
        resource_requests = self._get_resource_requests_with_time()
        if resource_requests is None:
            return None

        autoscaling_instructions = {
            "scaling_time": self.last_state_time,
            "resource_requests": resource_requests}
        if resource_requests is not None and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Scaling time table: {}".format(self.scaling_time_table))
            logger.debug("Resource requests: {}".format(resource_requests))

        return autoscaling_instructions

    @staticmethod
    def _get_time_format(time_spec):
        c = time_spec.count(':')
        if c == 0:
            time_format = '%H'
        elif c == 1:
            time_format = '%H:%M'
        else:
            time_format = '%H:%M:%S'
        return time_format

    @staticmethod
    def _get_seconds_in_day(t):
        return t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec

    def _time_spec_to_seconds(self, time_spec):
        if self.scaling_periodic == SCALING_WITH_TIME_PERIODIC_MONTHLY:
            # convert 01 07:23:02
            time_format = self._get_time_format(time_spec)
            time_format = "%d " + time_format
            t = time.strptime(time_spec, time_format)
            return (t.tm_mday - 1) * 24 * 3600 + self._get_seconds_in_day(t)
        elif self.scaling_periodic == SCALING_WITH_TIME_PERIODIC_WEEKLY:
            # convert Mon 07:23:02
            time_format = self._get_time_format(time_spec)
            time_format = "%a " + time_format
            t = time.strptime(time_spec, time_format)
            return t.tm_wday * 24 * 3600 + self._get_seconds_in_day(t)
        else:
            # Daily
            # convert 07:23:02
            time_format = self._get_time_format(time_spec)
            t = time.strptime(time_spec, time_format)
            return self._get_seconds_in_day(t)

    @staticmethod
    def isfloat(nodes_spec):
        try:
            float(nodes_spec)
            return True
        except ValueError:
            return False

    def get_nodes_from_base(self, nodes_spec, base_nodes):
        if nodes_spec.startswith('*'):
            nodes_spec = nodes_spec[1:]
            if self.isfloat(nodes_spec):
                return round(base_nodes * float(nodes_spec))
        elif nodes_spec.startswith('+'):
            nodes_spec = nodes_spec[1:]
            if nodes_spec.isdigit():
                return max(0, round(base_nodes + int(nodes_spec)))
        elif nodes_spec.startswith('-'):
            nodes_spec = nodes_spec[1:]
            if nodes_spec.isdigit():
                return max(0, round(base_nodes - int(nodes_spec)))
        raise ValueError("Invalid node specification for multiplier: {}".format(nodes_spec))

    def _expand_time_table(self, expanding_time_table):
        remaining = 0
        prev_expanding_slot = expanding_time_table[-1] if expanding_time_table else None
        for expanding_slot in expanding_time_table:
            if expanding_slot[1] < 0:
                # not expanded
                nodes_spec = expanding_slot[2]
                if isinstance(nodes_spec, int) or (
                        isinstance(nodes_spec, str) and nodes_spec.isdigit()):
                    nodes = int(nodes_spec)
                    if nodes == 0:
                        # Set nodes to minimal workers
                        nodes = self.min_workers
                    expanding_slot[1] = nodes
                elif isinstance(nodes_spec, str):
                    # if it is not digit, multiplier or addition or reduction
                    # we need expand base on its nodes of previous time slot
                    if self.scaling_math_base == SCALING_WITH_TIME_MATH_ON_MIN_WORKERS:
                        nodes = self.get_nodes_from_base(nodes_spec, self.min_workers)
                        expanding_slot[1] = nodes
                    else:
                        if prev_expanding_slot is not None and prev_expanding_slot[1] >= 0:
                            nodes = self.get_nodes_from_base(nodes_spec, prev_expanding_slot[1])
                            expanding_slot[1] = nodes
                        else:
                            # cannot expand by now
                            remaining += 1
                else:
                    raise ValueError("Invalid node specification: {}".format(nodes_spec))
            prev_expanding_slot = expanding_slot
        return remaining

    def _formalize_time_table(self, time_table):
        try:
            scaling_time_table = []
            for time_spec, nodes_spec in time_table.items():
                seconds = self._time_spec_to_seconds(time_spec)
                if nodes_spec is None:
                    continue
                expanding_time_slot = [seconds, -1, nodes_spec]
                scaling_time_table.append(expanding_time_slot)

            def seconds_sort(time_slot):
                return time_slot[0]
            scaling_time_table.sort(key=seconds_sort)

            remaining = self._expand_time_table(scaling_time_table)
            if remaining > 0:
                remaining = self._expand_time_table(scaling_time_table)
                if remaining > 0:
                    raise ValueError("Invalid node specification. For math based on previous time, "
                                     "at least one time needs specific node number. "
                                     "Use 0 to refer the minimum workers from cluster configuration.")
            return scaling_time_table
        except Exception as e:
            logger.error(
                "Failed to parse the scaling time table: {}".format(str(e)))
            return []

    def _get_seconds_in_period(self, t):
        local_time = time.localtime(t)
        seconds_in_period = 0
        if self.scaling_periodic == SCALING_WITH_TIME_PERIODIC_MONTHLY:
            # Make the first day is start from 0
            seconds_in_period += (local_time.tm_mday - 1) * 24 * 3600
        elif self.scaling_periodic == SCALING_WITH_TIME_PERIODIC_WEEKLY:
            seconds_in_period += local_time.tm_wday * 24 * 3600
        seconds_in_period += self._get_seconds_in_day(local_time)
        return seconds_in_period

    def _get_nodes_request(self, seconds_in_period):
        if not self.scaling_time_table:
            return None

        prev_time_slot = self.scaling_time_table[-1]
        for time_slot in self.scaling_time_table:
            if time_slot[0] <= seconds_in_period:
                prev_time_slot = time_slot
                continue

            # This is the first timeslot greater than seconds_in_period
            return prev_time_slot[1]

        # if there is no time slot found, the time should circle to last timeslot
        return prev_time_slot[1]

    def _get_resource_requests_with_time(self):
        current_time = self.last_state_time
        seconds_in_period = self._get_seconds_in_period(current_time)
        return self._get_resource_requests_at_seconds(seconds_in_period)

    def _get_resource_requests_at_seconds(self, seconds_in_period):
        number_of_nodes = self._get_nodes_request(seconds_in_period)
        if number_of_nodes is None:
            return None
        return self._get_resource_requests_for(number_of_nodes)


def _create_scaling_policy(scaling_policy_name: str, config, head_ip):
    if SCALING_WITH_LOAD == scaling_policy_name:
        return ScalingWithLoad(config, head_ip)
    elif SCALING_WITH_TIME == scaling_policy_name:
        return ScalingWithTime(config, head_ip)

    return None
