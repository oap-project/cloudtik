from collections import Counter
from dataclasses import dataclass
from functools import reduce
import logging
from numbers import Number
import time
from typing import Dict, List, Tuple, Any

import numpy as np

from cloudtik.core._private.constants import CLOUDTIK_MEMORY_RESOURCE_UNIT_BYTES,\
    CLOUDTIK_MAX_RESOURCE_DEMAND_VECTOR_SIZE
from cloudtik.core._private.cluster.resource_demand_scheduler import \
    NodeIP, ResourceDict

logger = logging.getLogger(__name__)

# A Dict and the count of how many times it occurred.
# Refer to freq_of_dicts() below.
DictCount = Tuple[Dict, Number]


@dataclass
class ClusterMetricsSummary:
    # Map of resource name (e.g. "memory") to pair of (Used, Available) numbers
    usage: Dict[str, Tuple[Number, Number]]
    # Counts of demand bundles from task/actor demand.
    # e.g. [({"CPU": 1}, 5), ({"GPU":1}, 2)]
    resource_demand: List[DictCount]
    # Counts of demand bundles requested by cloudtik.core.api.request_resources
    request_demand: List[DictCount]
    node_types: List[DictCount]


def add_resources(dict1: Dict[str, float],
                  dict2: Dict[str, float]) -> Dict[str, float]:
    """Add the values in two dictionaries.

    Returns:
        dict: A new dictionary (inputs remain unmodified).
    """
    new_dict = dict1.copy()
    for k, v in dict2.items():
        new_dict[k] = v + new_dict.get(k, 0)
    return new_dict


def freq_of_dicts(dicts: List[Dict],
                  serializer=lambda d: frozenset(d.items()),
                  deserializer=dict) -> List[DictCount]:
    """Count a list of dictionaries (or unhashable types).

    This is somewhat annoying because mutable data structures aren't hashable,
    and set/dict keys must be hashable.

    Args:
        dicts (List[D]): A list of dictionaries to be counted.
        serializer (D -> S): A custom serialization function. The output type S
            must be hashable. The default serializer converts a dictionary into
            a frozenset of KV pairs.
        deserializer (S -> U): A custom deserialization function. See the
            serializer for information about type S. For dictionaries U := D.

    Returns:
        List[Tuple[U, int]]: Returns a list of tuples. Each entry in the list
            is a tuple containing a unique entry from `dicts` and its
            corresponding frequency count.
    """
    freqs = Counter(map(lambda d: serializer(d), dicts))
    as_list = []
    for as_set, count in freqs.items():
        as_list.append((deserializer(as_set), count))
    return as_list


class ClusterMetrics:
    """Container for cluster load metrics.

    Metrics here are updated from heartbeats. The scaler
    queries these metrics to determine when to scale up, and which nodes
    can be removed.
    """

    def __init__(self):
        self.node_id_by_ip = {}

        # Heartbeat metrics
        self.last_heartbeat_time_by_ip = {}

        # Resources metrics
        self.last_used_time_by_ip = {}
        self.last_resource_time_by_ip = {}
        self.static_resources_by_ip = {}
        self.dynamic_resources_by_ip = {}
        self.resource_load_by_ip = {}

        # Resource requests (on demand or autoscale)
        self.last_demanding_time = 0
        self.autoscaling_instructions = {}
        self.resource_demands = []
        self.resource_requests = []

    def update_heartbeat(self,
                         ip: str,
                         node_id: str,
                         last_heartbeat_time):
        self.node_id_by_ip[ip] = node_id
        self.last_heartbeat_time_by_ip[ip] = last_heartbeat_time

    def update_autoscaling_instructions(self,
                                        autoscaling_instructions: Dict[str, Any]):
        self.autoscaling_instructions = autoscaling_instructions

        # resource_demands is a List[Dict[str, float]]
        resource_demands = []
        if autoscaling_instructions is not None:
            demanding_time = autoscaling_instructions.get("demanding_time")
            _resource_demands = autoscaling_instructions.get("resource_demands")

            # Only the new demanding will be updated
            if demanding_time > self.last_demanding_time and _resource_demands:
                resource_demands = _resource_demands
                self.last_demanding_time = demanding_time

        self.resource_demands = resource_demands

    def update_node_resources(self,
                              ip: str,
                              node_id: str,
                              last_resource_time,
                              static_resources: Dict[str, Any],
                              dynamic_resources: Dict[str, Any],
                              resource_load: Dict[str, Any]):
        self.node_id_by_ip[ip] = node_id
        self.static_resources_by_ip[ip] = static_resources
        self.resource_load_by_ip[ip] = resource_load

        # We are not guaranteed to have a corresponding dynamic resource
        # for every static resource because dynamic resources are based on
        # the available resources in the heartbeat, which does not exist
        # if it is zero. Thus, we have to update dynamic resources here.
        dynamic_resources_update = dynamic_resources.copy()
        for resource_name, capacity in self.static_resources_by_ip[ip].items():
            if resource_name not in dynamic_resources_update:
                dynamic_resources_update[resource_name] = 0.0
        self.dynamic_resources_by_ip[ip] = dynamic_resources_update

        # Every time we update the resource state,
        # If a node is not idle, we will update its last used time
        # If a node is idle, it's last used time will not be updated and keep in the previous used time
        # This last used time can be used to check how long a node is in an idle state
        if (ip not in self.last_used_time_by_ip
                or ("in_use" in resource_load and resource_load["in_use"])
                or not self._is_node_idle(ip)):
            self.last_used_time_by_ip[ip] = last_resource_time

        self.last_resource_time_by_ip[ip] = last_resource_time

    def _is_node_idle(self, ip):
        # TODO: We may need some tolerance when making such comparisons
        # before enable this check
        # if self.static_resources_by_ip[ip] != self.dynamic_resources_by_ip[ip]:
        #    return False
        return True

    def mark_active(self, ip, last_heartbeat_time=None):
        assert ip is not None, "IP should be known at this time"
        logger.debug("Node {} is newly setup, treating as active".format(ip))
        if not last_heartbeat_time:
            last_heartbeat_time = time.time()
        self.last_heartbeat_time_by_ip[ip] = last_heartbeat_time

    def is_active(self, ip):
        return ip in self.last_heartbeat_time_by_ip

    def prune_active_ips(self, active_ips: List[str]):
        """The ips stored by LoadMetrics are obtained by polling
        the redis in ClusterController.update_cluster_metrics().

        On the other hand, the scaler gets a list of node ips from
        its NodeProvider.

        This method removes from LoadMetrics the ips unknown to the scaler.

        Args:
            active_ips (List[str]): The node ips known to the cluster controller.
        """
        active_ips = set(active_ips)

        def prune(mapping, should_log):
            unwanted_ips = set(mapping) - active_ips
            for unwanted_ip in unwanted_ips:
                if should_log:
                    logger.info("Cluster Metrics: " f"Removed ip: {unwanted_ip}.")
                del mapping[unwanted_ip]
            if unwanted_ips and should_log:
                logger.info(
                    "Cluster Metrics: "
                    "Removed {} stale ip mappings: {} not in {}".format(
                        len(unwanted_ips), unwanted_ips, active_ips))
            assert not (unwanted_ips & set(mapping))

        prune(self.last_used_time_by_ip, should_log=True)
        prune(self.static_resources_by_ip, should_log=False)
        prune(self.node_id_by_ip, should_log=False)
        prune(self.dynamic_resources_by_ip, should_log=False)
        prune(self.resource_load_by_ip, should_log=False)
        prune(self.last_heartbeat_time_by_ip, should_log=False)
        prune(self.last_resource_time_by_ip, should_log=False)

    def get_node_resources(self):
        """Return a list of node resources (static resource sizes).

        Example:
            >>> metrics.get_node_resources()
            [{"CPU": 1}, {"CPU": 4, "GPU": 8}]  # for two different nodes
        """
        return self.static_resources_by_ip.values()

    def get_static_node_resources_by_ip(self) -> Dict[NodeIP, ResourceDict]:
        """Return a dict of node resources for every node ip.

        Example:
            >>> lm.get_static_node_resources_by_ip()
            {127.0.0.1: {"CPU": 1}, 127.0.0.2: {"CPU": 4, "GPU": 8}}
        """
        return self.static_resources_by_ip

    def get_resource_utilization(self):
        return self.dynamic_resources_by_ip

    def _get_resource_usage(self):
        num_nodes = 0
        num_non_idle = 0
        resources_used = {}
        resources_total = {}
        for ip, max_resources in self.static_resources_by_ip.items():
            # Nodes without resources don't count as nodes (e.g. unmanaged
            # nodes)
            if any(max_resources.values()):
                num_nodes += 1
            avail_resources = self.dynamic_resources_by_ip[ip]
            resource_load = self.resource_load_by_ip[ip]
            max_frac = 0.0
            for resource_id, amount in resource_load.get("utilization", {}).items():
                if amount > 0:
                    max_frac = 1.0  # the resource is saturated
            for resource_id, amount in max_resources.items():
                used = amount - avail_resources[resource_id]
                if resource_id not in resources_used:
                    resources_used[resource_id] = 0.0
                    resources_total[resource_id] = 0.0
                resources_used[resource_id] += used
                resources_total[resource_id] += amount
                used = max(0, used)
                if amount > 0:
                    frac = used / float(amount)
                    if frac > max_frac:
                        max_frac = frac
            if max_frac > 0:
                num_non_idle += 1

        return resources_used, resources_total

    def get_resource_demands(self, clip=True):
        if clip:
            # Bound the total number of bundles to
            # CLOUDTIK_MAX_RESOURCE_DEMAND_VECTOR_SIZE. This guarantees the resource
            # demand scheduler bin packing algorithm takes a reasonable amount
            # of time to run.
            return (
                self.resource_demands[:CLOUDTIK_MAX_RESOURCE_DEMAND_VECTOR_SIZE]
            )
        else:
            return self.resource_demands

    def get_resource_requests(self):
        return self.resource_requests

    def resources_avail_summary(self) -> str:
        """Return a concise string of cluster size to report to event logs.

        For example, "3 CPUs, 4 GPUs".
        """
        total_resources = reduce(add_resources,
                                 self.static_resources_by_ip.values()
                                 ) if self.static_resources_by_ip else {}
        out = "{} CPUs".format(int(total_resources.get("CPU", 0)))
        if "GPU" in total_resources:
            out += ", {} GPUs".format(int(total_resources["GPU"]))
        return out

    def summary(self):
        available_resources = reduce(add_resources,
                                     self.dynamic_resources_by_ip.values()
                                     ) if self.dynamic_resources_by_ip else {}
        total_resources = reduce(add_resources,
                                 self.static_resources_by_ip.values()
                                 ) if self.static_resources_by_ip else {}
        usage_dict = {}
        for key in total_resources:
            if key in ["memory"]:
                total = total_resources[key] * \
                    CLOUDTIK_MEMORY_RESOURCE_UNIT_BYTES
                available = available_resources[key] * \
                    CLOUDTIK_MEMORY_RESOURCE_UNIT_BYTES
                usage_dict[key] = (total - available, total)
            else:
                total = total_resources[key]
                usage_dict[key] = (total - available_resources[key], total)

        summarized_resource_demands = freq_of_dicts(
            self.get_resource_demands(clip=False))
        summarized_resource_requests = freq_of_dicts(
            self.get_resource_requests())

        nodes_summary = freq_of_dicts(self.static_resources_by_ip.values())

        return ClusterMetricsSummary(
            usage=usage_dict,
            resource_demand=summarized_resource_demands,
            request_demand=summarized_resource_requests,
            node_types=nodes_summary)

    def set_resource_requests(self, requested_resources):
        if requested_resources is not None:
            assert isinstance(requested_resources, list), requested_resources
        self.resource_requests = [
            request for request in requested_resources if len(request) > 0
        ]

    def info_string(self):
        return " - " + "\n - ".join(
            ["{}: {}".format(k, v) for k, v in sorted(self._info().items())])

    def _info(self):
        resources_used, resources_total = self._get_resource_usage()

        now = time.time()
        idle_times = [now - t for t in self.last_used_time_by_ip.values()]
        heartbeat_times = [
            now - t for t in self.last_heartbeat_time_by_ip.values()
        ]
        most_delayed_heartbeats = sorted(
            self.last_heartbeat_time_by_ip.items(),
            key=lambda pair: pair[1])[:5]
        most_delayed_heartbeats = {
            ip: (now - t)
            for ip, t in most_delayed_heartbeats
        }

        def format_resource(key, value):
            if key in ["memory"]:
                return "{} GiB".format(
                    round(
                        value * CLOUDTIK_MEMORY_RESOURCE_UNIT_BYTES /
                        (1024 * 1024 * 1024), 2))
            else:
                return round(value, 2)

        return {
            "ResourceUsage": ", ".join([
                "{}/{} {}".format(
                    format_resource(rid, resources_used[rid]),
                    format_resource(rid, resources_total[rid]), rid)
                for rid in sorted(resources_used)
                if not rid.startswith("node:")
            ]),
            "NodeIdleSeconds": "Min={} Mean={} Max={}".format(
                int(np.min(idle_times)) if idle_times else -1,
                int(np.mean(idle_times)) if idle_times else -1,
                int(np.max(idle_times)) if idle_times else -1),
            "TimeSinceLastHeartbeat": "Min={} Mean={} Max={}".format(
                int(np.min(heartbeat_times)) if heartbeat_times else -1,
                int(np.mean(heartbeat_times)) if heartbeat_times else -1,
                int(np.max(heartbeat_times)) if heartbeat_times else -1),
            "MostDelayedHeartbeats": most_delayed_heartbeats,
        }
