"""Implements multi-node-type cluster scaling.

This file implements a scaling algorithm that is aware of multiple node
types. The scaler will pass in
a vector of resource shape demands, and the resource demand scheduler will
return a list of node types that can satisfy the demands given constraints
(i.e., reverse bin packing).
"""

import copy
import logging
import collections
from numbers import Real
from typing import Dict, Any
from typing import List
from typing import Optional
from typing import Tuple

from cloudtik.core.node_provider import NodeProvider
from cloudtik.core._private.constants import CLOUDTIK_CONSERVE_GPU_NODES, to_memory_units
from cloudtik.core.tags import (
    CLOUDTIK_TAG_USER_NODE_TYPE, NODE_KIND_UNMANAGED,
    NODE_KIND_WORKER, CLOUDTIK_TAG_NODE_KIND, NODE_KIND_HEAD)

logger = logging.getLogger(__name__)

# The minimum number of nodes to launch concurrently.
UPSCALING_INITIAL_NUM_NODES = 5

# e.g., cpu_4_ondemand.
NodeType = str

# e.g., {"resources": ..., "max_workers": ...}.
NodeTypeConfigDict = Dict[str, Any]

# e.g., {"GPU": 1}.
ResourceDict = Dict[str, Real]

# e.g., "node-1".
NodeID = str

# e.g., "127.0.0.1".
NodeIP = str


class ResourceDemandScheduler:
    def __init__(self,
                 provider: NodeProvider,
                 node_types: Dict[NodeType, NodeTypeConfigDict],
                 max_workers: int,
                 head_node_type: NodeType,
                 upscaling_speed: float = 1) -> None:
        self.provider = provider
        self.node_types = _convert_memory_unit(node_types)
        self.node_resource_updated = set()
        self.max_workers = max_workers
        self.head_node_type = head_node_type
        self.upscaling_speed = upscaling_speed

    def _get_head_and_workers(
            self, nodes: List[NodeID]) -> Tuple[NodeID, List[NodeID]]:
        """Returns the head node's id and the list of all worker node ids,
        given a list `nodes` of all node ids in the cluster.
        """
        head_id, worker_ids = None, []
        for node in nodes:
            tags = self.provider.node_tags(node)
            if tags[CLOUDTIK_TAG_NODE_KIND] == NODE_KIND_HEAD:
                head_id = node
            elif tags[CLOUDTIK_TAG_NODE_KIND] == NODE_KIND_WORKER:
                worker_ids.append(node)
        return head_id, worker_ids

    def reset_config(self,
                     provider: NodeProvider,
                     node_types: Dict[NodeType, NodeTypeConfigDict],
                     max_workers: int,
                     head_node_type: NodeType,
                     upscaling_speed: float = 1) -> None:
        """Updates the class state variables.
        """
        new_node_types = copy.deepcopy(node_types)
        final_node_types = _convert_memory_unit(new_node_types)

        self.provider = provider
        self.node_types = copy.deepcopy(final_node_types)
        self.node_resource_updated = set()
        self.max_workers = max_workers
        self.head_node_type = head_node_type
        self.upscaling_speed = upscaling_speed

    def is_feasible(self, bundle: ResourceDict) -> bool:
        for node_type, config in self.node_types.items():
            max_of_type = config.get("max_workers", 0)
            node_resources = config["resources"]
            if (node_type == self.head_node_type or max_of_type > 0) and _fits(
                    node_resources, bundle):
                return True
        return False

    def get_nodes_to_launch(
            self,
            nodes: List[NodeID],
            launching_nodes: Dict[NodeType, int],
            resource_demands: List[ResourceDict],
            unused_resources_by_ip: Dict[NodeIP, ResourceDict],
            max_resources_by_ip: Dict[NodeIP, ResourceDict],
            ensure_min_cluster_size: List[ResourceDict] = None,
    ) -> (Dict[NodeType, int], List[ResourceDict]):
        """Given resource demands, return node types to add to the cluster.

        This method:
            (1) calculates the resources present in the cluster.
            (2) calculates the remaining nodes to add to respect min_workers
                constraint per node type.
            (3) calculates the unfulfilled resource bundles.
            (4) calculates which nodes need to be launched to fulfill all
                the bundle requests, subject to max_worker constraints.

        Args:
            nodes: List of existing nodes in the cluster.
            launching_nodes: Summary of node types currently being launched.
            resource_demands: Vector of resource demands from the scheduler.
            unused_resources_by_ip: Mapping from ip to available resources.
            max_resources_by_ip: Mapping from ip to static node resources.
            ensure_min_cluster_size: Try to ensure the cluster can fit at least
                this set of resources. This differs from resources_demands in
                that we don't take into account existing usage.

        Returns:
            Dict of count to add for each node type, and residual of resources
            that still cannot be fulfilled.
        """
        # Note: currently, we don't update the total resources from runtime
        # But we use the node types static memory information here
        # self._update_node_resources_from_runtime(nodes, max_resources_by_ip)

        node_resources: List[ResourceDict]
        node_type_counts: Dict[NodeType, int]
        node_resources, node_type_counts = self.calculate_node_resources(
            nodes, launching_nodes, unused_resources_by_ip)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Cluster resources: {}".format(node_resources))
            logger.debug("Node counts: {}".format(node_type_counts))
            logger.debug("Minimum cluster size: {}".format(ensure_min_cluster_size))

        # Step 2: add nodes to add to satisfy min_workers for each type
        (node_resources,
         node_type_counts,
         adjusted_min_workers) = \
            _add_min_workers_nodes(
                node_resources, node_type_counts, self.node_types,
                self.max_workers, self.head_node_type, ensure_min_cluster_size)

        # Add 1 to account for the head node.
        max_to_add = self.max_workers + 1 - sum(node_type_counts.values())

        # Step 3/4: add nodes for pending tasks
        unfulfilled, _ = get_bin_pack_residual(node_resources,
                                               resource_demands)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Resource demands: {}".format(resource_demands))
            logger.debug("Unfulfilled demands: {}".format(unfulfilled))

        nodes_to_add_based_on_demand, final_unfulfilled = get_nodes_for(
            self.node_types, node_type_counts, self.head_node_type, max_to_add,
            unfulfilled)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Final unfulfilled: {}".format(final_unfulfilled))

        # Merge nodes to add based on demand and nodes to add based on
        # min_workers constraint. We add them because nodes to add based on
        # demand was calculated after the min_workers constraint was respected.
        total_nodes_to_add = {}

        for node_type in self.node_types:
            nodes_to_add = (adjusted_min_workers.get(
                node_type, 0) + nodes_to_add_based_on_demand.get(node_type, 0))
            if nodes_to_add > 0:
                total_nodes_to_add[node_type] = nodes_to_add

        # Limit the number of concurrent launches
        total_nodes_to_add = self._get_concurrent_resource_demand_to_launch(
            total_nodes_to_add, unused_resources_by_ip.keys(), nodes,
            launching_nodes, adjusted_min_workers)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Node requests: {}".format(total_nodes_to_add))
        return total_nodes_to_add, final_unfulfilled

    def _update_node_resources_from_runtime(
            self, nodes: List[NodeID],
            max_resources_by_ip: Dict[NodeIP, ResourceDict]):
        """Update static node type resources with runtime resources

        This will update the cached static node type resources with the runtime
        resources. Because we can not know the correct memory from config file.
        """
        need_update = len(self.node_types) != len(self.node_resource_updated)

        if not need_update:
            return
        for node_id in nodes:
            tags = self.provider.node_tags(node_id)

            if CLOUDTIK_TAG_USER_NODE_TYPE not in tags:
                continue

            node_type = tags[CLOUDTIK_TAG_USER_NODE_TYPE]
            if (node_type in self.node_resource_updated
                    or node_type not in self.node_types):
                # continue if the node type has been updated or is not an known
                # node type
                continue
            ip = self.provider.internal_ip(node_id)
            runtime_resources = max_resources_by_ip.get(ip)
            if runtime_resources:
                runtime_resources = copy.deepcopy(runtime_resources)
                resources = self.node_types[node_type].get("resources", {})
                for key in ["CPU", "GPU", "memory"]:
                    if key in runtime_resources:
                        resources[key] = runtime_resources[key]
                self.node_types[node_type]["resources"] = resources

                node_kind = tags[CLOUDTIK_TAG_NODE_KIND]
                if node_kind == NODE_KIND_WORKER:
                    # Here, we do not record the resources have been updated
                    # if it is the head node kind. Because it need be updated
                    # by worker kind runtime resource. The most difference
                    # between head and worker is the memory resources. The head
                    # node needs to configure redis memory which is not needed
                    # for worker nodes.
                    self.node_resource_updated.add(node_type)

    def _get_concurrent_resource_demand_to_launch(
            self,
            to_launch: Dict[NodeType, int],
            connected_nodes: List[NodeIP],
            non_terminated_nodes: List[NodeID],
            pending_launches_nodes: Dict[NodeType, int],
            adjusted_min_workers: Dict[NodeType, int],
    ) -> Dict[NodeType, int]:
        """Updates the max concurrent resources to launch for each node type.

        Given the current nodes that should be launched, the non terminated
        nodes (running and pending) and the pending to be launched nodes. This
        method calculates the maximum number of nodes to launch concurrently
        for each node type as follows:
            1) Calculates the running nodes.
            2) Calculates the pending nodes and gets the launching nodes.
            3) Limits the total number of pending + currently-launching +
               to-be-launched nodes to:
               max(5, self.upscaling_speed * max(running_nodes[node_type], 1)).

        Args:
            to_launch: List of number of nodes to launch based on resource
                demand for every node type.
            connected_nodes: Running nodes (from LoadMetrics).
            non_terminated_nodes: Non terminated nodes (pending/running).
            pending_launches_nodes: Nodes that are in the launch queue.
            adjusted_min_workers: Nodes to launch to satisfy
                min_workers and request_resources(). This overrides the launch
                limits since the user is hinting to immediately scale up to
                this size.
        Returns:
            Dict[NodeType, int]: Maximum number of nodes to launch for each
                node type.
        """
        updated_nodes_to_launch = {}
        running_nodes, pending_nodes = \
            self._separate_running_and_pending_nodes(
                non_terminated_nodes, connected_nodes,
            )
        for node_type in to_launch:
            # Enforce here max allowed pending nodes to be frac of total
            # running nodes.
            max_allowed_pending_nodes = max(
                UPSCALING_INITIAL_NUM_NODES,
                int(self.upscaling_speed * max(running_nodes[node_type], 1)))
            total_pending_nodes = pending_launches_nodes.get(
                node_type, 0) + pending_nodes[node_type]

            upper_bound = max(
                max_allowed_pending_nodes - total_pending_nodes,

                # Allow more nodes if this is to respect min_workers or
                # request_resources().
                adjusted_min_workers.get(node_type, 0))

            if upper_bound > 0:
                updated_nodes_to_launch[node_type] = min(
                    upper_bound, to_launch[node_type])

        return updated_nodes_to_launch

    def _separate_running_and_pending_nodes(
            self,
            non_terminated_nodes: List[NodeID],
            connected_nodes: List[NodeIP],
    ) -> (Dict[NodeType, int], Dict[NodeType, int]):
        """Splits connected and non terminated nodes to pending & running."""

        running_nodes = collections.defaultdict(int)
        pending_nodes = collections.defaultdict(int)
        for node_id in non_terminated_nodes:
            tags = self.provider.node_tags(node_id)
            if CLOUDTIK_TAG_USER_NODE_TYPE in tags:
                node_type = tags[CLOUDTIK_TAG_USER_NODE_TYPE]
                node_ip = self.provider.internal_ip(node_id)
                if node_ip in connected_nodes:
                    running_nodes[node_type] += 1
                else:
                    pending_nodes[node_type] += 1
        return running_nodes, pending_nodes

    def calculate_node_resources(
            self, nodes: List[NodeID], pending_nodes: Dict[NodeID, int],
            unused_resources_by_ip: Dict[str, ResourceDict]
    ) -> (List[ResourceDict], Dict[NodeType, int]):
        """Returns node resource list and node type counts.

           Counts the running nodes, pending nodes.
           Args:
                nodes: Existing nodes.
                pending_nodes: Pending nodes.
           Returns:
                node_resources: a list of running + pending resources.
                    E.g., [{"CPU": 4}, {"GPU": 2}].
                node_type_counts: running + pending workers per node type.
        """

        node_resources = []
        node_type_counts = collections.defaultdict(int)

        def add_node(node_type, available_resources=None):
            if node_type not in self.node_types:
                # We should not get here, but if for some reason we do, log an
                # error and skip the errant node_type.
                logger.error(
                    f"Missing entry for node_type {node_type} in "
                    f"cluster config: {self.node_types} under entry "
                    "available_node_types. This node's resources will be "
                    "ignored. If you are using an unmanaged node, manually "
                    f"set the {CLOUDTIK_TAG_NODE_KIND} tag to "
                    f"\"{NODE_KIND_UNMANAGED}\" in your cloud provider's "
                    "management console.")
                return None
            # Careful not to include the same dict object multiple times.
            available = copy.deepcopy(self.node_types[node_type]["resources"])
            # If available_resources is None this might be because the node is
            # no longer pending, but the node hasn't sent a heartbeat to redis
            # yet.
            if bool(available_resources):
                available = copy.deepcopy(available_resources)

            node_resources.append(available)
            node_type_counts[node_type] += 1

        for node_id in nodes:
            tags = self.provider.node_tags(node_id)
            if CLOUDTIK_TAG_USER_NODE_TYPE in tags:
                node_type = tags[CLOUDTIK_TAG_USER_NODE_TYPE]
                # NOTE: Special handling for head node -> consider head node resources are all used!
                if tags[CLOUDTIK_TAG_NODE_KIND] == NODE_KIND_HEAD:
                    # Head node: consider all the head resources are used, because we cannot use it for workers
                    if node_type in self.node_types:
                        available_resources = copy.deepcopy(self.node_types[node_type]["resources"])
                        for resource_id in available_resources:
                            available_resources[resource_id] = 0
                    else:
                        available_resources = {"CPU": 0}
                    add_node(node_type, available_resources)
                elif tags[CLOUDTIK_TAG_NODE_KIND] == NODE_KIND_WORKER:
                    # Worker node
                    ip = self.provider.internal_ip(node_id)
                    available_resources = unused_resources_by_ip.get(ip)
                    add_node(node_type, available_resources)

        for node_type, count in pending_nodes.items():
            for _ in range(count):
                add_node(node_type)

        return node_resources, node_type_counts


def _convert_memory_unit(node_types: Dict[NodeType, NodeTypeConfigDict]
                         ) -> Dict[NodeType, NodeTypeConfigDict]:
    """Convert memory and object_store_memory to memory unit"""
    node_types = copy.deepcopy(node_types)
    for node_type in node_types:
        res = node_types[node_type].get("resources", {})
        if "memory" in res:
            size = float(res["memory"])
            res["memory"] = to_memory_units(size, False)
        if "object_store_memory" in res:
            size = float(res["object_store_memory"])
            res["object_store_memory"] = to_memory_units(
                size, False)
        if res:
            node_types[node_type]["resources"] = res
    return node_types


def _node_type_counts_to_node_resources(
        node_types: Dict[NodeType, NodeTypeConfigDict],
        node_type_counts: Dict[NodeType, int]) -> List[ResourceDict]:
    """Converts a node_type_counts dict into a list of node_resources."""
    resources = []
    for node_type, count in node_type_counts.items():
        # Be careful, each entry in the list must be deep copied!
        resources += [
            node_types[node_type]["resources"].copy() for _ in range(count)
        ]
    return resources


def _add_min_workers_nodes(
        node_resources: List[ResourceDict],
        node_type_counts: Dict[NodeType, int],
        node_types: Dict[NodeType, NodeTypeConfigDict], max_workers: int,
        head_node_type: NodeType, ensure_min_cluster_size: List[ResourceDict]
) -> (List[ResourceDict], Dict[NodeType, int], Dict[NodeType, int]):
    """Updates resource demands to respect the min_workers and
    request_resources() constraints.

    Args:
        node_resources: Resources of exisiting nodes already launched/pending.
        node_type_counts: Counts of existing nodes already launched/pending.
        node_types: Node types config.
        max_workers: global max_workers constaint.
        ensure_min_cluster_size: resource demands from request_resources().

    Returns:
        node_resources: The updated node resources after adding min_workers
            and request_resources() constraints per node type.
        node_type_counts: The updated node counts after adding min_workers
            and request_resources() constraints per node type.
        total_nodes_to_add_dict: The nodes to add to respect min_workers and
            request_resources() constraints.
    """
    total_nodes_to_add_dict = {}
    for node_type, config in node_types.items():
        existing = node_type_counts.get(node_type, 0)
        target = min(
            config.get("min_workers", 0), max_workers)
        if node_type == head_node_type:
            # Add 1 to account for head node.
            target = target + 1
        if existing < target:
            total_nodes_to_add_dict[node_type] = target - existing
            node_type_counts[node_type] = target
            node_resources.extend([
                copy.deepcopy(node_types[node_type]["resources"])
                for _ in range(total_nodes_to_add_dict[node_type])
            ])

    if ensure_min_cluster_size:
        max_to_add = max_workers + 1 - sum(node_type_counts.values())
        max_node_resources = []
        # Fit request_resources() on all the resources as if they are idle.
        for node_type in node_type_counts:
            max_node_resources.extend([
                copy.deepcopy(node_types[node_type]["resources"])
                for _ in range(node_type_counts[node_type])
            ])
        # Get the unfulfilled to ensure min cluster size.
        resource_requests_unfulfilled, _ = get_bin_pack_residual(
            max_node_resources, ensure_min_cluster_size)
        # Get the nodes to meet the unfulfilled.
        nodes_to_add_request_resources, _ = get_nodes_for(
            node_types, node_type_counts, head_node_type, max_to_add,
            resource_requests_unfulfilled)
        # Update the resources, counts and total nodes to add.
        for node_type in nodes_to_add_request_resources:
            nodes_to_add = nodes_to_add_request_resources.get(node_type, 0)
            if nodes_to_add > 0:
                node_type_counts[
                    node_type] = nodes_to_add + node_type_counts.get(
                        node_type, 0)
                node_resources.extend([
                    copy.deepcopy(node_types[node_type]["resources"])
                    for _ in range(nodes_to_add)
                ])
                total_nodes_to_add_dict[
                    node_type] = nodes_to_add + total_nodes_to_add_dict.get(
                        node_type, 0)
    return node_resources, node_type_counts, total_nodes_to_add_dict


def get_nodes_for(node_types: Dict[NodeType, NodeTypeConfigDict],
                  existing_nodes: Dict[NodeType, int],
                  head_node_type: NodeType,
                  max_to_add: int,
                  resources: List[ResourceDict],
                  strict_spread: bool = False
                  ) -> (Dict[NodeType, int], List[ResourceDict]):
    """Determine nodes to add given resource demands and constraints.

    Args:
        node_types: node types config.
        existing_nodes: counts of existing nodes already launched.
            This sets constraints on the number of new nodes to add.
        max_to_add: global constraint on nodes to add.
        resources: resource demands to fulfill.
        strict_spread: If true, each element in `resources` must be placed on a
            different node.

    Returns:
        Dict of count to add for each node type, and residual of resources
        that still cannot be fulfilled.
    """
    nodes_to_add = collections.defaultdict(int)

    while resources and sum(nodes_to_add.values()) < max_to_add:
        utilization_scores = []
        for node_type in node_types:
            max_workers_of_node_type = node_types[node_type].get(
                "max_workers", 0)
            if head_node_type == node_type:
                # Add 1 to account for head node.
                max_workers_of_node_type = max_workers_of_node_type + 1
            if (existing_nodes.get(node_type, 0) + nodes_to_add.get(
                    node_type, 0) >= max_workers_of_node_type):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Will not launch node of {node_type} type as it already "
                                 f"exceeds the max number ({max_workers_of_node_type})")
                continue
            node_resources = node_types[node_type]["resources"]
            if strict_spread:
                # If handling strict spread, only one bundle can be placed on
                # the node.
                score = _utilization_score(node_resources, [resources[0]])
            else:
                score = _utilization_score(node_resources, resources)
            if score is not None:
                utilization_scores.append((score, node_type))

        # Give up, no feasible node.
        if not utilization_scores:
            logger.warning(
                f"The scaler could not find a node type to satisfy the "
                f"request: {resources}. "
            )
            break

        utilization_scores = sorted(utilization_scores, reverse=True)
        best_node_type = utilization_scores[0][1]
        nodes_to_add[best_node_type] += 1
        if strict_spread:
            resources = resources[1:]
        else:
            allocated_resource = node_types[best_node_type]["resources"]
            residual, _ = get_bin_pack_residual([allocated_resource],
                                                resources)
            assert len(residual) < len(resources), (resources, residual)
            resources = residual

    return nodes_to_add, resources


def _utilization_score(node_resources: ResourceDict,
                       resources: List[ResourceDict]) -> Optional[bool, int, float, float]:
    remaining = copy.deepcopy(node_resources)
    fittable = []
    resource_types = set()
    for r in resources:
        for k, v in r.items():
            if v > 0:
                resource_types.add(k)
        if _fits(remaining, r):
            fittable.append(r)
            _inplace_subtract(remaining, r)
    if not fittable:
        return None

    util_by_resources = []
    num_matching_resource_types = 0
    for k, v in node_resources.items():
        # Don't divide by zero.
        if v < 1:
            # Could test v == 0 on the nose, but v < 1 feels safer.
            # (Note that node resources are integers.)
            continue
        if k in resource_types:
            num_matching_resource_types += 1
        util = (v - remaining[k]) / v
        util_by_resources.append(v * (util**3))

    # Could happen if node_resources has only zero values.
    if not util_by_resources:
        return None

    # Prefer not to launch a GPU node if there aren't any GPU requirements in the
    # resource bundle.
    gpu_ok = True
    if CLOUDTIK_CONSERVE_GPU_NODES:
        is_gpu_node = "GPU" in node_resources and node_resources["GPU"] > 0
        any_gpu_task = any("GPU" in r for r in resources)
        if is_gpu_node and not any_gpu_task:
            gpu_ok = False

    # Prioritize avoiding gpu nodes for non-gpu workloads first,
    # then prioritize matching multiple resource types,
    # then prioritize using all resources,
    # then prioritize overall balance of multiple resources.
    return (
        gpu_ok,
        num_matching_resource_types,
        min(util_by_resources),
        # util_by_resources should be non empty
        float(sum(util_by_resources)) / len(util_by_resources),
    )


def get_bin_pack_residual(node_resources: List[ResourceDict],
                          resource_demands: List[ResourceDict],
                          strict_spread: bool = False
                          ) -> (List[ResourceDict], List[ResourceDict]):
    """Return a subset of resource_demands that cannot fit in the cluster.

    Args:
        node_resources (List[ResourceDict]): List of resources per node.
        resource_demands (List[ResourceDict]): List of resource bundles that
            need to be bin packed onto the nodes.
        strict_spread (bool): If true, each element in resource_demands must be
            placed on a different entry in `node_resources`.

    Returns:
        List[ResourceDict]: the residual list resources that do not fit.
        List[ResourceDict]: The updated node_resources after the method.
    """

    unfulfilled = []

    # A most naive bin packing algorithm.
    nodes = copy.deepcopy(node_resources)
    # List of nodes that cannot be used again due to strict spread.
    used = []
    # We order the resource demands in the following way:
    # More complex demands first.
    # Break ties: heavier demands first.
    # Break ties: lexicographically (to ensure stable ordering).
    for demand in sorted(
            resource_demands,
            key=lambda demand: (len(demand.values()),
                                sum(demand.values()),
                                sorted(demand.items())),
            reverse=True):
        found = False
        node = None
        for i in range(len(nodes)):
            node = nodes[i]
            if _fits(node, demand):
                found = True
                # In the strict_spread case, we can't reuse nodes.
                if strict_spread:
                    used.append(node)
                    del nodes[i]
                break
        if found and node:
            _inplace_subtract(node, demand)
        else:
            unfulfilled.append(demand)

    return unfulfilled, nodes + used


def _fits(node: ResourceDict, resources: ResourceDict) -> bool:
    for k, v in resources.items():
        if v > node.get(k, 0.0):
            return False
    return True


def _inplace_subtract(node: ResourceDict, resources: ResourceDict) -> None:
    for k, v in resources.items():
        if v == 0:
            # This is an edge case since someone can
            # do `cloudtik.core.api.Cluster.scale(resources={"GPU": 0}"})`.
            continue
        assert k in node, (k, node)
        node[k] -= v
        assert node[k] >= 0.0, (node, k, v)


def _inplace_add(a: collections.defaultdict, b: Dict) -> None:
    """Generically adds values in `b` to `a`.
    a[k] should be defined for all k in b.keys()"""
    for k, v in b.items():
        a[k] += v


def get_node_type_counts(
        provider, nodes: List[NodeID], pending_nodes: Dict[NodeID, int],
        node_types
) -> Dict[NodeType, int]:
    """Returns node type counts.
       Counts the running nodes, pending nodes.
    """
    node_type_counts = collections.defaultdict(int)

    def add_node(node_type):
        if node_type not in node_types:
            # We should not get here
            return
        node_type_counts[node_type] += 1

    for node_id in nodes:
        tags = provider.node_tags(node_id)
        if CLOUDTIK_TAG_USER_NODE_TYPE in tags:
            node_type = tags[CLOUDTIK_TAG_USER_NODE_TYPE]
            add_node(node_type)

    for node_type, count in pending_nodes.items():
        for _ in range(count):
            add_node(node_type)
    return node_type_counts


def get_unfulfilled_for_bundles(
        bundles: List[ResourceDict], node_types, node_type_counts):
    max_node_resources = []
    # Fit request_resources() on all the resources as if they are idle.
    for node_type in node_type_counts:
        max_node_resources.extend([
            copy.deepcopy(node_types[node_type]["resources"])
            for _ in range(node_type_counts[node_type])
        ])
    # Get the unfulfilled to ensure min cluster size.
    resource_requests_unfulfilled, _ = get_bin_pack_residual(
        max_node_resources, bundles)
    return resource_requests_unfulfilled
