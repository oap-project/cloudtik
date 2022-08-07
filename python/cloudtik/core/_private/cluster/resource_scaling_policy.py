import json
import logging
import time

from cloudtik.core._private.state.resource_state import ResourceStateClient, ClusterResourceState

CLOUDTIK_CLUSTER_RESOURCE_TIGHT_THRESHOLD = 4

logger = logging.getLogger(__name__)


class ResourceScalingPolicy:
    def __init__(self,
                 head_ip, port,
                 resource_state_client: ResourceStateClient):
        self.head_ip = head_ip
        self.port = port
        self.resource_state_client = resource_state_client
        self.config = None

    def reset(self, config):
        self.config = config

    def update(self):
        # Pulling data from resource management system
        autoscaling_instructions = self.get_autoscaling_instructions()
        node_resource_states = self.get_node_resource_states()

        cluster_resource_state = ClusterResourceState()
        cluster_resource_state.set_autoscaling_instructions(autoscaling_instructions)
        cluster_resource_state.set_node_resource_states(node_resource_states)

        self.resource_state_client.update_cluster_resource_state(cluster_resource_state)

    def need_more_resources(self, cluster_metrics):
        # TODO: Refine the algorithm here for better scaling decisions
        if (cluster_metrics["appsPending"] > 0
                and cluster_metrics["availableVirtualCores"] < CLOUDTIK_CLUSTER_RESOURCE_TIGHT_THRESHOLD):
            return True

        return False

    def get_autoscaling_instructions(self):
        import urllib.request
        import urllib.error
        cluster_metrics_url = "http://{}:{}/ws/v1/cluster/metrics".format(self.head_ip, self.port)
        try:
            response = urllib.request.urlopen(cluster_metrics_url, timeout=10)
            content = response.read()
        except urllib.error.HTTPError as e:
            logger.error("Failed to retrieve the cluster metrics: {}", str(e))
            return None

        cluster_metrics = json.loads(content)

        # Use the following information to make the decisions
        """
        "appsPending": 0,
        "appsRunning": 0,
        "availableMB": 17408,
        "allocatedMB": 0,
        "availableVirtualCores": 7,
        "allocatedVirtualCores": 1,
        "containersAllocated": 0,
        "containersReserved": 0,
        "containersPending": 0,
        "totalMB": 17408,
        "totalVirtualCores": 8,
        """
        autoscaling_instructions = {}

        resource_demands = []
        if self.need_more_resources(cluster_metrics):
            # There are applications cannot have the resource to run
            # We request more resources with a configured step of up scaling speed
            # TODO: improvement with the right number of cores request and the number of demands
            resource_demand = {
                "CPU": 4
            }
            resource_demands.append(resource_demand)

        autoscaling_instructions["resource_demands"] = resource_demands
        return autoscaling_instructions

    def get_node_resource_states(self):
        import urllib.request
        import urllib.error
        cluster_nodes_url = "http://{}:{}/ws/v1/cluster/nodes".format(self.head_ip, self.port)
        try:
            response = urllib.request.urlopen(cluster_nodes_url, timeout=10)
            content = response.read()
        except urllib.error.HTTPError as e:
            logger.error("Failed to retrieve the cluster nodes metrics: {}", str(e))
            return None

        cluster_nodes_metrics = json.loads(content)
        node_resource_states = {}
        if ("nodes" in cluster_nodes_metrics
                and "node" in cluster_nodes_metrics["nodes"]):
            node_list = cluster_nodes_metrics["nodes"]["node"]
            now = time.time()
            for node in node_list:
                # TODO: get the right id and ip
                node_id = node["id"]
                node_ip = node["nodeHostName"]
                total_resources = {
                    "CPU": node["availableVirtualCores"],
                    "memory": int(node["availMemoryMB"]) * 1024 * 1024
                }
                free_resources = {
                    "CPU": node["availableVirtualCores"] - node["usedVirtualCores"],
                    "memory": int(node["availMemoryMB"] - node["usedMemoryMB"]) * 1024 * 1024
                }
                cpu_load = 0.0
                if "resourceUtilization" in node:
                    cpu_load = node["resourceUtilization"].get("nodeCPUUsage", 0.0)
                resource_load = {
                    "utilization": {
                        "CPU": cpu_load
                    },
                    "in_use": True if node["numContainers"] > 0 else False
                }
                node_resource_state = {
                    "node_id": node_id,
                    "node_ip": node_ip,
                    "resource_time": now,
                    "total_resources": total_resources,
                    "free_resources": free_resources,
                    "resource_load": resource_load
                }
                node_resource_states[node_id] = node_resource_state

        return node_resource_states
