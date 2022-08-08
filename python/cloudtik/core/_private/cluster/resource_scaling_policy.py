import json
import logging
import time

from cloudtik.core._private.services import address_to_ip
from cloudtik.core._private.state.resource_state import ResourceStateClient, ClusterResourceState
from cloudtik.core._private.utils import make_node_id, get_resource_demands_for_cpu

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
        self.last_resource_demands_time = 0
        self.last_resource_state_snapshot = None

    def reset(self, config):
        self.config = config

    def update(self):
        # Pulling data from resource management system
        autoscaling_instructions = self.get_autoscaling_instructions()
        node_resource_states, lost_nodes = self.get_node_resource_states()

        cluster_resource_state = ClusterResourceState()
        cluster_resource_state.set_autoscaling_instructions(autoscaling_instructions)
        cluster_resource_state.set_node_resource_states(node_resource_states)

        self.resource_state_client.update_cluster_resource_state(
            cluster_resource_state, lost_nodes)

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
        except urllib.error.URLError as e:
            logger.error("Failed to retrieve the cluster metrics: {}", str(e))
            return None

        cluster_metrics_response = json.loads(content)

        # Use the following information to make the decisions
        """
        "appsPending": 0,
        "appsRunning": 0,

        "availableMB": 17408,
        "allocatedMB": 0,
        "totalMB": 17408,

        "availableVirtualCores": 7,
        "allocatedVirtualCores": 1,
        "totalVirtualCores": 8,

        "containersAllocated": 0,
        "containersReserved": 0,
        "containersPending": 0,
        """
        autoscaling_instructions = {}
        resource_demands = []

        if "clusterMetrics" in cluster_metrics_response:
            cluster_metrics = cluster_metrics_response["clusterMetrics"]

            if logger.isEnabledFor(logging.DEBUG):
                cluster_info = {
                    "appsPending": cluster_metrics["appsPending"],
                    "appsRunning": cluster_metrics["appsRunning"],
                    "totalVirtualCores": cluster_metrics["totalVirtualCores"],
                    "allocatedVirtualCores": cluster_metrics["allocatedVirtualCores"],
                    "availableVirtualCores": cluster_metrics["availableVirtualCores"],
                    "activeNodes": cluster_metrics["activeNodes"],
                    "unhealthyNodes": cluster_metrics["unhealthyNodes"],
                }
                logger.debug("Cluster metrics: {}".format(cluster_info))

            if self.need_more_resources(cluster_metrics):
                # There are applications cannot have the resource to run
                # We request more resources with a configured step of up scaling speed
                # TODO: improvement with the right number of cores request and the number of demands
                num_cores = 4
                resource_demands_for_cpu = get_resource_demands_for_cpu(num_cores, self.config)
                resource_demands += resource_demands_for_cpu

                self.last_resource_demands_time = time.time()
                self.last_resource_state_snapshot = {
                    "totalVirtualCores": cluster_metrics["totalVirtualCores"],
                    "allocatedVirtualCores": cluster_metrics["allocatedVirtualCores"],
                    "availableVirtualCores": cluster_metrics["availableVirtualCores"],
                    "requestingVirtualCores": num_cores
                }

                logger.info("Scaling event: {}/{} cpus are free. Requesting {} more cpus...".format(
                    cluster_metrics["availableVirtualCores"], cluster_metrics["totalVirtualCores"], num_cores))

        autoscaling_instructions["resource_demands"] = resource_demands
        if len(resource_demands) > 0:
            logger.debug("Resource demands: {}".format(resource_demands))
        return autoscaling_instructions

    @staticmethod
    def address_to_ip(address):
        try:
            return address_to_ip(address)
        except Exception:
            return None

    def get_node_resource_states(self):
        import urllib.request
        import urllib.error
        cluster_nodes_url = "http://{}:{}/ws/v1/cluster/nodes".format(self.head_ip, self.port)
        try:
            response = urllib.request.urlopen(cluster_nodes_url, timeout=10)
            content = response.read()
        except urllib.error.URLError as e:
            logger.error("Failed to retrieve the cluster nodes metrics: {}", str(e))
            return None, None

        """
        "nodeHostName":"host.domain.com",
        "nodeHTTPAddress":"host.domain.com:8042",
        "lastHealthUpdate": 1476995346399,
        "version": "3.0.0",
        "healthReport":"",
        "numContainers":0,
        "usedMemoryMB":0,
        "availMemoryMB":8192,
        "usedVirtualCores":0,
        "availableVirtualCores":8,
        "resourceUtilization":
        {
          "nodePhysicalMemoryMB":1027,
          "nodeVirtualMemoryMB":1027,
          "nodeCPUUsage":0.016661113128066063,
          "aggregatedContainersPhysicalMemoryMB":0,
          "aggregatedContainersVirtualMemoryMB":0,
          "containersCPUUsage":0
        }
        """

        cluster_nodes_response = json.loads(content)
        node_resource_states = {}
        lost_nodes = {}
        if ("nodes" in cluster_nodes_response
                and "node" in cluster_nodes_response["nodes"]):
            cluster_nodes = cluster_nodes_response["nodes"]["node"]
            now = time.time()
            for node in cluster_nodes:
                host_name = node["nodeHostName"]
                node_ip = address_to_ip(host_name)
                if node_ip is None:
                    continue

                node_id = make_node_id(node_ip)
                if node["state"] != "RUNNING":
                    lost_nodes["node_id"] = node_id
                    continue

                total_resources = {
                    "CPU": node["availableVirtualCores"] + node["usedVirtualCores"],
                    "memory": int(node["availMemoryMB"] + node["usedMemoryMB"]) * 1024 * 1024
                }
                free_resources = {
                    "CPU": node["availableVirtualCores"],
                    "memory": int(node["availMemoryMB"]) * 1024 * 1024
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
                    "available_resources": free_resources,
                    "resource_load": resource_load
                }
                # logger.debug("Node resources: {}".format(node_resource_state))
                node_resource_states[node_id] = node_resource_state

        return node_resource_states, lost_nodes
