import copy
import json
import logging
import time
from pathlib import Path
from threading import RLock
from uuid import uuid4
from typing import Any, Dict, List

from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.resource.resources.models import DeploymentMode
from azure.mgmt.msi import ManagedServiceIdentityClient

from cloudtik.core._private.cli_logger import cli_logger
from cloudtik.core.node_provider import NodeProvider
from cloudtik.core.tags import CLOUDTIK_TAG_CLUSTER_NAME, CLOUDTIK_TAG_NODE_NAME

from cloudtik.providers._private._azure.config import (AZURE_MSI_NAME, get_azure_sdk_function,
                                                       verify_azure_cloud_storage, bootstrap_azure)

from cloudtik.providers._private._azure.utils import get_azure_config
from cloudtik.providers._private.utils import validate_config_dict

VM_NAME_MAX_LEN = 64
VM_NAME_UUID_LEN = 8
RESOURCE_CHECK_TIME = 20

logger = logging.getLogger(__name__)
azure_logger = logging.getLogger(
    "azure.core.pipeline.policies.http_logging_policy")
azure_logger.setLevel(logging.WARNING)


def synchronized(f):
    def wrapper(self, *args, **kwargs):
        self.lock.acquire()
        try:
            return f(self, *args, **kwargs)
        finally:
            self.lock.release()

    return wrapper


def get_credential(provider_config):
    managed_identity_client_id = provider_config.get("managed_identity_client_id")
    if managed_identity_client_id is None:
        # No managed identity
        credential = DefaultAzureCredential(
            exclude_managed_identity_credential=True,
            exclude_shared_token_cache_credential=True)
    else:
        credential = DefaultAzureCredential(
            exclude_shared_token_cache_credential=True,
            managed_identity_client_id=managed_identity_client_id
        )
    return credential


class AzureNodeProvider(NodeProvider):
    """Node Provider for Azure

    This provider assumes Azure credentials are set by running ``az login``
    and the default subscription is configured through ``az account``
    or set in the ``provider`` field of the scaler configuration.

    Nodes may be in one of three states: {pending, running, terminated}. Nodes
    appear immediately once started by ``create_node``, and transition
    immediately to terminated when ``terminate_node`` is called.
    """

    def __init__(self, provider_config, cluster_name):
        NodeProvider.__init__(self, provider_config, cluster_name)
        subscription_id = provider_config["subscription_id"]
        self.credential = get_credential(provider_config)
        self.compute_client = ComputeManagementClient(self.credential,
                                                      subscription_id)
        self.network_client = NetworkManagementClient(self.credential,
                                                      subscription_id)
        self.resource_client = ResourceManagementClient(
            self.credential, subscription_id)

        self.lock = RLock()

        # cache node objects
        self.cached_nodes = {}

    def with_environment_variables(self):
        return get_azure_config(self.provider_config)

    @synchronized
    def _get_filtered_nodes(self, tag_filters):
        def match_tags(vm):
            for k, v in tag_filters.items():
                if vm.tags is None or vm.tags.get(k) != v:
                    return False
            return True

        vms = self.compute_client.virtual_machines.list(
            resource_group_name=self.provider_config["resource_group"])

        nodes = [self._extract_metadata(vm) for vm in filter(match_tags, vms)]
        self.cached_nodes = {node["name"]: node for node in nodes}
        return self.cached_nodes

    def _extract_metadata(self, vm):
        # get tags
        metadata = {"name": vm.name, "tags": vm.tags, "status": "", "vm_size": ""}

        # get status
        resource_group = self.provider_config["resource_group"]
        instance = self.compute_client.virtual_machines.instance_view(
            resource_group_name=resource_group, vm_name=vm.name).as_dict()
        for status in instance["statuses"]:
            status_list = status["code"].split("/")
            code = status_list[0]
            state = status_list[1]
            # skip provisioning status
            if code == "PowerState":
                metadata["status"] = state
                break

        # get ip data
        nic_id = vm.network_profile.network_interfaces[0].id
        metadata["nic_name"] = nic_id.split("/")[-1]
        nic = self.network_client.network_interfaces.get(
            resource_group_name=resource_group,
            network_interface_name=metadata["nic_name"])
        ip_config = nic.ip_configurations[0]

        public_ip_address = ip_config.public_ip_address
        if public_ip_address is not None:
            public_ip_id = public_ip_address.id
            metadata["public_ip_name"] = public_ip_id.split("/")[-1]
            public_ip = self.network_client.public_ip_addresses.get(
                resource_group_name=resource_group,
                public_ip_address_name=metadata["public_ip_name"])
            metadata["external_ip"] = public_ip.ip_address
        else:
            metadata["external_ip"] = None

        metadata["internal_ip"] = ip_config.private_ip_address

        # get vmSize
        metadata["vm_size"] = vm.hardware_profile.vm_size

        return metadata

    def non_terminated_nodes(self, tag_filters):
        """Return a list of node ids filtered by the specified tags dict.

        This list must not include terminated nodes. For performance reasons,
        providers are allowed to cache the result of a call to nodes() to
        serve single-node queries (e.g. is_running(node_id)). This means that
        nodes() must be called again to refresh results.

        Examples:
            >>> provider.non_terminated_nodes({CLOUDTIK_TAG_NODE_KIND: "worker"})
            ["node-1", "node-2"]
        """
        cluster_name_tag = {CLOUDTIK_TAG_CLUSTER_NAME: self.cluster_name}
        tag_filters.update(cluster_name_tag)
        nodes = self._get_filtered_nodes(tag_filters=tag_filters)
        return [
            k for k, v in nodes.items()
            if not v["status"].startswith("deallocat")
        ]

    def get_node_info(self, node_id):
        node = self._get_cached_node(node_id)
        node_info = {"node_id": node["name"].split("-")[-1],
                     "instance_type": node["vm_size"],
                     "private_ip": node["internal_ip"],
                     "public_ip": node["external_ip"],
                     "instance_status": node["status"]}
        node_info.update(self.node_tags(node_id))

        return node_info

    def is_running(self, node_id):
        """Return whether the specified node is running."""
        # always get current status
        node = self._get_node(node_id=node_id)
        return node["status"] == "running"

    def is_terminated(self, node_id):
        """Return whether the specified node is terminated."""
        # always get current status
        node = self._get_node(node_id=node_id)
        return node["status"].startswith("deallocat")

    def node_tags(self, node_id):
        """Returns the tags of the given node (string dict)."""
        return self._get_cached_node(node_id=node_id)["tags"]

    def external_ip(self, node_id):
        """Returns the external ip of the given node."""
        ip = (self._get_cached_node(node_id=node_id)["external_ip"]
              or self._get_node(node_id=node_id)["external_ip"])
        return ip

    def internal_ip(self, node_id):
        """Returns the internal ip of the given node."""
        ip = (self._get_cached_node(node_id=node_id)["internal_ip"]
              or self._get_node(node_id=node_id)["internal_ip"])
        return ip

    def create_node(self, node_config, tags, count):
        """Creates a number of nodes within the namespace."""
        # TODO: restart deallocated nodes if possible
        resource_group = self.provider_config["resource_group"]
        use_internal_ips = self.provider_config.get("use_internal_ips", False)
        # load the template file
        current_path = Path(__file__).parent
        template_path = current_path.joinpath("azure-vm-template.json")
        with open(template_path, "r") as template_fp:
            template = json.load(template_fp)

        # get the tags
        config_tags = node_config.get("tags", {}).copy()
        config_tags.update(tags)
        config_tags[CLOUDTIK_TAG_CLUSTER_NAME] = self.cluster_name

        name_tag = config_tags.get(CLOUDTIK_TAG_NODE_NAME, "node")
        unique_id = uuid4().hex[:VM_NAME_UUID_LEN]
        vm_name = "{name}-{id}".format(name=name_tag, id=unique_id)

        template_params = node_config["azure_arm_parameters"].copy()
        template_params["vmName"] = vm_name
        template_params["vmTags"] = config_tags
        template_params["vmCount"] = count

        parameters = {
            "properties": {
                "mode": DeploymentMode.incremental,
                "template": template,
                "parameters": {
                    key: {
                        "value": value
                    }
                    for key, value in template_params.items()
                }
            }
        }

        # TODO: we could get the private/public ips back directly
        create_or_update = get_azure_sdk_function(
            client=self.resource_client.deployments,
            function_name="create_or_update")
        create_or_update(
            resource_group_name=resource_group,
            deployment_name="cloudtik-vm-{}".format(name_tag),
            parameters=parameters).wait()

    @synchronized
    def set_node_tags(self, node_id, tags):
        """Sets the tag values (string dict) for the specified node."""
        node_tags = self._get_cached_node(node_id)["tags"]
        node_tags.update(tags)
        update = get_azure_sdk_function(
            client=self.compute_client.virtual_machines,
            function_name="update")
        update(
            resource_group_name=self.provider_config["resource_group"],
            vm_name=node_id,
            parameters={"tags": node_tags})
        self.cached_nodes[node_id]["tags"] = node_tags

    def terminate_node(self, node_id):
        """Terminates the specified node. This will delete the VM and
           associated resources (NIC, IP, Storage) for the specified node."""

        resource_group = self.provider_config["resource_group"]
        try:
            # get metadata for node
            metadata = self._get_node(node_id)
        except KeyError:
            # node no longer exists
            return

        # TODO: deallocate instead of delete to allow possible reuse
        # self.compute_client.virtual_machines.deallocate(
        #   resource_group_name=resource_group,
        #   vm_name=node_id)

        # gather disks to delete later
        vm = self.compute_client.virtual_machines.get(
            resource_group_name=resource_group, vm_name=node_id)
        disks = {d.name for d in vm.storage_profile.data_disks}
        disks.add(vm.storage_profile.os_disk.name)

        try:
            # delete machine, must wait for this to complete
            delete = get_azure_sdk_function(
                client=self.compute_client.virtual_machines,
                function_name="delete")
            delete(resource_group_name=resource_group, vm_name=node_id).wait()
        except Exception as e:
            logger.warning("Failed to delete VM: {}".format(e))

        try:
            # delete nic
            delete = get_azure_sdk_function(
                client=self.network_client.network_interfaces,
                function_name="delete")
            delete(
                resource_group_name=resource_group,
                network_interface_name=metadata["nic_name"])
        except Exception as e:
            logger.warning("Failed to delete nic: {}".format(e))

        # delete ip address
        if "public_ip_name" in metadata:
            retry_time = RESOURCE_CHECK_TIME
            delete = get_azure_sdk_function(
                client=self.network_client.public_ip_addresses,
                function_name="delete")
            cli_logger.print("Deleting public ip address...")
            while retry_time > 0:
                try:
                    delete(
                        resource_group_name=resource_group,
                        public_ip_address_name=metadata["public_ip_name"])
                    cli_logger.print("Successfully deleted public ip address.")
                    break
                except Exception as e:
                    retry_time = retry_time - 1
                    if retry_time > 0:
                        cli_logger.warning(
                            "Failed to delete public ip address. "
                            "Remaining {} tries to delete public ip address...".format(retry_time))
                        time.sleep(1)
                    else:
                        cli_logger.error("Failed to delete public ip address. {}", str(e))

        # delete disks
        for disk in disks:
            try:
                delete = get_azure_sdk_function(
                    client=self.compute_client.disks, function_name="delete")
                delete(resource_group_name=resource_group, disk_name=disk)
            except Exception as e:
                logger.warning("Failed to delete disk: {}".format(e))

    def _get_node(self, node_id):
        self._get_filtered_nodes({})  # Side effect: updates cache
        return self.cached_nodes[node_id]

    def _get_cached_node(self, node_id):
        if node_id in self.cached_nodes:
            return self.cached_nodes[node_id]
        return self._get_node(node_id=node_id)

    @staticmethod
    def bootstrap_config(cluster_config):
        return bootstrap_azure(cluster_config)

    def prepare_for_head_node(
            self, cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a new cluster config with custom configs for head node."""
        managed_identity_client_id = self._get_managed_identity_client_id(cluster_config)
        if managed_identity_client_id:
            cluster_config["provider"]["managed_identity_client_id"] = managed_identity_client_id

        return cluster_config

    @staticmethod
    def fillout_available_node_types_resources(
            cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Fills out missing "resources" field for available_node_types."""
        if "available_node_types" not in cluster_config:
            return cluster_config
        cluster_config = copy.deepcopy(cluster_config)

        # Get instance information from cloud provider
        provider_config = cluster_config["provider"]
        subscription_id = provider_config["subscription_id"]
        vm_location = provider_config["location"]

        credential = get_credential(provider_config)
        compute_client = ComputeManagementClient(credential, subscription_id)

        vmsizes = compute_client.virtual_machine_sizes.list(vm_location)
        instances_dict = {
            instance.name: {"memory": instance.memory_in_mb, "cpu": instance.number_of_cores}
            for instance in vmsizes
        }

        # Update the instance information to node type
        available_node_types = cluster_config["available_node_types"]
        for node_type in available_node_types:
            instance_type = available_node_types[node_type]["node_config"]["azure_arm_parameters"]["vmSize"]
            if instance_type in instances_dict:
                cpus = instances_dict[instance_type]["cpu"]
                detected_resources = {"CPU": cpus}

                memory_total = instances_dict[instance_type]["memory"]
                memory_total_in_bytes = int(memory_total) * 1024 * 1024
                detected_resources["memory"] = memory_total_in_bytes

                detected_resources.update(
                    available_node_types[node_type].get("resources", {}))
                if detected_resources != \
                        available_node_types[node_type].get("resources", {}):
                    available_node_types[node_type][
                        "resources"] = detected_resources
                    logger.debug("Updating the resources of {} to {}.".format(
                        node_type, detected_resources))
            else:
                raise ValueError("Instance type " + instance_type +
                                 " is not available in Azure location: " +
                                 vm_location + ".")
        return cluster_config

    @staticmethod
    def validate_config(
            provider_config: Dict[str, Any]) -> None:
        config_dict = {
            "subscription_id": provider_config.get("subscription_id"),
            "resource_group": provider_config.get("resource_group")}

        validate_config_dict(provider_config["type"], config_dict)

    @staticmethod
    def validate_storage_config(
            provider_config: Dict[str, Any]) -> None:
        config_dict = {
            "azure.storage.type": provider_config.get("azure_cloud_storage", {}).get("azure.storage.type"),
            "azure.storage.account": provider_config.get("azure_cloud_storage", {}).get("azure.storage.account"),
            "azure.container": provider_config.get("azure_cloud_storage", {}).get("azure.container"),
            "azure.account.key": provider_config.get("azure_cloud_storage", {}).get("azure.account.key")}

        validate_config_dict(provider_config["type"], config_dict)

        verify_cloud_storage = provider_config.get("verify_cloud_storage", True)
        if verify_cloud_storage:
            cli_logger.verbose("Verifying Azure cloud storage configurations...")
            verify_azure_cloud_storage(provider_config)
            cli_logger.verbose("Successfully verified Azure cloud storage configurations.")

    def _get_managed_identity_client_id(self, cluster_config):
        try:
            # The latest version doesn't require credential wrapper any longer
            # credential_adapter = AzureIdentityCredentialAdapter(self.credential)
            msi_client = ManagedServiceIdentityClient(self.credential,
                                                      self.provider_config["subscription_id"])

            user_assigned_identity_name = self.provider_config.get("userAssignedIdentity", AZURE_MSI_NAME)
            user_assigned_identity = msi_client.user_assigned_identities.get(
                self.provider_config["resource_group"],
                user_assigned_identity_name)
            return user_assigned_identity.client_id
        except Exception as e:
            logger.warning("Failed to get azure client id: {}".format(e))
            return None
