import copy
from functools import partial
import os
import logging
import random
import string
from typing import Any, Dict, Optional

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

from googleapiclient import discovery, errors

from google.oauth2 import service_account

from cloudtik.core.workspace_provider import Existence, CLOUDTIK_MANAGED_CLOUD_STORAGE, \
    CLOUDTIK_MANAGED_CLOUD_STORAGE_URI

from cloudtik.core.tags import CLOUDTIK_TAG_NODE_KIND, NODE_KIND_HEAD, CLOUDTIK_TAG_CLUSTER_NAME
from cloudtik.core._private.cli_logger import cli_logger, cf
from cloudtik.core._private.services import get_node_ip_address
from cloudtik.core._private.utils import check_cidr_conflict, unescape_private_key, is_use_internal_ip, \
    is_managed_cloud_storage, is_use_managed_cloud_storage, is_worker_role_for_cloud_storage, \
    _is_use_managed_cloud_storage, is_use_peering_vpc, is_use_working_vpc, _is_use_working_vpc, \
    is_peering_firewall_allow_working_subnet, is_peering_firewall_allow_ssh_only
from cloudtik.providers._private.gcp.node import GCPCompute
from cloudtik.providers._private.gcp.utils import _get_node_info, construct_clients_from_provider_config, \
    wait_for_compute_global_operation, wait_for_compute_region_operation, _create_storage, \
    wait_for_crm_operation, HAS_TPU_PROVIDER_FIELD, _is_head_node_a_tpu, _has_tpus_in_node_configs, \
    export_gcp_cloud_storage_config, get_service_account_email, construct_storage_client, construct_storage, \
    get_gcp_cloud_storage_config, get_gcp_cloud_storage_config_for_update, GCP_GCS_BUCKET, get_gcp_cloud_storage_uri
from cloudtik.providers._private.utils import StorageTestingError

logger = logging.getLogger(__name__)

VERSION = "v1"

GCP_RESOURCE_NAME_PREFIX = "cloudtik"

# Those roles will always be added.
HEAD_SERVICE_ACCOUNT_ROLES = [
    "roles/storage.admin", "roles/compute.admin",
    "roles/iam.serviceAccountUser"
]

GCP_HEAD_SERVICE_ACCOUNT_ID = GCP_RESOURCE_NAME_PREFIX + "-{}"
GCP_HEAD_SERVICE_ACCOUNT_DISPLAY_NAME = "CloudTik Head Service Account - {}"

GCP_WORKER_SERVICE_ACCOUNT_ID = GCP_RESOURCE_NAME_PREFIX + "-w-{}"
GCP_WORKER_SERVICE_ACCOUNT_DISPLAY_NAME = "CloudTik Worker Service Account - {}"

GCP_WORKSPACE_VPC_NAME = GCP_RESOURCE_NAME_PREFIX + "-{}-vpc"

GCP_WORKSPACE_VPC_PEERING_NAME = GCP_RESOURCE_NAME_PREFIX + "-{}-a-peer"
GCP_WORKING_VPC_PEERING_NAME = GCP_RESOURCE_NAME_PREFIX + "-{}-b-peer"

# Those roles will always be added.
WORKER_SERVICE_ACCOUNT_ROLES = [
    "roles/storage.admin",
    "roles/iam.serviceAccountUser"
]

# Those roles will only be added if there are TPU nodes defined in config.
TPU_SERVICE_ACCOUNT_ROLES = ["roles/tpu.admin"]

# NOTE: iam.serviceAccountUser allows the Head Node to create worker nodes
# with ServiceAccounts.

GCP_WORKSPACE_NUM_CREATION_STEPS = 7
GCP_WORKSPACE_NUM_DELETION_STEPS = 6
GCP_WORKSPACE_TARGET_RESOURCES = 8

GCP_MANAGED_STORAGE_GCS_BUCKET = "gcp.managed.storage.gcs.bucket"


def key_pair_name(i, region, project_id, ssh_user):
    """Returns the ith default gcp_key_pair_name."""
    key_name = "{}_gcp_{}_{}_{}_{}".format(GCP_RESOURCE_NAME_PREFIX, region, project_id, ssh_user,
                                           i)
    return key_name


def key_pair_paths(key_name):
    """Returns public and private key paths for a given key_name."""
    public_key_path = os.path.expanduser("~/.ssh/{}.pub".format(key_name))
    private_key_path = os.path.expanduser("~/.ssh/{}.pem".format(key_name))
    return public_key_path, private_key_path


def generate_rsa_key_pair():
    """Create public and private ssh-keys."""

    key = rsa.generate_private_key(
        backend=default_backend(), public_exponent=65537, key_size=2048)

    public_key = key.public_key().public_bytes(
        serialization.Encoding.OpenSSH,
        serialization.PublicFormat.OpenSSH).decode("utf-8")

    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()).decode("utf-8")

    return public_key, pem


def post_prepare_gcp(config: Dict[str, Any]) -> Dict[str, Any]:
    config = copy.deepcopy(config)
    config = _configure_project_id(config)

    try:
        config = fill_available_node_types_resources(config)
    except Exception as exc:
        cli_logger.warning(
            "Failed to detect node resources. Make sure you have properly configured the GCP credentials: {}.",
            str(exc))
        raise
    return config


def fill_available_node_types_resources(
        cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    """Fills out missing "resources" field for available_node_types."""
    if "available_node_types" not in cluster_config:
        return cluster_config

    # Get instance information from cloud provider
    provider_config = cluster_config["provider"]
    _, _, compute, tpu = construct_clients_from_provider_config(
        provider_config)

    response = compute.machineTypes().list(
        project=provider_config["project_id"],
        zone=provider_config["availability_zone"],
    ).execute()

    instances_list = response.get("items", [])
    instances_dict = {
        instance["name"]: instance
        for instance in instances_list
    }

    # Update the instance information to node type
    available_node_types = cluster_config["available_node_types"]
    for node_type in available_node_types:
        instance_type = available_node_types[node_type]["node_config"][
            "machineType"]
        if instance_type in instances_dict:
            cpus = instances_dict[instance_type]["guestCpus"]
            detected_resources = {"CPU": cpus}

            memory_total = instances_dict[instance_type]["memoryMb"]
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
                             " is not available in GCP zone: " +
                             provider_config["availability_zone"] + ".")
    return cluster_config


def get_workspace_head_nodes(provider_config, workspace_name):
    _, _, compute, tpu = \
        construct_clients_from_provider_config(provider_config)
    return _get_workspace_head_nodes(
        provider_config, workspace_name, compute=compute)


def _get_workspace_head_nodes(provider_config, workspace_name, compute):
    use_working_vpc = _is_use_working_vpc(provider_config)
    project_id = provider_config.get("project_id")
    availability_zone = provider_config.get("availability_zone")
    vpc_id = _get_gcp_vpc_id(
        provider_config, workspace_name, compute, use_working_vpc)
    if vpc_id is None:
        raise RuntimeError(
            "Failed to get the VPC. The workspace {} doesn't exist or is in the wrong state.".format(
                workspace_name
            ))
    vpc_self_link = compute.networks().get(project=project_id, network=vpc_id).execute()["selfLink"]

    filter_expr = '(labels.{key} = {value}) AND (status = RUNNING)'.\
        format(key=CLOUDTIK_TAG_NODE_KIND, value=NODE_KIND_HEAD)

    response = compute.instances().list(
        project=project_id,
        zone=availability_zone,
        filter=filter_expr,
    ).execute()

    all_heads = response.get("items", [])
    workspace_heads = []
    for head in all_heads:
        in_workspace = False
        for networkInterface in head.get("networkInterfaces", []):
            if networkInterface.get("network") == vpc_self_link:
                in_workspace = True
        if in_workspace:
            workspace_heads.append(head)

    return workspace_heads


def create_gcp_workspace(config):
    config = copy.deepcopy(config)

    # Steps of configuring the workspace
    config = _create_workspace(config)

    return config


def _create_workspace(config):
    crm, iam, compute, tpu = \
        construct_clients_from_provider_config(config["provider"])
    workspace_name = config["workspace_name"]
    managed_cloud_storage = is_managed_cloud_storage(config)
    use_peering_vpc = is_use_peering_vpc(config)

    current_step = 1
    total_steps = GCP_WORKSPACE_NUM_CREATION_STEPS
    if managed_cloud_storage:
        total_steps += 1
    if use_peering_vpc:
        total_steps += 1

    try:
        with cli_logger.group("Creating workspace: {}", workspace_name):
            with cli_logger.group(
                    "Configuring project",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                config = _configure_project(config, crm)

            current_step = _create_network_resources(config, current_step, total_steps)

            with cli_logger.group(
                    "Creating service accounts",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                config = _create_workspace_service_accounts(config, crm, iam)

            if managed_cloud_storage:
                with cli_logger.group(
                        "Creating GCS bucket",
                        _numbered=("[]", current_step, total_steps)):
                    current_step += 1
                    config = _create_workspace_cloud_storage(config)

    except Exception as e:
        cli_logger.error("Failed to create workspace with the name {}. "
                         "You need to delete and try create again. {}", workspace_name, str(e))
        raise e

    cli_logger.success(
        "Successfully created workspace: {}.",
        cf.bold(workspace_name))

    return config


def get_workspace_vpc_id(config, compute):
    return _get_workspace_vpc_id(
        config["provider"], config["workspace_name"], compute)


def _get_workspace_vpc_name(workspace_name):
    return GCP_WORKSPACE_VPC_NAME.format(workspace_name)


def _get_workspace_vpc_id(provider_config, workspace_name, compute):
    project_id = provider_config.get("project_id")
    vpc_name = _get_workspace_vpc_name(workspace_name)
    cli_logger.verbose("Getting the VPC Id for workspace: {}...".format(vpc_name))

    vpc_ids = [vpc["id"] for vpc in compute.networks().list(project=project_id).execute().get("items", "")
               if vpc["name"] == vpc_name]
    if len(vpc_ids) == 0:
        cli_logger.verbose("The VPC for workspace is not found: {}.".format(vpc_name))
        return None
    else:
        cli_logger.verbose_error("Successfully get the VPC Id of {} for workspace.".format(vpc_name))
        return vpc_ids[0]


def _delete_vpc(config, compute):
    use_working_vpc = is_use_working_vpc(config)
    if use_working_vpc:
        cli_logger.print("Will not delete the current working VPC.")
        return

    vpc_id = get_workspace_vpc_id(config, compute)
    project_id = config["provider"].get("project_id")
    vpc_name = _get_workspace_vpc_name(config["workspace_name"])

    if vpc_id is None:
        cli_logger.print("The VPC: {} doesn't exist.".format(vpc_name))
        return

    """ Delete the VPC """
    cli_logger.print("Deleting the VPC: {}...".format(vpc_name))

    try:
        operation = compute.networks().delete(project=project_id, network=vpc_id).execute()
        wait_for_compute_global_operation(project_id, operation, compute)
        cli_logger.print("Successfully deleted the VPC: {}.".format(vpc_name))
    except Exception as e:
        cli_logger.error("Failed to delete the VPC: {}. {}", vpc_name, str(e))
        raise e


def create_vpc(config, compute):
    project_id = config["provider"].get("project_id")
    vpc_name = _get_workspace_vpc_name(config["workspace_name"])
    network_body = {
        "autoCreateSubnetworks": False,
        "description": "Auto created network by cloudtik",
        "name": vpc_name,
        "routingConfig": {
            "routingMode": "REGIONAL"
        },
        "mtu": 1460
    }

    cli_logger.print("Creating workspace VPC: {}...", vpc_name)
    # create vpc
    try:
        operation = compute.networks().insert(project=project_id, body=network_body).execute()
        wait_for_compute_global_operation(project_id, operation, compute)
        cli_logger.print("Successfully created workspace VPC: {}", vpc_name)
    except Exception as e:
        cli_logger.error("Failed to create workspace VPC. {}", str(e))
        raise e


def get_vpc_name_by_id(config, compute, vpc_id):
    provider_config = config["provider"]
    project_id = provider_config.get("project_id")
    return compute.networks().get(project=project_id, network=vpc_id).execute()["name"]


def get_working_node_vpc_id(config, compute):
    return _get_working_node_vpc_id(config["provider"], compute)


def get_working_node_vpc_name(config, compute):
    return _get_working_node_vpc_name(config["provider"], compute)


def _find_working_node_network_interface(provider_config, compute):
    ip_address = get_node_ip_address(address="8.8.8.8:53")
    project_id = provider_config.get("project_id")
    zone = provider_config.get("availability_zone")
    instances = compute.instances().list(project=project_id, zone=zone).execute()["items"]
    for instance in instances:
        for networkInterface in instance.get("networkInterfaces"):
            if networkInterface.get("networkIP") == ip_address:
                return networkInterface
    return None


def _find_working_node_vpc(provider_config, compute):
    network_interface = _find_working_node_network_interface(provider_config, compute)
    if network_interface is None:
        cli_logger.verbose_error(
            "Failed to get the VPC of the working node. "
            "Please check whether the working node is a GCP instance.")
        return None

    network = network_interface.get("network").split("/")[-1]
    cli_logger.verbose("Successfully get the VPC for working node.")
    return network


def _split_subnetwork_info(project_id, subnetwork_url):
    info = subnetwork_url.split("projects/" + project_id + "/regions/")[-1].split("/")
    subnetwork_region = info[0]
    subnet_name = info[-1]
    return subnetwork_region, subnet_name


def _find_working_node_subnetwork(provider_config, compute):
    network_interface = _find_working_node_network_interface(provider_config, compute)
    if network_interface is None:
        return None

    subnetwork = network_interface.get("subnetwork")
    cli_logger.verbose("Successfully get the VPC for working node.")
    return subnetwork


def _get_working_node_vpc(provider_config, compute):
    network = _find_working_node_vpc(provider_config, compute)
    if network is None:
        return None

    project_id = provider_config.get("project_id")
    return compute.networks().get(project=project_id, network=network).execute()


def _get_working_node_vpc_id(provider_config, compute):
    vpc = _get_working_node_vpc(provider_config, compute)
    if vpc is None:
        return None
    return vpc["id"]


def _get_working_node_vpc_name(provider_config, compute):
    vpc = _get_working_node_vpc(provider_config, compute)
    if vpc is None:
        return None
    return vpc["name"]


def _configure_gcp_subnets_cidr(config, compute, vpc_id):
    project_id = config["provider"].get("project_id")
    region = config["provider"].get("region")
    vpc_self_link = compute.networks().get(project=project_id, network=vpc_id).execute()["selfLink"]
    subnets = compute.subnetworks().list(project=project_id, region=region,
                                         filter='((network = \"{}\"))'.format(vpc_self_link)).execute().get("items", [])
    cidr_list = []

    if len(subnets) == 0:
        for i in range(0, 2):
            cidr_list.append("10.0." + str(i) + ".0/24")
    else:
        cidr_blocks = [subnet["ipCidrRange"] for subnet in subnets]
        ip = cidr_blocks[0].split("/")[0].split(".")
        for i in range(0, 256):
            tmp_cidr_block = ip[0] + "." + ip[1] + "." + str(i) + ".0/24"
            if check_cidr_conflict(tmp_cidr_block, cidr_blocks):
                cidr_list.append(tmp_cidr_block)
                cli_logger.print("Choose CIDR: {}".format(tmp_cidr_block))

            if len(cidr_list) == 2:
                break

    return cidr_list


def _delete_subnet(config, compute, is_private=True):
    if is_private:
        subnet_attribute = "private"
    else:
        subnet_attribute = "public"
    project_id = config["provider"].get("project_id")
    region = config["provider"].get("region")
    workspace_name = config["workspace_name"]
    subnet_name = "cloudtik-{}-{}-subnet".format(workspace_name,
                                                     subnet_attribute)

    if get_subnet(config, subnet_name, compute) is None:
        cli_logger.print("The {} subnet {} isn't found in workspace."
                         .format(subnet_attribute, subnet_name))
        return

    # """ Delete custom subnet """
    cli_logger.print("Deleting {} subnet: {}...".format(subnet_attribute, subnet_name))
    try:
        operation = compute.subnetworks().delete(project=project_id, region=region,
                                         subnetwork=subnet_name).execute()
        wait_for_compute_region_operation(project_id, region, operation, compute)
        cli_logger.print("Successfully deleted {} subnet: {}."
                         .format(subnet_attribute, subnet_name))
    except Exception as e:
        cli_logger.error("Failed to delete the {} subnet: {}. {}",
                         subnet_attribute, subnet_name, str(e))
        raise e


def _create_and_configure_subnets(config, compute, vpc_id):
    workspace_name = config["workspace_name"]
    project_id = config["provider"]["project_id"]
    region = config["provider"]["region"]

    cidr_list = _configure_gcp_subnets_cidr(config, compute, vpc_id)
    assert len(cidr_list) == 2, "We must create 2 subnets for VPC: {}!".format(vpc_id)

    subnets_attribute = ["public", "private"]
    for i in range(2):
        subnet_name = "cloudtik-{}-{}-subnet".format(workspace_name, subnets_attribute[i])
        cli_logger.print("Creating subnet for the vpc: {} with CIDR: {}...".format(vpc_id, cidr_list[i]))
        network_body = {
            "description": "Auto created {} subnet for cloudtik".format(subnets_attribute[i]),
            "enableFlowLogs": False,
            "ipCidrRange": cidr_list[i],
            "name": subnet_name,
            "network": "projects/{}/global/networks/{}".format(project_id, vpc_id),
            "stackType": "IPV4_ONLY",
            "privateIpGoogleAccess": False if subnets_attribute[i] == "public" else True,
            "region": region
        }
        try:
            operation = compute.subnetworks().insert(project=project_id, region=region, body=network_body).execute()
            wait_for_compute_region_operation(project_id, region, operation, compute)
            cli_logger.print("Successfully created subnet: {}.".format(subnet_name))
        except Exception as e:
            cli_logger.error("Failed to create subnet. {}",  str(e))
            raise e


def _create_router(config, compute, vpc_id):
    project_id = config["provider"]["project_id"]
    region = config["provider"]["region"]
    workspace_name = config["workspace_name"]
    router_name = "cloudtik-{}-private-router".format(workspace_name)
    vpc_name = _get_workspace_vpc_name(workspace_name)
    cli_logger.print("Creating router for the private subnet: {}...".format(router_name))
    router_body = {
        "bgp": {
            "advertiseMode": "CUSTOM"
        },
        "description": "auto created for the workspace: {}".format(vpc_name),
        "name": router_name,
        "network": "projects/{}/global/networks/{}".format(project_id, vpc_id),
        "region": "projects/{}/regions/{}".format(project_id, region)
    }
    try:
        operation = compute.routers().insert(project=project_id, region=region, body=router_body).execute()
        wait_for_compute_region_operation(project_id, region, operation, compute)
        cli_logger.print("Successfully created router for the private subnet: cloudtik-{}-subnet.".
                     format(config["workspace_name"]))
    except Exception as e:
        cli_logger.error("Failed to create router. {}", str(e))
        raise e


def _create_nat_for_router(config, compute):
    project_id = config["provider"]["project_id"]
    region = config["provider"]["region"]
    workspace_name = config["workspace_name"]
    nat_name = "cloudtik-{}-nat".format(workspace_name)

    cli_logger.print("Creating nat-gateway for private router: {}... ".format(nat_name))

    router = "cloudtik-{}-private-router".format(workspace_name)
    subnet_name = "cloudtik-{}-private-subnet".format(workspace_name)
    private_subnet = get_subnet(config, subnet_name, compute)
    private_subnet_self_link = private_subnet.get("selfLink")
    router_body ={
        "nats": [
            {
                "natIpAllocateOption": "AUTO_ONLY",
                "name": nat_name,
                "subnetworks": [
                    {
                        "sourceIpRangesToNat": [
                            "ALL_IP_RANGES"
                        ],
                        "name": private_subnet_self_link
                    }
                ],
                "sourceSubnetworkIpRangesToNat": "LIST_OF_SUBNETWORKS"
            }
        ]
    }

    try:
        operation = compute.routers().patch(project=project_id, region=region, router=router, body=router_body).execute()
        wait_for_compute_region_operation(project_id, region, operation, compute)
        cli_logger.print("Successfully created nat-gateway for the private router: {}.".
                         format(nat_name))
    except Exception as e:
        cli_logger.error("Failed to create nat-gateway. {}", str(e))
        raise e


def _delete_router(config, compute):
    project_id = config["provider"]["project_id"]
    region = config["provider"]["region"]
    workspace_name = config["workspace_name"]
    router_name = "cloudtik-{}-private-router".format(workspace_name)

    if get_router(config, router_name, compute) is None:
        return

    # """ Delete custom subnet """
    cli_logger.print("Deleting the router: {}...".format(router_name))
    try:
        operation = compute.routers().delete(project=project_id, region=region, router=router_name).execute()
        wait_for_compute_region_operation(project_id, region, operation, compute)
        cli_logger.print("Successfully deleted the router: {}.".format(router_name))
    except Exception as e:
        cli_logger.error("Failed to delete the router: {}. {}", router_name, str(e))
        raise e


def check_firewall_exist(config, compute, firewall_name):
    if get_firewall(config, compute, firewall_name) is None:
        return False
    else:
        return True


def get_firewall(config, compute, firewall_name):
    project_id = config["provider"]["project_id"]
    firewall = None
    cli_logger.verbose("Getting the existing firewall: {}...".format(firewall_name))
    try:
        firewall = compute.firewalls().get(project=project_id, firewall=firewall_name).execute()
        cli_logger.verbose("Successfully get the firewall: {}.".format(firewall_name))
    except Exception:
        cli_logger.verbose_error("Failed to get the firewall: {}.".format(firewall_name))
    return firewall


def create_firewall(compute, project_id, firewall_body):
    cli_logger.print("Creating firewall: {}... ".format(firewall_body.get("name")))
    try:
        operation = compute.firewalls().insert(project=project_id, body=firewall_body).execute()
        wait_for_compute_global_operation(project_id, operation, compute)
        cli_logger.print("Successfully created firewall: {}.".format(firewall_body.get("name")))
    except Exception as e:
        cli_logger.error("Failed to create firewall. {}", str(e))
        raise e


def update_firewall(compute, project_id, firewall_body):
    cli_logger.print("Updating firewall: {}... ".format(firewall_body.get("name")))
    try:
        operation = compute.firewalls().update(
            project=project_id, firewall=firewall_body.get("name"), body=firewall_body).execute()
        wait_for_compute_global_operation(project_id, operation, compute)
        cli_logger.print("Successfully updated firewall: {}.".format(firewall_body.get("name")))
    except Exception as e:
        cli_logger.error("Failed to update firewall. {}", str(e))
        raise e


def create_or_update_firewall(config, compute, firewall_body):
    firewall_name = firewall_body.get("name")
    project_id = config["provider"]["project_id"]

    if not check_firewall_exist(config, compute, firewall_name):
        create_firewall(compute, project_id, firewall_body)
    else:
        cli_logger.print("The firewall {} already exists. Will update the rules... ".format(firewall_name))
        update_firewall(compute, project_id, firewall_body)


def _get_subnetwork_ip_cidr_range(project_id, compute, subnetwork):
    subnetwork_region, subnet_name = _split_subnetwork_info(project_id, subnetwork)
    return compute.subnetworks().get(
        project=project_id, region=subnetwork_region, subnetwork=subnet_name).execute().get("ipCidrRange")


def get_subnetworks_ip_cidr_range(config, compute, vpc_id):
    provider_config = config["provider"]
    project_id = provider_config["project_id"]
    subnetworks = compute.networks().get(project=project_id, network=vpc_id).execute().get("subnetworks")
    subnetwork_cidrs = []
    for subnetwork in subnetworks:
        subnetwork_cidrs.append(_get_subnetwork_ip_cidr_range(project_id, compute, subnetwork))
    return subnetwork_cidrs


def get_working_node_ip_cidr_range(config, compute):
    provider_config = config["provider"]
    project_id = provider_config["project_id"]
    subnetwork_cidrs = []
    subnetwork = _find_working_node_subnetwork(provider_config, compute)
    if subnetwork is not None:
        subnetwork_cidrs.append(_get_subnetwork_ip_cidr_range(project_id, compute, subnetwork))
    return subnetwork_cidrs


def _create_default_allow_internal_firewall(config, compute, vpc_id):
    project_id = config["provider"]["project_id"]
    workspace_name = config["workspace_name"]
    subnetwork_cidrs = get_subnetworks_ip_cidr_range(config, compute, vpc_id)
    firewall_name = "cloudtik-{}-default-allow-internal-firewall".format(workspace_name)
    firewall_body = {
        "name": firewall_name,
        "network": "projects/{}/global/networks/{}".format(project_id, vpc_id),
        "allowed": [
            {
                "IPProtocol": "tcp",
                "ports": [
                    "0-65535"
                ]
            },
            {
                "IPProtocol": "udp",
                "ports": [
                    "0-65535"
                ]
            },
            {
                "IPProtocol": "icmp"
            }
        ],
        "sourceRanges": subnetwork_cidrs
    }

    create_or_update_firewall(config, compute, firewall_body)


def _get_allow_working_node_firewall_rules(config, compute):
    firewall_rules = []
    subnetwork_cidrs = get_working_node_ip_cidr_range(config, compute)
    if len(subnetwork_cidrs) == 0:
        return firewall_rules

    firewall_rule = {
        "allowed": [
            {
                "IPProtocol": "tcp",
                "ports": [
                    "22" if is_peering_firewall_allow_ssh_only(config) else "0-65535"
                ]
            }
        ],
        "sourceRanges": subnetwork_cidrs
    }

    firewall_rules.append(firewall_rule)
    return firewall_rules


def _create_or_update_custom_firewalls(config, compute, vpc_id):
    firewall_rules = config["provider"] \
        .get("firewalls", {}) \
        .get("firewall_rules", [])

    if is_use_peering_vpc(config) and is_peering_firewall_allow_working_subnet(config):
        firewall_rules += _get_allow_working_node_firewall_rules(config, compute)

    project_id = config["provider"]["project_id"]
    workspace_name = config["workspace_name"]
    for i in range(len(firewall_rules)):
        firewall_body = {
            "name": "cloudtik-{}-custom-{}-firewall".format(workspace_name, i),
            "network": "projects/{}/global/networks/{}".format(project_id, vpc_id),
            "allowed": firewall_rules[i]["allowed"],
            "sourceRanges": firewall_rules[i]["sourceRanges"]
        }
        create_or_update_firewall(config, compute, firewall_body)


def _create_or_update_firewalls(config, compute, vpc_id):
    current_step = 1
    total_steps = 2

    with cli_logger.group(
            "Creating or updating internal firewall",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_default_allow_internal_firewall(config, compute, vpc_id)

    with cli_logger.group(
            "Creating or updating custom firewalls",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_or_update_custom_firewalls(config, compute, vpc_id)


def check_workspace_firewalls(config, compute):
    workspace_name = config["workspace_name"]
    firewall_names = ["cloudtik-{}-default-allow-internal-firewall".format(workspace_name)]

    for firewall_name in firewall_names:
        if not check_firewall_exist(config, compute, firewall_name):
            return False

    return True


def delete_firewall(compute, project_id, firewall_name):
    cli_logger.print("Deleting the firewall {}... ".format(firewall_name))
    try:
        operation = compute.firewalls().delete(project=project_id, firewall=firewall_name).execute()
        wait_for_compute_global_operation(project_id, operation, compute)
        cli_logger.print("Successfully delete the firewall {}.".format(firewall_name))
    except Exception as e:
        cli_logger.error(
            "Failed to delete the firewall {}. {}", firewall_name, str(e))
        raise e


def _delete_firewalls(config, compute):
    project_id = config["provider"]["project_id"]
    workspace_name = config["workspace_name"]
    cloudtik_firewalls = [firewall.get("name")
        for firewall in compute.firewalls().list(project=project_id).execute().get("items")
            if "cloudtik-{}".format(workspace_name) in firewall.get("name")]

    total_steps = len(cloudtik_firewalls)
    for i, cloudtik_firewall in enumerate(cloudtik_firewalls):
        with cli_logger.group(
                "Deleting firewall",
                _numbered=("()", i + 1, total_steps)):
            delete_firewall(compute, project_id, cloudtik_firewall)


def get_gcp_vpc_id(config, compute, use_working_vpc):
    return _get_gcp_vpc_id(
        config["provider"], config.get("workspace_name"), compute, use_working_vpc)


def _get_gcp_vpc_id(provider_config, workspace_name, compute, use_working_vpc):
    if use_working_vpc:
        vpc_id = _get_working_node_vpc_id(provider_config, compute)
    else:
        vpc_id = _get_workspace_vpc_id(provider_config, workspace_name, compute)
    return vpc_id


def update_gcp_workspace_firewalls(config):
    crm, iam, compute, tpu = \
        construct_clients_from_provider_config(config["provider"])

    workspace_name = config["workspace_name"]
    use_working_vpc = is_use_working_vpc(config)
    vpc_id = get_gcp_vpc_id(config, compute, use_working_vpc)
    if vpc_id is None:
        cli_logger.error("The workspace: {} doesn't exist!".format(config["workspace_name"]))
        return

    current_step = 1
    total_steps = 1

    try:

        with cli_logger.group(
                "Updating workspace firewalls",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _create_or_update_firewalls(config, compute, vpc_id)

    except Exception as e:
        cli_logger.error(
            "Failed to update the firewalls of workspace {}. {}", workspace_name, str(e))
        raise e

    cli_logger.success(
        "Successfully updated the firewalls of workspace: {}.",
        cf.bold(workspace_name))


def delete_gcp_workspace(config, delete_managed_storage: bool = False):
    crm, iam, compute, tpu = \
        construct_clients_from_provider_config(config["provider"])

    workspace_name = config["workspace_name"]
    managed_cloud_storage = is_managed_cloud_storage(config)
    use_working_vpc = is_use_working_vpc(config)
    use_peering_vpc = is_use_peering_vpc(config)
    vpc_id = get_gcp_vpc_id(config, compute, use_working_vpc)

    current_step = 1
    total_steps = GCP_WORKSPACE_NUM_DELETION_STEPS
    if vpc_id is None:
        total_steps = 1
    else:
        if use_peering_vpc:
            total_steps += 1
    if managed_cloud_storage and delete_managed_storage:
        total_steps += 1

    try:
        with cli_logger.group("Deleting workspace: {}", workspace_name):
            # Delete in a reverse way of creating
            if managed_cloud_storage and delete_managed_storage:
                with cli_logger.group(
                        "Deleting GCS bucket",
                        _numbered=("[]", current_step, total_steps)):
                    current_step += 1
                    _delete_workspace_cloud_storage(config, workspace_name)

            with cli_logger.group(
                    "Deleting service accounts",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _delete_workspace_service_accounts(config, iam)

            if vpc_id:
                _delete_network_resources(config, compute, current_step, total_steps)

    except Exception as e:
        cli_logger.error(
            "Failed to delete workspace {}. {}", workspace_name, str(e))
        raise e

    cli_logger.success(
            "Successfully deleted workspace: {}.",
            cf.bold(workspace_name))


def _delete_workspace_service_accounts(config, iam):
    current_step = 1
    total_steps = 2

    with cli_logger.group(
            "Deleting service account for head",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_head_service_account(config, iam)

    with cli_logger.group(
            "Deleting service account for worker",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_worker_service_account(config, iam)


def _delete_head_service_account(config, iam):
    workspace_name = config["workspace_name"]
    head_service_account_id = GCP_HEAD_SERVICE_ACCOUNT_ID.format(workspace_name)
    _delete_service_account(config["provider"], head_service_account_id, iam)


def _delete_worker_service_account(config, iam):
    workspace_name = config["workspace_name"]
    worker_service_account_id = GCP_WORKER_SERVICE_ACCOUNT_ID.format(workspace_name)
    _delete_service_account(config["provider"], worker_service_account_id, iam)


def _delete_service_account(cloud_provider, service_account_id, iam):
    project_id = cloud_provider["project_id"]
    email = get_service_account_email(
        account_id=service_account_id,
        project_id=project_id)
    service_account = _get_service_account(cloud_provider, email, iam)
    if service_account is None:
        cli_logger.warning("No service account with id {} found.".format(service_account_id))
        return

    try:
        cli_logger.print("Deleting service account: {}...".format(service_account_id))
        full_name = get_service_account_resource_name(project_id=project_id, account=email)
        iam.projects().serviceAccounts().delete(name=full_name).execute()
        cli_logger.print("Successfully deleted the service account.")
    except Exception as e:
        cli_logger.error("Failed to delete the service account. {}", str(e))
        raise e


def _delete_workspace_cloud_storage(config, workspace_name):
    _delete_managed_cloud_storage(config["provider"], workspace_name)


def _delete_managed_cloud_storage(cloud_provider, workspace_name):
    bucket = get_managed_gcs_bucket(cloud_provider, workspace_name)
    if bucket is None:
        cli_logger.warning("No GCS bucket with the name found.")
        return

    try:
        cli_logger.print("Deleting GCS bucket: {}...".format(bucket.name))
        bucket.delete(force=True)
        cli_logger.print("Successfully deleted GCS bucket.")
    except Exception as e:
        cli_logger.error("Failed to delete GCS bucket. {}", str(e))
        raise e


def _delete_network_resources(config, compute, current_step, total_steps):
    """
         Do the work - order of operation:
         Delete VPC peering connection if needed
         Delete public subnet
         Delete router for private subnet
         Delete private subnets
         Delete firewalls
         Delete vpc
    """
    use_peering_vpc = is_use_peering_vpc(config)

    # delete vpc peering connection
    if use_peering_vpc:
        with cli_logger.group(
                "Deleting VPC peering connection",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _delete_vpc_peering_connections(config, compute)

    # delete public subnets
    with cli_logger.group(
            "Deleting public subnet",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_subnet(config, compute, is_private=False)

    # delete router for private subnets
    with cli_logger.group(
            "Deleting router",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_router(config, compute)

    # delete private subnets
    with cli_logger.group(
            "Deleting private subnet",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_subnet(config, compute, is_private=True)

    # delete firewalls
    with cli_logger.group(
            "Deleting firewall rules",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_firewalls(config, compute)

    # delete vpc
    with cli_logger.group(
            "Deleting VPC",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_vpc(config, compute)


def _create_vpc(config, compute):
    workspace_name = config["workspace_name"]
    use_working_vpc = is_use_working_vpc(config)
    if use_working_vpc:
        # No need to create new vpc
        vpc_id = get_working_node_vpc_id(config, compute)
        if vpc_id is None:
            cli_logger.abort("Failed to get the VPC for the current machine. "
                             "Please make sure your current machine is an GCP virtual machine "
                             "to use use_internal_ips=True with use_working_vpc=True.")
    else:
        # Need to create a new vpc
        if get_workspace_vpc_id(config, compute) is None:
            create_vpc(config, compute)
            vpc_id = get_workspace_vpc_id(config, compute)
        else:
            cli_logger.abort("There is a existing VPC with the same name: {}, "
                             "if you want to create a new workspace with the same name, "
                             "you need to execute workspace delete first!".format(workspace_name))
    return vpc_id


def _create_head_service_account(config, crm, iam):
    workspace_name = config["workspace_name"]
    service_account_id = GCP_HEAD_SERVICE_ACCOUNT_ID.format(workspace_name)
    cli_logger.print("Creating head service account: {}...".format(service_account_id))

    try:
        service_account_config = {
            "displayName": GCP_HEAD_SERVICE_ACCOUNT_DISPLAY_NAME.format(workspace_name),
        }

        service_account = _create_service_account(
            config["provider"], service_account_id, service_account_config,
            iam)

        assert service_account is not None, "Failed to create head service account."

        if config["provider"].get(HAS_TPU_PROVIDER_FIELD, False):
            roles = HEAD_SERVICE_ACCOUNT_ROLES + TPU_SERVICE_ACCOUNT_ROLES
        else:
            roles = HEAD_SERVICE_ACCOUNT_ROLES

        _add_iam_role_binding_for_service_account(service_account, roles, crm)
        cli_logger.print("Successfully created head service account and configured with roles.")
    except Exception as e:
        cli_logger.error("Failed to create head service account. {}", str(e))
        raise e


def _create_worker_service_account(config, crm, iam):
    workspace_name = config["workspace_name"]
    service_account_id = GCP_WORKER_SERVICE_ACCOUNT_ID.format(workspace_name)
    cli_logger.print("Creating worker service account: {}...".format(service_account_id))

    try:
        service_account_config = {
            "displayName": GCP_WORKER_SERVICE_ACCOUNT_DISPLAY_NAME.format(workspace_name),
        }
        service_account = _create_service_account(
            config["provider"], service_account_id, service_account_config,
            iam)

        assert service_account is not None, "Failed to create worker service account."

        _add_iam_role_binding_for_service_account(service_account, WORKER_SERVICE_ACCOUNT_ROLES, crm)
        cli_logger.print("Successfully created worker service account and configured with roles.")
    except Exception as e:
        cli_logger.error("Failed to create worker service account. {}", str(e))
        raise e


def _create_workspace_service_accounts(config, crm, iam):
    current_step = 1
    total_steps = 2

    with cli_logger.group(
            "Creating service account for head",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_head_service_account(config, crm, iam)

    with cli_logger.group(
            "Creating service account for worker",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_worker_service_account(config, crm, iam)

    return config


def _create_workspace_cloud_storage(config):
    _create_managed_cloud_storage(config["provider"], config["workspace_name"])
    return config


def _create_managed_cloud_storage(cloud_provider, workspace_name):
    # If the managed cloud storage for the workspace already exists
    # Skip the creation step
    bucket = get_managed_gcs_bucket(cloud_provider, workspace_name)
    if bucket is not None:
        cli_logger.print("GCS bucket for the workspace already exists. Skip creation.")
        return

    region = cloud_provider["region"]
    storage_client = construct_storage_client(cloud_provider)
    suffix = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))
    bucket_name = "cloudtik-{workspace_name}-{region}-{suffix}".format(
        workspace_name=workspace_name,
        region=region,
        suffix=suffix
    )

    cli_logger.print("Creating GCS bucket for the workspace: {}".format(workspace_name))
    try:
        storage_client.create_bucket(bucket_or_name=bucket_name, location=region)
        cli_logger.print("Successfully created GCS bucket: {}.".format(bucket_name))
    except Exception as e:
        cli_logger.error("Failed to create GCS bucket. {}", str(e))
        raise e


def _create_network_resources(config, current_step, total_steps):
    crm, iam, compute, tpu = \
        construct_clients_from_provider_config(config["provider"])

    # create vpc
    with cli_logger.group(
            "Creating VPC",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        vpc_id = _create_vpc(config, compute)

    # create subnets
    with cli_logger.group(
            "Creating subnets",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_and_configure_subnets(config, compute, vpc_id)

    # create router
    with cli_logger.group(
            "Creating router",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_router(config, compute, vpc_id)

    # create nat-gateway for router
    with cli_logger.group(
            "Creating NAT for router",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_nat_for_router(config, compute)

    # create firewalls
    with cli_logger.group(
            "Creating firewall rules",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_or_update_firewalls(config, compute, vpc_id)

    if is_use_peering_vpc(config):
        with cli_logger.group(
                "Creating VPC peering connection",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _create_vpc_peering_connections(config, compute, vpc_id)

    return current_step


def check_gcp_workspace_existence(config):
    crm, iam, compute, tpu = \
        construct_clients_from_provider_config(config["provider"])
    workspace_name = config["workspace_name"]
    managed_cloud_storage = is_managed_cloud_storage(config)
    use_working_vpc = is_use_working_vpc(config)
    use_peering_vpc = is_use_peering_vpc(config)

    existing_resources = 0
    target_resources = GCP_WORKSPACE_TARGET_RESOURCES
    if managed_cloud_storage:
        target_resources += 1
    if use_peering_vpc:
        target_resources += 1

    """
         Do the work - order of operation
         Check project
         Check VPC
         Check private subnet
         Check public subnet
         Check router
         Check firewalls
         Check VPC peering if needed
         Check GCS bucket
         Check service accounts
    """
    project_existence = False
    cloud_storage_existence = False
    if get_workspace_project(config, crm) is not None:
        existing_resources += 1
        project_existence = True
        # All resources that depending on project
        vpc_id = get_gcp_vpc_id(config, compute, use_working_vpc)
        if vpc_id is not None:
            existing_resources += 1
            # Network resources that depending on VPC
            if get_subnet(config, "cloudtik-{}-private-subnet".format(workspace_name), compute) is not None:
                existing_resources += 1
            if get_subnet(config, "cloudtik-{}-public-subnet".format(workspace_name), compute) is not None:
                existing_resources += 1
            if get_router(config, "cloudtik-{}-private-router".format(workspace_name), compute) is not None:
                existing_resources += 1
            if check_workspace_firewalls(config, compute):
                existing_resources += 1
            if use_peering_vpc:
                peerings = get_workspace_vpc_peering_connections(config, compute, vpc_id)
                if len(peerings) == 2:
                    existing_resources += 1

        if managed_cloud_storage:
            if get_workspace_gcs_bucket(config, workspace_name) is not None:
                existing_resources += 1
                cloud_storage_existence = True

        if _get_workspace_service_account(config, iam, GCP_HEAD_SERVICE_ACCOUNT_ID) is not None:
            existing_resources += 1
        if _get_workspace_service_account(config, iam, GCP_WORKER_SERVICE_ACCOUNT_ID) is not None:
            existing_resources += 1

    if existing_resources == 0 or (
            existing_resources == 1 and project_existence):
        return Existence.NOT_EXIST
    elif existing_resources == target_resources:
        return Existence.COMPLETED
    else:
        if existing_resources == 2 and cloud_storage_existence:
            return Existence.STORAGE_ONLY
        return Existence.IN_COMPLETED


def check_gcp_workspace_integrity(config):
    existence = check_gcp_workspace_existence(config)
    return True if existence == Existence.COMPLETED else False


def get_gcp_workspace_info(config):
    info = {}
    get_gcp_managed_cloud_storage_info(config, config["provider"], info)
    return info


def get_gcp_managed_cloud_storage_info(config, cloud_provider, info):
    workspace_name = config["workspace_name"]
    bucket = get_managed_gcs_bucket(cloud_provider, workspace_name)
    managed_bucket_name = None if bucket is None else bucket.name
    if managed_bucket_name is not None:
        gcp_cloud_storage = {GCP_GCS_BUCKET: managed_bucket_name}
        managed_cloud_storage = {GCP_MANAGED_STORAGE_GCS_BUCKET: managed_bucket_name,
                                 CLOUDTIK_MANAGED_CLOUD_STORAGE_URI: get_gcp_cloud_storage_uri(gcp_cloud_storage)}
        info[CLOUDTIK_MANAGED_CLOUD_STORAGE] = managed_cloud_storage


def _fix_disk_type_for_disk(zone, disk):
    # fix disk type for all disks
    initialize_params = disk.get("initializeParams")
    if initialize_params is None:
        return

    disk_type = initialize_params.get("diskType")
    if disk_type is None or "diskTypes" in disk_type:
        return

    # Fix to format: zones/zone/diskTypes/diskType
    fix_disk_type = "zones/{}/diskTypes/{}".format(zone, disk_type)
    initialize_params["diskType"] = fix_disk_type


def _fix_disk_info_for_disk(zone, disk, boot, source_image):
    if boot:
        # Need to fix source image for only boot disk
        if "initializeParams" not in disk:
            disk["initializeParams"] = {"sourceImage": source_image}
        else:
            disk["initializeParams"]["sourceImage"] = source_image

    _fix_disk_type_for_disk(zone, disk)


def _fix_disk_info_for_node(node_config, zone):
    source_image = node_config.get("sourceImage", None)
    disks = node_config.get("disks", [])
    for disk in disks:
        boot = disk.get("boot", False)
        _fix_disk_info_for_disk(zone, disk, boot, source_image)

    # Remove the sourceImage from node config
    node_config.pop("sourceImage", None)


def _fix_disk_info(config):
    zone = config["provider"]["availability_zone"]
    for node_type in config["available_node_types"].values():
        node_config = node_type["node_config"]
        _fix_disk_info_for_node(node_config, zone)

    return config


def _configure_spot_for_node_type(node_type_config,
                                  prefer_spot_node):
    # To be improved if scheduling has other configurations
    # scheduling:
    #   - preemptible: true
    node_config = node_type_config["node_config"]
    if prefer_spot_node:
        # Add spot instruction
        node_config.pop("scheduling", None)
        node_config["scheduling"] = [{"preemptible": True}]
    else:
        # Remove spot instruction
        node_config.pop("scheduling", None)


def _configure_prefer_spot_node(config):
    prefer_spot_node = config["provider"].get("prefer_spot_node")

    # if no such key, we consider user don't want to override
    if prefer_spot_node is None:
        return config

    # User override, set or remove spot settings for worker node types
    node_types = config["available_node_types"]
    for node_type_name in node_types:
        if node_type_name == config["head_node_type"]:
            continue

        # worker node type
        node_type_data = node_types[node_type_name]
        _configure_spot_for_node_type(
            node_type_data, prefer_spot_node)

    return config


def bootstrap_gcp(config):
    workspace_name = config.get("workspace_name", "")
    if workspace_name == "":
        raise RuntimeError("Workspace name is not specified in cluster configuration.")

    config = bootstrap_gcp_from_workspace(config)
    return config


def bootstrap_gcp_from_workspace(config):
    if not check_gcp_workspace_integrity(config):
        workspace_name = config["workspace_name"]
        cli_logger.abort("GCP workspace {} doesn't exist or is in wrong state!", workspace_name)

    config = copy.deepcopy(config)

    # Used internally to store head IAM role.
    config["head_node"] = {}

    # Check if we have any TPUs defined, and if so,
    # insert that information into the provider config
    if _has_tpus_in_node_configs(config):
        config["provider"][HAS_TPU_PROVIDER_FIELD] = True

        # We can't run autoscaling through a serviceAccount on TPUs (atm)
        if _is_head_node_a_tpu(config):
            raise RuntimeError("TPUs are not supported as head nodes.")

    crm, iam, compute, tpu = \
        construct_clients_from_provider_config(config["provider"])

    config = _fix_disk_info(config)
    config = _configure_iam_role_from_workspace(config, iam)
    config = _configure_cloud_storage_from_workspace(config)
    config = _configure_key_pair(config, compute)
    config = _configure_subnet_from_workspace(config, compute)
    config = _configure_prefer_spot_node(config)
    return config


def bootstrap_gcp_workspace(config):
    # create a copy of the input config to modify
    config = copy.deepcopy(config)
    _configure_allowed_ssh_sources(config)
    return config


def _configure_project_id(config):
    project_id = config["provider"].get("project_id")
    if project_id is None and "workspace_name" in config:
        config["provider"]["project_id"] = config["workspace_name"]
    return config


def _configure_allowed_ssh_sources(config):
    provider_config = config["provider"]
    if "allowed_ssh_sources" not in provider_config:
        return

    allowed_ssh_sources = provider_config["allowed_ssh_sources"]
    if len(allowed_ssh_sources) == 0:
        return

    if "firewalls" not in provider_config:
        provider_config["firewalls"] = {}
    fire_walls = provider_config["firewalls"]

    if "firewall_rules" not in fire_walls:
        fire_walls["firewall_rules"] = []
    firewall_rules = fire_walls["firewall_rules"]

    firewall_rule = {
        "allowed": [
            {
              "IPProtocol": "tcp",
              "ports": [
                "22"
              ]
            }
        ],
        "sourceRanges": [allowed_ssh_source for allowed_ssh_source in allowed_ssh_sources]
    }
    firewall_rules.append(firewall_rule)


def _configure_project(config, crm):
    """Setup a Google Cloud Platform Project.

    Google Compute Platform organizes all the resources, such as storage
    buckets, users, and instances under projects. This is different from
    aws ec2 where everything is global.
    """
    config = copy.deepcopy(config)

    project_id = config["provider"].get("project_id")
    assert config["provider"]["project_id"] is not None, (
        "'project_id' must be set in the 'provider' section of the"
        " config. Notice that the project id must be globally unique.")
    project = _get_project(project_id, crm)

    if project is None:
        #  Project not found, try creating it
        _create_project(project_id, crm)
        project = _get_project(project_id, crm)
    else:
        cli_logger.print("Using the existing project: {}.".format(project_id))

    assert project is not None, "Failed to create project"
    assert project["lifecycleState"] == "ACTIVE", (
        "Project status needs to be ACTIVE, got {}".format(
            project["lifecycleState"]))

    config["provider"]["project_id"] = project["projectId"]

    return config


def _configure_cloud_storage_from_workspace(config):
    use_managed_cloud_storage = is_use_managed_cloud_storage(config)
    if use_managed_cloud_storage:
        _configure_managed_cloud_storage_from_workspace(config, config["provider"])

    return config


def _configure_managed_cloud_storage_from_workspace(config, cloud_provider):
    workspace_name = config["workspace_name"]
    gcs_bucket = get_managed_gcs_bucket(cloud_provider, workspace_name)
    if gcs_bucket is None:
        cli_logger.abort("No managed GCS bucket was found. If you want to use managed GCS bucket, "
                         "you should set managed_cloud_storage equal to True when you creating workspace.")

    cloud_storage = get_gcp_cloud_storage_config_for_update(config["provider"])
    cloud_storage[GCP_GCS_BUCKET] = gcs_bucket.name


def _get_workspace_service_account(config, iam, service_account_id_template):
    workspace_name = config["workspace_name"]
    service_account_id = service_account_id_template.format(workspace_name)
    email = get_service_account_email(
        account_id=service_account_id,
        project_id=config["provider"]["project_id"])
    service_account = _get_service_account(config["provider"], email, iam)
    return service_account


def _configure_iam_role_for_head(config, iam):
    head_service_account = _get_workspace_service_account(
        config, iam, GCP_HEAD_SERVICE_ACCOUNT_ID)
    if head_service_account is None:
        cli_logger.abort("Head service account not found for workspace.")

    head_service_accounts = [{
        "email": head_service_account["email"],
        # NOTE: The amount of access is determined by the scope + IAM
        # role of the service account. Even if the cloud-platform scope
        # gives (scope) access to the whole cloud-platform, the service
        # account is limited by the IAM rights specified below.
        "scopes": ["https://www.googleapis.com/auth/cloud-platform"]
    }]
    config["head_node"]["serviceAccounts"] = head_service_accounts


def _configure_iam_role_for_worker(config, iam):
    # worker service account
    worker_service_account = _get_workspace_service_account(
        config, iam, GCP_WORKER_SERVICE_ACCOUNT_ID)
    if worker_service_account is None:
        cli_logger.abort("Worker service account not found for workspace.")

    worker_service_accounts = [{
        "email": worker_service_account["email"],
        "scopes": ["https://www.googleapis.com/auth/cloud-platform"]
    }]

    for key, node_type in config["available_node_types"].items():
        if key == config["head_node_type"]:
            continue
        node_config = node_type["node_config"]
        node_config["serviceAccounts"] = worker_service_accounts


def _configure_iam_role_from_workspace(config, iam):
    config = copy.deepcopy(config)
    _configure_iam_role_for_head(config, iam)

    worker_role_for_cloud_storage = is_worker_role_for_cloud_storage(config)
    if worker_role_for_cloud_storage:
        _configure_iam_role_for_worker(config, iam)

    return config


def _configure_key_pair(config, compute):
    """Configure SSH access, using an existing key pair if possible.

    Creates a project-wide ssh key that can be used to access all the instances
    unless explicitly prohibited by instance config.

    The ssh-keys created are of format:

      [USERNAME]:ssh-rsa [KEY_VALUE] [USERNAME]

    where:

      [USERNAME] is the user for the SSH key, specified in the config.
      [KEY_VALUE] is the public SSH key value.
    """
    config = copy.deepcopy(config)

    if "ssh_private_key" in config["auth"]:
        return config

    ssh_user = config["auth"]["ssh_user"]

    project = compute.projects().get(
        project=config["provider"]["project_id"]).execute()

    # Key pairs associated with project meta data. The key pairs are general,
    # and not just ssh keys.
    ssh_keys_str = next(
        (item for item in project["commonInstanceMetadata"].get("items", [])
         if item["key"] == "ssh-keys"), {}).get("value", "")

    ssh_keys = ssh_keys_str.split("\n") if ssh_keys_str else []

    # Try a few times to get or create a good key pair.
    key_found = False
    for i in range(10):
        key_name = key_pair_name(i, config["provider"]["region"],
                                 config["provider"]["project_id"], ssh_user)
        public_key_path, private_key_path = key_pair_paths(key_name)

        for ssh_key in ssh_keys:
            key_parts = ssh_key.split(" ")
            if len(key_parts) != 3:
                continue

            if key_parts[2] == ssh_user and os.path.exists(private_key_path):
                # Found a key
                key_found = True
                break

        # Writing the new ssh key to the filesystem fails if the ~/.ssh
        # directory doesn't already exist.
        os.makedirs(os.path.expanduser("~/.ssh"), exist_ok=True)

        # Create a key since it doesn't exist locally or in GCP
        if not key_found and not os.path.exists(private_key_path):
            cli_logger.print("Creating new key pair: {}".format(key_name))
            public_key, private_key = generate_rsa_key_pair()

            _create_project_ssh_key_pair(project, public_key, ssh_user,
                                         compute)

            # Create the directory if it doesn't exists
            private_key_dir = os.path.dirname(private_key_path)
            os.makedirs(private_key_dir, exist_ok=True)

            # We need to make sure to _create_ the file with the right
            # permissions. In order to do that we need to change the default
            # os.open behavior to include the mode we want.
            with open(
                    private_key_path,
                    "w",
                    opener=partial(os.open, mode=0o600),
            ) as f:
                f.write(private_key)

            with open(public_key_path, "w") as f:
                f.write(public_key)

            key_found = True

            break

        if key_found:
            break

    assert key_found, "SSH keypair for user {} not found for {}".format(
        ssh_user, private_key_path)
    assert os.path.exists(private_key_path), (
        "Private key file {} not found for user {}"
        "".format(private_key_path, ssh_user))

    cli_logger.print("Private key not specified in config, using "
                     "{}".format(private_key_path))

    config["auth"]["ssh_private_key"] = private_key_path

    return config


def _configure_subnet_from_workspace(config, compute):
    workspace_name = config["workspace_name"]
    use_internal_ips = is_use_internal_ip(config)

    """Pick a reasonable subnet if not specified by the config."""
    config = copy.deepcopy(config)

    # Rationale: avoid subnet lookup if the network is already
    # completely manually configured

    # networkInterfaces is compute, networkConfig is TPU
    public_subnet = get_subnet(config, "cloudtik-{}-public-subnet".format(workspace_name), compute)
    private_subnet = get_subnet(config, "cloudtik-{}-private-subnet".format(workspace_name), compute)

    public_interfaces = [{
        "subnetwork": public_subnet["selfLink"],
        "accessConfigs": [{
            "name": "External NAT",
            "type": "ONE_TO_ONE_NAT",
        }],
    }]

    private_interfaces = [{
        "subnetwork": private_subnet["selfLink"],
    }]

    for key, node_type in config["available_node_types"].items():
        node_config = node_type["node_config"]
        if key == config["head_node_type"]:
            if use_internal_ips:
                # compute
                node_config["networkInterfaces"] = copy.deepcopy(private_interfaces)
                # TPU
                node_config["networkConfig"] = copy.deepcopy(private_interfaces)[0]
            else:
                # compute
                node_config["networkInterfaces"] = copy.deepcopy(public_interfaces)
                # TPU
                node_config["networkConfig"] = copy.deepcopy(public_interfaces)[0]
                node_config["networkConfig"].pop("accessConfigs")
        else:
            # compute
            node_config["networkInterfaces"] = copy.deepcopy(private_interfaces)
            # TPU
            node_config["networkConfig"] = copy.deepcopy(private_interfaces)[0]

    return config


def get_subnet(config, subnet_name, compute):
    cli_logger.verbose("Getting the existing subnet: {}.".format(subnet_name))
    try:
        subnet = compute.subnetworks().get(
            project=config["provider"]["project_id"],
            region=config["provider"]["region"],
            subnetwork=subnet_name,
        ).execute()
        cli_logger.verbose("Successfully get the subnet: {}.".format(subnet_name))
        return subnet
    except Exception:
        cli_logger.verbose_error("Failed to get the subnet: {}.".format(subnet_name))
        return None


def get_router(config, router_name, compute):
    cli_logger.verbose("Getting the existing router: {}.".format(router_name))
    try:
        router = compute.routers().get(
            project=config["provider"]["project_id"],
            region=config["provider"]["region"],
            router=router_name,
        ).execute()
        cli_logger.verbose("Successfully get the router: {}.".format(router_name))
        return router
    except Exception:
        cli_logger.verbose_error("Failed to get the router: {}.".format(router_name))
        return None


def _get_project(project_id, crm):
    try:
        project = crm.projects().get(projectId=project_id).execute()
    except errors.HttpError as e:
        if e.resp.status != 403:
            raise
        project = None

    return project


def get_workspace_project(config, crm):
    project_id = config["provider"]["project_id"]
    return _get_project(project_id, crm)


def get_workspace_gcs_bucket(config, workspace_name):
    return get_managed_gcs_bucket(config["provider"], workspace_name)


def get_managed_gcs_bucket(cloud_provider, workspace_name):
    gcs = construct_storage_client(cloud_provider)
    region = cloud_provider["region"]
    project_id = cloud_provider["project_id"]
    bucket_name_prefix = "cloudtik-{workspace_name}-{region}-".format(
        workspace_name=workspace_name,
        region=region
    )

    cli_logger.verbose("Getting GCS bucket with prefix: {}.".format(bucket_name_prefix))
    for bucket in gcs.list_buckets(project=project_id):
        if bucket_name_prefix in bucket.name:
            cli_logger.verbose("Successfully get the GCS bucket: {}.".format(bucket.name))
            return bucket

    cli_logger.verbose_error("Failed to get the GCS bucket for workspace.")
    return None


def _create_project(project_id, crm):
    cli_logger.print("Creating project: {}...".format(project_id))
    operation = crm.projects().create(body={
        "projectId": project_id,
        "name": project_id
    }).execute()

    result = wait_for_crm_operation(operation, crm)
    if "done" in result and result["done"]:
        cli_logger.print("Successfully created project: {}.".format(project_id))

    return result


def _get_service_account_by_id(cloud_provider, account_id, iam):
    email = get_service_account_email(
        account_id=account_id,
        project_id=cloud_provider["project_id"])
    return _get_service_account(cloud_provider, email, iam)


def _get_service_account(cloud_provider, account, iam):
    project_id = cloud_provider["project_id"]
    return _get_service_account_of_project(project_id, account, iam)


def _get_service_account_of_project(project_id, account, iam):
    full_name = get_service_account_resource_name(project_id=project_id, account=account)
    try:
        cli_logger.verbose("Getting the service account: {}...".format(account))
        service_account = iam.projects().serviceAccounts().get(
            name=full_name).execute()
        cli_logger.verbose("Successfully get the service account: {}.".format(account))
    except errors.HttpError as e:
        if e.resp.status != 404:
            raise
        cli_logger.verbose("The service account doesn't exist: {}...".format(account))
        service_account = None

    return service_account


def _create_service_account(cloud_provider, account_id, account_config, iam):
    project_id = cloud_provider["project_id"]
    service_account = iam.projects().serviceAccounts().create(
        name="projects/{project_id}".format(project_id=project_id),
        body={
            "accountId": account_id,
            "serviceAccount": account_config,
        }).execute()

    return service_account


def _add_iam_role_binding_for_service_account(service_account, roles, crm):
    project_id = service_account["projectId"]
    service_account_email = service_account["email"]
    return _add_iam_role_binding(
        project_id, service_account_email, roles, crm)


def _add_iam_role_binding(project_id, service_account_email, roles, crm):
    """Add new IAM roles for the service account."""
    member_id = "serviceAccount:" + service_account_email
    policy = crm.projects().getIamPolicy(
        resource=project_id, body={}).execute()

    changed = _add_role_bindings_to_policy(roles, member_id, policy)
    if not changed:
        # In some managed environments, an admin needs to grant the
        # roles, so only call setIamPolicy if needed.
        return

    result = crm.projects().setIamPolicy(
        resource=project_id, body={
            "policy": policy,
        }).execute()

    return result


def _remove_iam_role_binding(project_id, service_account_email, roles, crm):
    """Remove new IAM roles for the service account."""
    member_id = "serviceAccount:" + service_account_email
    policy = crm.projects().getIamPolicy(
        resource=project_id, body={}).execute()

    changed = _remove_role_bindings_from_policy(roles, member_id, policy)
    if not changed:
        return

    result = crm.projects().setIamPolicy(
        resource=project_id, body={
            "policy": policy,
        }).execute()
    return result


def _has_iam_role_binding(project_id, service_account_email, roles, crm):
    role_bindings = _get_iam_role_binding(
        project_id, service_account_email, roles, crm)
    if len(role_bindings) != len(roles):
        return False
    return True


def _get_iam_role_binding(project_id, service_account_email, roles, crm):
    """Get IAM roles bindings for the service account."""
    member_id = "serviceAccount:" + service_account_email
    policy = crm.projects().getIamPolicy(
        resource=project_id, body={}).execute()
    return _get_role_bindings_of_policy(roles, member_id, policy)


def get_service_account_resource_name(project_id, account):
    # 'account' can be the account id or the email
    return "projects/{project_id}/serviceAccounts/{account}".format(
           project_id=project_id, account=account)


def _add_role_bindings_to_policy(roles, member_id, policy):
    changed = False
    if "bindings" not in policy:
        bindings = []
        for role in roles:
            bindings.append({
                "members": [member_id],
                "role": role,
            })
        policy["bindings"] = bindings
        changed = True

    for role in roles:
        role_exists = False
        for binding in policy["bindings"]:
            if binding["role"] == role:
                if "members" not in binding:
                    binding["members"] = [member_id]
                    changed = True
                elif member_id not in binding["members"]:
                    binding["members"].append(member_id)
                    changed = True
                role_exists = True

        if not role_exists:
            changed = True
            policy["bindings"].append({
                "members": [member_id],
                "role": role,
            })
    return changed


def _remove_role_bindings_from_policy(roles, member_id, policy):
    changed = False
    if "bindings" not in policy:
        return changed
    for role in roles:
        for binding in policy["bindings"]:
            if binding["role"] == role:
                if "members" in binding and member_id in binding["members"]:
                    binding["members"].remove(member_id)
                    changed = True
    return changed


def _get_role_bindings_of_policy(roles, member_id, policy):
    role_bindings = []
    if "bindings" not in policy:
        return role_bindings

    for role in roles:
        for binding in policy["bindings"]:
            if binding["role"] == role:
                if "members" in binding and member_id in binding["members"]:
                    role_bindings.append({"role": role, "member": member_id})

    return role_bindings


def _check_service_account_existence(project_id, service_account_email, iam):
    sa = _get_service_account_of_project(project_id, service_account_email, iam)
    if sa is None:
        raise RuntimeError(
            "No service account found in project {}: {}".format(project_id, service_account_email))


def _add_service_account_iam_role_binding(
        project_id, service_account_email, roles, member_id, iam):
    """Add new IAM roles for the service account."""
    _check_service_account_existence(project_id, service_account_email, iam)
    resource = get_service_account_resource_name(project_id, service_account_email)
    policy = iam.projects().serviceAccounts().getIamPolicy(
        resource=resource).execute()

    changed = _add_role_bindings_to_policy(roles, member_id, policy)
    if not changed:
        # In some managed environments, an admin needs to grant the
        # roles, so only call setIamPolicy if needed.
        return

    result = iam.projects().serviceAccounts().setIamPolicy(
        resource=resource, body={
            "policy": policy,
        }).execute()

    return result


def _remove_service_account_iam_role_binding(
        project_id, service_account_email, roles, member_id, iam):
    """Remove new IAM roles for the service account."""
    _check_service_account_existence(project_id, service_account_email, iam)
    resource = get_service_account_resource_name(project_id, service_account_email)
    policy = iam.projects().serviceAccounts().getIamPolicy(
        resource=resource).execute()

    changed = _remove_role_bindings_from_policy(roles, member_id, policy)
    if not changed:
        return

    result = iam.projects().serviceAccounts().setIamPolicy(
        resource=resource, body={
            "policy": policy,
        }).execute()
    return result


def _has_service_account_iam_role_binding(
        project_id, service_account_email, roles, member_id, iam):
    sa = _get_service_account_of_project(project_id, service_account_email, iam)
    if sa is None:
        return False
    role_bindings = _get_service_account_iam_role_binding(
        project_id, service_account_email, roles, member_id, iam)
    if len(role_bindings) != len(roles):
        return False
    return True


def _get_service_account_iam_role_binding(
        project_id, service_account_email, roles, member_id, iam):
    """Get IAM roles bindings for the service account."""
    _check_service_account_existence(project_id, service_account_email, iam)
    resource = get_service_account_resource_name(project_id, service_account_email)
    policy = iam.projects().serviceAccounts().getIamPolicy(
        resource=resource).execute()
    return _get_role_bindings_of_policy(roles, member_id, policy)


def _create_project_ssh_key_pair(project, public_key, ssh_user, compute):
    """Inserts an ssh-key into project commonInstanceMetadata"""

    key_parts = public_key.split(" ")

    # Sanity checks to make sure that the generated key matches expectation
    assert len(key_parts) == 2, key_parts
    assert key_parts[0] == "ssh-rsa", key_parts

    new_ssh_meta = "{ssh_user}:ssh-rsa {key_value} {ssh_user}".format(
        ssh_user=ssh_user, key_value=key_parts[1])

    common_instance_metadata = project["commonInstanceMetadata"]
    items = common_instance_metadata.get("items", [])

    ssh_keys_i = next(
        (i for i, item in enumerate(items) if item["key"] == "ssh-keys"), None)

    if ssh_keys_i is None:
        items.append({"key": "ssh-keys", "value": new_ssh_meta})
    else:
        ssh_keys = items[ssh_keys_i]
        ssh_keys["value"] += "\n" + new_ssh_meta
        items[ssh_keys_i] = ssh_keys

    common_instance_metadata["items"] = items

    operation = compute.projects().setCommonInstanceMetadata(
        project=project["name"], body=common_instance_metadata).execute()

    wait_for_compute_global_operation(project["name"], operation,
                                                 compute)

    return 


def verify_gcs_storage(provider_config: Dict[str, Any]):
    gcs_storage = get_gcp_cloud_storage_config(provider_config)
    if gcs_storage is None:
        return

    try:
        use_managed_cloud_storage = _is_use_managed_cloud_storage(provider_config)
        if use_managed_cloud_storage:
            storage_gcs = construct_storage(provider_config)
        else:
            private_key_id = gcs_storage.get("gcs.service.account.private.key.id")
            if private_key_id is None:
                # The bucket may be able to accessible from roles
                # Verify through the client credential
                storage_gcs = construct_storage(provider_config)
            else:
                private_key = gcs_storage.get("gcs.service.account.private.key")
                private_key = unescape_private_key(private_key)

                credentials_field = {
                    "project_id": provider_config.get("project_id"),
                    "private_key_id": private_key_id,
                    "private_key": private_key,
                    "client_email": gcs_storage.get("gcs.service.account.client.email"),
                    "token_uri": "https://oauth2.googleapis.com/token"
                }

                credentials = service_account.Credentials.from_service_account_info(
                    credentials_field)
                storage_gcs = _create_storage(credentials)

        storage_gcs.buckets().get(bucket=gcs_storage[GCP_GCS_BUCKET]).execute()
    except Exception as e:
        raise StorageTestingError("Error happens when verifying GCS storage configurations. "
                                  "If you want to go without passing the verification, "
                                  "set 'verify_cloud_storage' to False under provider config. "
                                  "Error: {}.".format(str(e))) from None


def get_cluster_name_from_head(head_node) -> Optional[str]:
    for key, value in head_node.get("labels", {}).items():
        if key == CLOUDTIK_TAG_CLUSTER_NAME:
            return value
    return None


def list_gcp_clusters(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    _, _, compute, tpu = \
        construct_clients_from_provider_config(config["provider"])
    head_nodes = _get_workspace_head_nodes(
        config["provider"], config["workspace_name"], compute=compute)

    clusters = {}
    for head_node in head_nodes:
        cluster_name = get_cluster_name_from_head(head_node)
        if cluster_name:
            gcp_resource = GCPCompute(
                compute, config["provider"]["project_id"],
                config["provider"]["availability_zone"], cluster_name)
            gcp_node = gcp_resource.from_instance(head_node)
            clusters[cluster_name] = _get_node_info(gcp_node)
    return clusters


def with_gcp_environment_variables(provider_config, node_type_config: Dict[str, Any], node_id: str):
    config_dict = {}
    export_gcp_cloud_storage_config(provider_config, config_dict)

    if "GCP_PROJECT_ID" not in config_dict:
        project_id = provider_config.get("project_id")
        if project_id:
            config_dict["GCP_PROJECT_ID"] = project_id

    return config_dict


def _create_vpc_peering_connections(config, compute, vpc_id):
    working_vpc_id = get_working_node_vpc_id(config, compute)
    if working_vpc_id is None:
        cli_logger.abort("Failed to get the VPC for the current machine. "
                         "Please make sure your current machine is an AWS virtual machine "
                         "to use use_internal_ips=True with use_working_vpc=True.")

    current_step = 1
    total_steps = 2

    with cli_logger.group(
            "Creating workspace VPC peering connection",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_workspace_vpc_peering_connection(config, compute, vpc_id, working_vpc_id)

    with cli_logger.group(
            "Creating working VPC peering connection",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_working_vpc_peering_connection(config, compute, vpc_id, working_vpc_id)


def _create_vpc_peering_connection(config, compute, vpc_id, peer_name, peering_vpc_name):
    provider_config = config["provider"]
    project_id = provider_config.get("project_id")
    cli_logger.print("Creating VPC peering connection: {}...".format(peer_name))
    peer_network = "projects/{}/global/networks/{}".format(project_id, peering_vpc_name)
    try:
        # Creating the VPC peering
        networks_add_peering_request_body = {
            "networkPeering": {
                "name": peer_name,
                "network": peer_network,
                "exchangeSubnetRoutes": True,
            }
        }

        operation = compute.networks().addPeering(
            project=project_id, network=vpc_id, body=networks_add_peering_request_body).execute()
        wait_for_compute_global_operation(project_id, operation, compute)

        cli_logger.print(
            "Successfully created VPC peering connection: {}.".format(peer_name))
    except Exception as e:
        cli_logger.error("Failed to create VPC peering connection. {}", str(e))
        raise e


def _delete_vpc_peering_connection(config, compute, vpc_id, peer_name):
    provider_config = config["provider"]
    project_id = provider_config.get("project_id")
    cli_logger.print("Deleting VPC peering connection: {}".format(peer_name))
    try:
        networks_remove_peering_request_body = {
            "name": peer_name
        }
        operation = compute.networks().removePeering(
            project=project_id, network=vpc_id, body=networks_remove_peering_request_body ).execute()
        wait_for_compute_global_operation(project_id, operation, compute)

        cli_logger.print(
            "Successfully deleted VPC peering connection: {}.".format(peer_name))
    except Exception as e:
        cli_logger.error("Failed to delete VPC peering connection. {}", str(e))
        raise e


def _get_vpc_peering_connection(config, compute, vpc_id, peer_name):
    provider_config = config["provider"]
    project_id = provider_config.get("project_id")
    vpc_info = compute.networks().get(project=project_id, network=vpc_id).execute()
    peerings = vpc_info.get("peerings")
    if peerings is not None:
        for peering in peerings:
            if peering["name"] == peer_name:
                return peering
    return None


def _create_workspace_vpc_peering_connection(config, compute, vpc_id, working_vpc_id):
    workspace_name = config["workspace_name"]
    peer_name = GCP_WORKSPACE_VPC_PEERING_NAME.format(workspace_name)
    working_vpc_name = get_vpc_name_by_id(config, compute, working_vpc_id)
    _create_vpc_peering_connection(
        config, compute, vpc_id,
        peer_name=peer_name, peering_vpc_name=working_vpc_name
    )


def _create_working_vpc_peering_connection(config, compute, vpc_id, working_vpc_id):
    workspace_name = config["workspace_name"]
    peer_name = GCP_WORKING_VPC_PEERING_NAME.format(workspace_name)
    workspace_vpc_name = _get_workspace_vpc_name(workspace_name)
    _create_vpc_peering_connection(
        config, compute, working_vpc_id,
        peer_name=peer_name, peering_vpc_name=workspace_vpc_name
    )


def _delete_vpc_peering_connections(config, compute):
    current_step = 1
    total_steps = 2

    with cli_logger.group(
            "Deleting working VPC peering connection",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_working_vpc_peering_connection(config, compute)

    with cli_logger.group(
            "Deleting workspace VPC peering connection",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_workspace_vpc_peering_connection(config, compute)


def _delete_workspace_vpc_peering_connection(config, compute):
    workspace_name = config["workspace_name"]
    peer_name = GCP_WORKSPACE_VPC_PEERING_NAME.format(workspace_name)
    vpc_id = get_workspace_vpc_id(config, compute)

    peering = _get_vpc_peering_connection(config, compute, vpc_id, peer_name)
    if peering is None:
        cli_logger.print(
            "The workspace peering connection {} doesn't exist. Skip deletion.".format(peer_name))
        return

    _delete_vpc_peering_connection(
        config, compute, vpc_id, peer_name
    )


def _delete_working_vpc_peering_connection(config, compute):
    workspace_name = config["workspace_name"]
    peer_name = GCP_WORKING_VPC_PEERING_NAME.format(workspace_name)
    working_vpc_id = get_working_node_vpc_id(config, compute)

    peering = _get_vpc_peering_connection(config, compute, working_vpc_id, peer_name)
    if peering is None:
        cli_logger.print(
            "The workspace peering connection {} doesn't exist. Skip deletion.".format(peer_name))
        return

    _delete_vpc_peering_connection(
        config, compute, working_vpc_id, peer_name
    )


def get_workspace_vpc_peering_connections(config, compute, vpc_id):
    workspace_name = config["workspace_name"]
    workspace_peer_name = GCP_WORKSPACE_VPC_PEERING_NAME.format(workspace_name)
    vpc_peerings = {}
    workspace_peering = _get_vpc_peering_connection(config, compute, vpc_id, workspace_peer_name)
    if workspace_peering:
        vpc_peerings["a"] = workspace_peering

    working_vpc_id = get_working_node_vpc_id(config, compute)
    if working_vpc_id is not None:
        working_peer_name = GCP_WORKING_VPC_PEERING_NAME.format(workspace_name)
        working_peering = _get_vpc_peering_connection(config, compute, working_vpc_id, working_peer_name)
        if working_peering:
            vpc_peerings["b"] = working_peering

    return vpc_peerings
