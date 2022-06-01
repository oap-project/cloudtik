import copy
from functools import partial
import json
import os
import logging
import random
import string
import time
from typing import Any, Dict, Optional

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

from googleapiclient import discovery, errors
from google.cloud import storage
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials as OAuthCredentials

from cloudtik.providers._private.gcp.node import (GCPNodeType, MAX_POLLS,
                                                  POLL_INTERVAL, GCPCompute)

from cloudtik.core.tags import CLOUDTIK_TAG_NODE_KIND, NODE_KIND_HEAD, CLOUDTIK_TAG_CLUSTER_NAME
from cloudtik.core._private.cli_logger import cli_logger, cf
from cloudtik.core._private.services import get_node_ip_address
from cloudtik.core._private.utils import check_cidr_conflict, unescape_private_key, is_use_internal_ip, \
    _is_use_internal_ip, is_managed_cloud_storage, is_use_managed_cloud_storage, is_worker_role_for_cloud_storage, \
    _is_use_managed_cloud_storage
from cloudtik.providers._private.gcp.utils import _get_node_info
from cloudtik.providers._private.utils import StorageTestingError

logger = logging.getLogger(__name__)

VERSION = "v1"
TPU_VERSION = "v2alpha"  # change once v2 is stable

GCP_RESOURCE_NAME_PREFIX = "cloudtik"
GCP_DEFAULT_SERVICE_ACCOUNT_ID = GCP_RESOURCE_NAME_PREFIX + "-sa-" + VERSION

SERVICE_ACCOUNT_EMAIL_TEMPLATE = (
    "{account_id}@{project_id}.iam.gserviceaccount.com")
DEFAULT_SERVICE_ACCOUNT_CONFIG = {
    "displayName": "CloudTik Service Account ({})".format(VERSION),
}

# Those roles will always be added.
DEFAULT_SERVICE_ACCOUNT_ROLES = [
    "roles/storage.admin", "roles/compute.admin",
    "roles/iam.serviceAccountUser"
]

GCP_HEAD_SERVICE_ACCOUNT_ID = GCP_RESOURCE_NAME_PREFIX + "-{}"
GCP_HEAD_SERVICE_ACCOUNT_DISPLAY_NAME = "CloudTik Head Service Account - {}"

GCP_WORKER_SERVICE_ACCOUNT_ID = GCP_RESOURCE_NAME_PREFIX + "-w-{}"
GCP_WORKER_SERVICE_ACCOUNT_DISPLAY_NAME = "CloudTik Worker Service Account - {}"

# Those roles will always be added.
WORKER_SERVICE_ACCOUNT_ROLES = [
    "roles/storage.admin",
    "roles/iam.serviceAccountUser"
]

# Those roles will only be added if there are TPU nodes defined in config.
TPU_SERVICE_ACCOUNT_ROLES = ["roles/tpu.admin"]

# If there are TPU nodes in config, this field will be set
# to True in config["provider"].
HAS_TPU_PROVIDER_FIELD = "_has_tpus"

# NOTE: iam.serviceAccountUser allows the Head Node to create worker nodes
# with ServiceAccounts.

NUM_GCP_WORKSPACE_CREATION_STEPS = 7
NUM_GCP_WORKSPACE_DELETION_STEPS = 6


def get_node_type(node: dict) -> GCPNodeType:
    """Returns node type based on the keys in ``node``.

    This is a very simple check. If we have a ``machineType`` key,
    this is a Compute instance. If we don't have a ``machineType`` key,
    but we have ``acceleratorType``, this is a TPU. Otherwise, it's
    invalid and an exception is raised.

    This works for both node configs and API returned nodes.
    """

    if "machineType" not in node and "acceleratorType" not in node:
        raise ValueError(
            "Invalid node. For a Compute instance, 'machineType' is "
            "required. "
            "For a TPU instance, 'acceleratorType' and no 'machineType' "
            "is required. "
            f"Got {list(node)}")

    if "machineType" not in node and "acceleratorType" in node:
        # remove after TPU pod support is added!
        if node["acceleratorType"] not in ("v2-8", "v3-8"):
            raise ValueError(
                "For now, only v2-8' and 'v3-8' accelerator types are "
                "supported. Support for TPU pods will be added in the future.")

        return GCPNodeType.TPU
    return GCPNodeType.COMPUTE


def wait_for_crm_operation(operation, crm):
    """Poll for cloud resource manager operation until finished."""
    cli_logger.verbose("wait_for_crm_operation: "
                       "Waiting for operation {} to finish...".format(operation))

    for _ in range(MAX_POLLS):
        result = crm.operations().get(name=operation["name"]).execute()
        if "error" in result:
            raise Exception(result["error"])

        if "done" in result and result["done"]:
            cli_logger.verbose("wait_for_crm_operation: Operation done.")
            break

        time.sleep(POLL_INTERVAL)

    return result


def wait_for_compute_region_operation(project_name, region, operation, compute):
    """Poll for global compute operation until finished."""
    cli_logger.verbose("wait_for_compute_region_operation: "
                       "Waiting for operation {} to finish...".format(operation["name"]))

    for _ in range(MAX_POLLS):
        result = compute.regionOperations().get(
            project=project_name,
            region=region,
            operation=operation["name"],
        ).execute()
        if "error" in result:
            raise Exception(result["error"])

        if result["status"] == "DONE":
            cli_logger.verbose("wait_for_compute_region_operation: "
                               "Operation done.")
            break

        time.sleep(POLL_INTERVAL)

    return result
 
 
def wait_for_compute_global_operation(project_name, operation, compute):
    """Poll for global compute operation until finished."""
    cli_logger.verbose("wait_for_compute_global_operation: "
                       "Waiting for operation {} to finish...".format(operation["name"]))

    for _ in range(MAX_POLLS):
        result = compute.globalOperations().get(
            project=project_name,
            operation=operation["name"],
        ).execute()
        if "error" in result:
            raise Exception(result["error"])

        if result["status"] == "DONE":
            cli_logger.verbose("wait_for_compute_global_operation: "
                               "Operation done.")
            break

        time.sleep(POLL_INTERVAL)

    return result


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


def _has_tpus_in_node_configs(config: dict) -> bool:
    """Check if any nodes in config are TPUs."""
    node_configs = [
        node_type["node_config"]
        for node_type in config["available_node_types"].values()
    ]
    return any(get_node_type(node) == GCPNodeType.TPU for node in node_configs)


def _is_head_node_a_tpu(config: dict) -> bool:
    """Check if the head node is a TPU."""
    node_configs = {
        node_id: node_type["node_config"]
        for node_id, node_type in config["available_node_types"].items()
    }
    return get_node_type(
        node_configs[config["head_node_type"]]) == GCPNodeType.TPU


def _create_crm(gcp_credentials=None):
    return discovery.build(
        "cloudresourcemanager",
        "v1",
        credentials=gcp_credentials,
        cache_discovery=False)


def _create_iam(gcp_credentials=None):
    return discovery.build(
        "iam", "v1", credentials=gcp_credentials, cache_discovery=False)


def _create_compute(gcp_credentials=None):
    return discovery.build(
        "compute", "v1", credentials=gcp_credentials, cache_discovery=False)


def _create_storage(gcp_credentials=None):
    return discovery.build(
        "storage", "v1", credentials=gcp_credentials, cache_discovery=False)


def _create_tpu(gcp_credentials=None):
    return discovery.build(
        "tpu",
        TPU_VERSION,
        credentials=gcp_credentials,
        cache_discovery=False,
        discoveryServiceUrl="https://tpu.googleapis.com/$discovery/rest")


def _create_storage_client(gcp_credentials=None):
    return storage.Client(credentials=gcp_credentials)

def construct_clients_from_provider_config(provider_config):
    """
    Attempt to fetch and parse the JSON GCP credentials from the provider
    config yaml file.

    tpu resource (the last element of the tuple) will be None if
    `_has_tpus` in provider config is not set or False.
    """
    gcp_credentials = provider_config.get("gcp_credentials")
    if gcp_credentials is None:
        logger.debug("gcp_credentials not found in cluster yaml file. "
                     "Falling back to GOOGLE_APPLICATION_CREDENTIALS "
                     "environment variable.")
        tpu_resource = _create_tpu() if provider_config.get(
            HAS_TPU_PROVIDER_FIELD, False) else None
        # If gcp_credentials is None, then discovery.build will search for
        # credentials in the local environment.
        return _create_crm(), \
            _create_iam(), \
            _create_compute(), \
            tpu_resource

    assert ("type" in gcp_credentials), \
        "gcp_credentials cluster yaml field missing 'type' field."
    assert ("credentials" in gcp_credentials), \
        "gcp_credentials cluster yaml field missing 'credentials' field."

    cred_type = gcp_credentials["type"]
    credentials_field = gcp_credentials["credentials"]

    if cred_type == "service_account":
        # If parsing the gcp_credentials failed, then the user likely made a
        # mistake in copying the credentials into the config yaml.
        try:
            service_account_info = json.loads(credentials_field)
        except json.decoder.JSONDecodeError:
            raise RuntimeError(
                "gcp_credentials found in cluster yaml file but "
                "formatted improperly.")
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info)
    elif cred_type == "credentials_token":
        # Otherwise the credentials type must be credentials_token.
        credentials = OAuthCredentials(credentials_field)

    tpu_resource = _create_tpu(credentials) if provider_config.get(
        HAS_TPU_PROVIDER_FIELD, False) else None

    return _create_crm(credentials), \
        _create_iam(credentials), \
        _create_compute(credentials), \
        tpu_resource


def get_workspace_head_nodes(provider_config, workspace_name):
    _, _, compute, tpu = \
        construct_clients_from_provider_config(provider_config)
    return _get_workspace_head_nodes(
        provider_config, workspace_name, compute=compute)


def _get_workspace_head_nodes(provider_config, workspace_name, compute):
    use_internal_ips = _is_use_internal_ip(provider_config)
    project_id = provider_config.get("project_id")
    availability_zone = provider_config.get("availability_zone")
    vpc_id = _get_gcp_vpcId(
        provider_config, workspace_name, compute, use_internal_ips)
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

    current_step = 1
    total_steps = NUM_GCP_WORKSPACE_CREATION_STEPS
    if managed_cloud_storage:
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
        cli_logger.error("Failed to create workspace. {}", str(e))
        raise e

    cli_logger.print(
        "Successfully created workspace: {}.",
        cf.bold(workspace_name))

    return config


def get_workspace_vpc_id(config, compute):
    return _get_workspace_vpc_id(
        config["provider"], config["workspace_name"], compute)


def _get_workspace_vpc_id(provider_config, workspace_name, compute):
    project_id = provider_config.get("project_id")
    vpc_name = 'cloudtik-{}-vpc'.format(workspace_name)
    cli_logger.verbose("Getting the VpcId for workspace: {}...".
                     format(vpc_name))

    VpcIds = [vpc["id"] for vpc in compute.networks().list(project=project_id).execute().get("items", "")
           if vpc["name"] == vpc_name]
    if len(VpcIds) == 0:
        cli_logger.verbose("The VPC for workspace is not found: {}.".
                         format(vpc_name))
        return None
    else:
        cli_logger.verbose_error("Successfully get the VpcId of {} for workspace.".
                         format(vpc_name))
        return VpcIds[0]


def _delete_vpc(config, compute):
    use_internal_ips = is_use_internal_ip(config)
    if use_internal_ips:
        cli_logger.print("Will not delete the current VPC.")
        return

    VpcId = get_workspace_vpc_id(config, compute)
    project_id = config["provider"].get("project_id")
    vpc_name = 'cloudtik-{}-vpc'.format(config["workspace_name"])

    if VpcId is None:
        cli_logger.print("The VPC: {} doesn't exist.".format(vpc_name))
        return

    """ Delete the VPC """
    cli_logger.print("Deleting the VPC: {}...".format(vpc_name))

    try:
        operation = compute.networks().delete(project=project_id, network=VpcId).execute()
        wait_for_compute_global_operation(project_id, operation, compute)
        cli_logger.print("Successfully deleted the VPC: {}.".format(vpc_name))
    except Exception as e:
        cli_logger.error("Failed to delete the VPC: {}. {}".format(vpc_name, str(e)))
        raise e

    return


def create_vpc(config, compute):
    project_id = config["provider"].get("project_id")
    network_body = {
        "autoCreateSubnetworks": False,
        "description": "Auto created network by cloudtik",
        "name": 'cloudtik-{}-vpc'.format(config["workspace_name"]),
        "routingConfig": {
            "routingMode": "REGIONAL"
        },
        "mtu": 1460
    }

    cli_logger.print("Creating workspace vpc on GCP...")
    # create vpc
    try:
        operation = compute.networks().insert(project=project_id, body=network_body).execute()
        wait_for_compute_global_operation(project_id, operation, compute)
        cli_logger.print("Successfully created workspace VPC: cloudtik-{}-vpc.".format(config["workspace_name"]))
    except Exception as e:
        cli_logger.error(
            "Failed to create workspace VPC. {}", str(e))
        raise e


def get_working_node_vpc_id(config, compute):
    return _get_working_node_vpc_id(config["provider"], compute)


def _get_working_node_vpc_id(provider_config, compute):
    ip_address = get_node_ip_address(address="8.8.8.8:53")
    project_id = provider_config.get("project_id")
    zone = provider_config.get("availability_zone")
    instances = compute.instances().list(project=project_id, zone=zone).execute()["items"]
    network = None
    for instance in instances:
        for networkInterface in instance.get("networkInterfaces"):
            if networkInterface.get("networkIP") == ip_address:
                network = networkInterface.get("network").split("/")[-1]
                break

    if network is None:
        cli_logger.error("Failed to get the VpcId of the working node. "
                         "Please check whether the working node is a GCP instance or not!")
        return None

    cli_logger.print("Successfully get the VpcId for working node.")
    return compute.networks().get(project=project_id, network=network).execute()["id"]


def _configure_gcp_subnets_cidr(config, compute, VpcId):
    project_id = config["provider"].get("project_id")
    region = config["provider"].get("region")
    vpc_self_link = compute.networks().get(project=project_id, network=VpcId).execute()["selfLink"]
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
    subnetwork_name = "cloudtik-{}-{}-subnet".format(workspace_name,
                                                     subnet_attribute)

    if get_subnet(config, subnetwork_name, compute) is None:
        cli_logger.print("The {} subnet {} isn't found in workspace."
                         .format(subnet_attribute, subnetwork_name))
        return

    # """ Delete custom subnet """
    cli_logger.print("Deleting {} subnet: {}...".format(subnet_attribute, subnetwork_name))
    try:
        operation = compute.subnetworks().delete(project=project_id, region=region,
                                         subnetwork=subnetwork_name).execute()
        wait_for_compute_region_operation(project_id, region, operation, compute)
        cli_logger.print("Successfully deleted {} subnet: {}."
                         .format(subnet_attribute, subnetwork_name))
    except Exception as e:
        cli_logger.error("Failed to delete the {} subnet: {}! {}"
                         .format(subnet_attribute, subnetwork_name, str(e)))
        raise e

    return


def _create_and_configure_subnets(config, compute, VpcId):
    project_id = config["provider"]["project_id"]
    region = config["provider"]["region"]

    cidr_list = _configure_gcp_subnets_cidr(config, compute, VpcId)
    assert len(cidr_list) == 2, "We must create 2 subnets for VPC: {}!".format(VpcId)

    subnets_attribute = ["public", "private"]
    for i in range(2):
        cli_logger.print("Creating subnet for the vpc: {} with CIDR: {}...".format(VpcId, cidr_list[i]))
        network_body = {
            "description": "Auto created {} subnet for cloudtik".format(subnets_attribute[i]),
            "enableFlowLogs": False,
            "ipCidrRange": cidr_list[i],
            "name": "cloudtik-{}-{}-subnet".format(config["workspace_name"], subnets_attribute[i]),
            "network": "projects/{}/global/networks/{}".format(project_id, VpcId),
            "stackType": "IPV4_ONLY",
            "privateIpGoogleAccess": False if subnets_attribute[i] == "public"  else True,
            "region": region
        }
        try:
            operation = compute.subnetworks().insert(project=project_id, region=region, body=network_body).execute()
            wait_for_compute_region_operation(project_id, region, operation, compute)
            cli_logger.print("Successfully created subnet: cloudtik-{}-{}-subnet.".
                             format(config["workspace_name"], subnets_attribute[i]))
        except Exception as e:
            cli_logger.error("Failed to create subnet. {}",  str(e))
            raise e

    return


def _create_router(config, compute, VpcId):
    project_id = config["provider"]["project_id"]
    region = config["provider"]["region"]
    workspace_name = config["workspace_name"]
    router_body = {
        "bgp": {
            "advertiseMode": "CUSTOM"
        },
        "description": "auto created for the workspace: cloudtik-{}-vpc".format(workspace_name),
        "name": "cloudtik-{}-private-router".format(workspace_name),
        "network": "projects/{}/global/networks/{}".format(project_id, VpcId),
        "region": "projects/{}/regions/{}".format(project_id, region)
    }
    cli_logger.print("Creating router for the private subnet: "
                     "cloudtik-{}-private-subnet...".format(workspace_name))
    try:
        operation = compute.routers().insert(project=project_id, region=region, body=router_body).execute()
        wait_for_compute_region_operation(project_id, region, operation, compute)
        cli_logger.print("Successfully created router for the private subnet: cloudtik-{}-subnet.".
                     format(config["workspace_name"]))
    except Exception as e:
        cli_logger.error("Failed to create router. {}", str(e))
        raise e

    return


def _create_nat_for_router(config, compute):
    project_id = config["provider"]["project_id"]
    region = config["provider"]["region"]
    workspace_name = config["workspace_name"]
    router = "cloudtik-{}-private-router".format(workspace_name)
    subnetwork_name = "cloudtik-{}-private-subnet".format(workspace_name)
    private_subnet = get_subnet(config, subnetwork_name, compute)
    private_subnet_selfLink = private_subnet.get("selfLink")
    nat_name = "cloutik-{}-nat".format(workspace_name)
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
                        "name": private_subnet_selfLink
                    }
                ],
                "sourceSubnetworkIpRangesToNat": "LIST_OF_SUBNETWORKS"
            }
        ]
    }

    cli_logger.print("Creating nat-gateway for private router: {}... ".format(nat_name))
    try:
        operation =  compute.routers().patch(project=project_id, region=region, router=router, body=router_body).execute()
        wait_for_compute_region_operation(project_id, region, operation, compute)
        cli_logger.print("Successfully created nat-gateway for the private router: {}.".
                         format(nat_name))
    except Exception as e:
        cli_logger.error("Failed to create nat-gateway. {}", str(e))
        raise e

    return


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
        cli_logger.error("Failed to delete the router: {}. {}".format(router_name, str(e)))
        raise e

    return


def check_firewall_exsit(config, compute, firewall_name):
    if get_firewall(config, compute, firewall_name) is None:
        cli_logger.verbose("The firewall {} doesn't exist.".format(firewall_name))
        return False
    else:
        cli_logger.verbose("The firewall {} exists.".format(firewall_name))
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
        operation =  compute.firewalls().insert(project=project_id, body=firewall_body).execute()
        wait_for_compute_global_operation(project_id, operation, compute)
        cli_logger.print("Successfully created firewall: {}.".format(firewall_body.get("name")))
    except Exception as e:
        cli_logger.error("Failed to create firewall. {}", str(e))
        raise e


def update_firewall(compute, project_id, firewall_body):
    cli_logger.print("Updating firewall: {}... ".format(firewall_body.get("name")))
    try:
        operation =  compute.firewalls().update(
            project=project_id, firewall=firewall_body.get("name"), body=firewall_body).execute()
        wait_for_compute_global_operation(project_id, operation, compute)
        cli_logger.print("Successfully updated firewall: {}.".format(firewall_body.get("name")))
    except Exception as e:
        cli_logger.error("Failed to update firewall. {}", str(e))
        raise e


def create_or_update_firewall(config, compute, firewall_body):
    firewall_name = firewall_body.get("name")
    project_id = config["provider"]["project_id"]

    if not check_firewall_exsit(config, compute, firewall_name):
        create_firewall(compute, project_id, firewall_body)
    else:
        cli_logger.print("The firewall {} already exists. Will update the rules... ".format(firewall_name))
        update_firewall(compute, project_id, firewall_body)


def get_subnetworks_ipCidrRange(config, compute, VpcId):
    project_id = config["provider"]["project_id"]
    subnetworks = compute.networks().get(project=project_id, network=VpcId).execute().get("subnetworks")
    subnetwork_cidrs = []
    for subnetwork in subnetworks:
        info = subnetwork.split("projects/" + project_id + "/regions/")[-1].split("/")
        subnetwork_region = info[0]
        subnetwork_name = info[-1]
        subnetwork_cidrs.append(compute.subnetworks().get(project=project_id,
                                                          region=subnetwork_region, subnetwork=subnetwork_name)
                                .execute().get("ipCidrRange"))
    return subnetwork_cidrs


def _create_default_allow_internal_firewall(config, compute, VpcId):
    project_id = config["provider"]["project_id"]
    workspace_name = config["workspace_name"]
    subnetwork_cidrs = get_subnetworks_ipCidrRange(config, compute, VpcId)
    firewall_name = "cloudtik-{}-default-allow-internal-firewall".format(workspace_name)
    firewall_body = {
        "name": firewall_name,
        "network": "projects/{}/global/networks/{}".format(project_id, VpcId),
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


def _create_or_update_custom_firewalls(config, compute, VpcId):
    firewall_rules = config["provider"] \
        .get("firewalls", {}) \
        .get("firewall_rules", [])

    project_id = config["provider"]["project_id"]
    workspace_name = config["workspace_name"]
    for i in range(len(firewall_rules)):
        firewall_body = {
            "name": "cloudtik-{}-custom-{}-firewall".format(workspace_name, i),
            "network": "projects/{}/global/networks/{}".format(project_id, VpcId),
            "allowed": firewall_rules[i]["allowed"],
            "sourceRanges": firewall_rules[i]["sourceRanges"]
        }
        create_or_update_firewall(config, compute, firewall_body)


def _create_or_update_firewalls(config, compute, VpcId):
    current_step = 1
    total_steps = 2

    with cli_logger.group(
            "Creating or updating internal firewall",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_default_allow_internal_firewall(config, compute, VpcId)

    with cli_logger.group(
            "Creating or updating custom firewalls",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_or_update_custom_firewalls(config, compute, VpcId)


def check_workspace_firewalls(config, compute):
    workspace_name = config["workspace_name"]
    firewall_names = ["cloudtik-{}-default-allow-internal-firewall".format(workspace_name)]

    for firewall_name in firewall_names:
        if not check_firewall_exsit(config, compute, firewall_name):
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
            "Failed to delete the firewall {}. {}".format(firewall_name, str(e)))
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


def get_gcp_vpcId(config, compute, use_internal_ips):
    return _get_gcp_vpcId(
        config["provider"], config.get("workspace_name"), compute, use_internal_ips)


def _get_gcp_vpcId(provider_config, workspace_name, compute, use_internal_ips):
    if use_internal_ips:
        VpcId = _get_working_node_vpc_id(provider_config, compute)
    else:
        VpcId = _get_workspace_vpc_id(provider_config, workspace_name, compute)
    return VpcId


def update_gcp_workspace_firewalls(config):
    crm, iam, compute, tpu = \
        construct_clients_from_provider_config(config["provider"])

    workspace_name = config["workspace_name"]
    use_internal_ips = is_use_internal_ip(config)
    VpcId = get_gcp_vpcId(config, compute, use_internal_ips)
    if VpcId is None:
        cli_logger.print("Workspace: {} doesn't exist!".format(config["workspace_name"]))
        return

    current_step = 1
    total_steps = 1

    try:

        with cli_logger.group(
                "Updating workspace firewalls",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _create_or_update_firewalls(config, compute, VpcId)

    except Exception as e:
        cli_logger.error(
            "Failed to update the firewalls of workspace {}. {}".format(workspace_name, str(e)))
        raise e

    cli_logger.print(
        "Successfully updated the firewalls of workspace: {}.",
        cf.bold(workspace_name))
    return None


def delete_gcp_workspace(config, delete_managed_storage: bool = False):
    crm, iam, compute, tpu = \
        construct_clients_from_provider_config(config["provider"])

    workspace_name = config["workspace_name"]
    use_internal_ips = is_use_internal_ip(config)
    managed_cloud_storage = is_managed_cloud_storage(config)
    VpcId = get_gcp_vpcId(config, compute, use_internal_ips)
    if VpcId is None:
        cli_logger.print("Workspace: {} doesn't exist!".format(config["workspace_name"]))
        return

    current_step = 1
    total_steps = NUM_GCP_WORKSPACE_DELETION_STEPS
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

            _delete_network_resources(config, compute, current_step, total_steps)

    except Exception as e:
        cli_logger.error(
            "Failed to delete workspace {}. {}".format(workspace_name, str(e)))
        raise e

    cli_logger.print(
            "Successfully deleted workspace: {}.",
            cf.bold(workspace_name))
    return None


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
    _delete_service_account(config, iam, head_service_account_id)


def _delete_worker_service_account(config, iam):
    workspace_name = config["workspace_name"]
    worker_service_account_id = GCP_WORKER_SERVICE_ACCOUNT_ID.format(workspace_name)
    _delete_service_account(config, iam, worker_service_account_id)


def _delete_service_account(config, iam, service_account_id):
    project_id = config["provider"]["project_id"]
    email = SERVICE_ACCOUNT_EMAIL_TEMPLATE.format(
        account_id=service_account_id,
        project_id=project_id)
    service_account = _get_service_account(email, config, iam)
    if service_account is None:
        cli_logger.warning("No service account with id {} found.".format(service_account_id))
        return

    try:
        cli_logger.print("Deleting service account: {}...".format(service_account_id))
        full_name = ("projects/{project_id}/serviceAccounts/{account}"
                     "".format(project_id=project_id, account=email))
        iam.projects().serviceAccounts().delete(name=full_name).execute()
        cli_logger.print("Successfully deleted the service account.")
    except Exception as e:
        cli_logger.error("Failed to delete the service account. {}", str(e))
        raise e
    return


def _delete_workspace_cloud_storage(config, workspace_name):
    bucket = get_workspace_gcs_bucket(config, workspace_name)
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
    return


def _delete_network_resources(config, compute, current_step, total_steps):
    """
         Do the work - order of operation
         1.) Delete public subnet
         2.) Delete router for private subnet 
         3.) Delete private subnets
         4.) Delete firewalls
         5.) Delete vpc
    """

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
    use_internal_ips = is_use_internal_ip(config)
    if use_internal_ips:
        # No need to create new vpc
        VpcId = get_working_node_vpc_id(config, compute)
        if VpcId is None:
            cli_logger.abort("Only when the  working node is "
                             "an GCP  instance can use use_internal_ips=True.")
    else:

        # Need to create a new vpc
        if get_workspace_vpc_id(config, compute) is None:
            create_vpc(config, compute)
            VpcId = get_workspace_vpc_id(config, compute)
        else:
            cli_logger.abort("There is a existing VPC with the same name: {}, "
                             "if you want to create a new workspace with the same name, "
                             "you need to execute workspace delete first!".format(workspace_name))
    return VpcId


def _create_head_service_account(config, crm, iam):
    workspace_name = config["workspace_name"]
    service_account_id = GCP_HEAD_SERVICE_ACCOUNT_ID.format(workspace_name)
    cli_logger.print("Creating head service account: {}...".format(service_account_id))

    try:
        service_account_config = {
            "displayName": GCP_HEAD_SERVICE_ACCOUNT_DISPLAY_NAME.format(workspace_name),
        }

        service_account = _create_service_account(
            service_account_id, service_account_config, config,
            iam)

        assert service_account is not None, "Failed to create head service account."

        if config["provider"].get(HAS_TPU_PROVIDER_FIELD, False):
            roles = DEFAULT_SERVICE_ACCOUNT_ROLES + TPU_SERVICE_ACCOUNT_ROLES
        else:
            roles = DEFAULT_SERVICE_ACCOUNT_ROLES

        _add_iam_policy_binding(service_account, roles, crm)
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
            service_account_id, service_account_config, config,
            iam)

        assert service_account is not None, "Failed to create worker service account."

        _add_iam_policy_binding(service_account, WORKER_SERVICE_ACCOUNT_ROLES, crm)
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
    workspace_name = config["workspace_name"]

    # If the managed cloud storage for the workspace already exists
    # Skip the creation step
    bucket = get_workspace_gcs_bucket(config, workspace_name)
    if bucket is not None:
        cli_logger.print("GCS bucket for the workspace already exists. Skip creation.")
        return

    region = config["provider"]["region"]
    storage_client = _create_storage_client()
    suffix = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))
    bucket_name = "cloudtik-{workspace_name}-{region}-{suffix}".format(
        workspace_name=workspace_name.lower(),
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
    return config


def _create_network_resources(config, current_step, total_steps):
    crm, iam, compute, tpu = \
        construct_clients_from_provider_config(config["provider"])

    # create vpc
    with cli_logger.group(
            "Creating VPC",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        VpcId = _create_vpc(config, compute)

    # create subnets
    with cli_logger.group(
            "Creating subnets",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_and_configure_subnets(config, compute, VpcId)

    # create router
    with cli_logger.group(
            "Creating router",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_router(config, compute, VpcId)

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
        _create_or_update_firewalls(config, compute, VpcId)

    return current_step


def check_gcp_workspace_resource(config):
    crm, iam, compute, tpu = \
        construct_clients_from_provider_config(config["provider"])
    use_internal_ips = is_use_internal_ip(config)
    workspace_name = config["workspace_name"]
    managed_cloud_storage = is_managed_cloud_storage(config)

    """
         Do the work - order of operation
         1.) Check VPC 
         2.) Check private subnet
         3.) Check public subnet
         4.) Check router
         5.) Check firewalls
         6.) Check GCS bucket
         7.) Check service accounts
    """
    if get_gcp_vpcId(config, compute, use_internal_ips) is None:
        return False
    if get_subnet(config, "cloudtik-{}-private-subnet".format(workspace_name), compute) is None:
        return False
    if get_subnet(config, "cloudtik-{}-public-subnet".format(workspace_name), compute) is None:
        return False
    if get_router(config, "cloudtik-{}-private-router".format(workspace_name), compute) is None:
        return False
    if not check_workspace_firewalls(config, compute):
        return False
    if managed_cloud_storage:
        if get_workspace_gcs_bucket(config, workspace_name) is None:
            return False
    if _get_workspace_service_account(config, iam, GCP_HEAD_SERVICE_ACCOUNT_ID) is None:
        return False
    if _get_workspace_service_account(config, iam, GCP_WORKER_SERVICE_ACCOUNT_ID) is None:
        return False
    return True


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
        return

    # User override, set or remove spot settings for worker node types
    node_types = config["available_node_types"]
    for node_type_name in node_types:
        if node_type_name == config["head_node_type"]:
            continue

        # worker node type
        node_type_data = node_types[node_type_name]
        _configure_spot_for_node_type(
            node_type_data, prefer_spot_node)


def bootstrap_gcp(config):
    workspace_name = config.get("workspace_name", "")
    if workspace_name == "":
        config = bootstrap_gcp_default(config)
    else:
        config = bootstrap_gcp_from_workspace(config)

    _configure_prefer_spot_node(config)
    return config


def bootstrap_gcp_default(config):
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
    config = _configure_project(config, crm)
    config = _configure_iam_role(config, crm, iam)
    config = _configure_key_pair(config, compute)
    config = _configure_subnet(config, compute)

    return config


def bootstrap_gcp_from_workspace(config):
    if not check_gcp_workspace_resource(config):
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

    return config


def bootstrap_gcp_workspace(config):
    # create a copy of the input config to modify
    config = copy.deepcopy(config)
    _configure_allowed_ssh_sources(config)
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
        "'project_id' must be set in the 'provider' section of the scaler"
        " config. Notice that the project id must be globally unique.")
    project = _get_project(project_id, crm)

    if project is None:
        #  Project not found, try creating it
        _create_project(project_id, crm)
        project = _get_project(project_id, crm)

    assert project is not None, "Failed to create project"
    assert project["lifecycleState"] == "ACTIVE", (
        "Project status needs to be ACTIVE, got {}".format(
            project["lifecycleState"]))

    config["provider"]["project_id"] = project["projectId"]

    return config


def _configure_cloud_storage_from_workspace(config):
    use_managed_cloud_storage = is_use_managed_cloud_storage(config)
    if use_managed_cloud_storage:
        workspace_name = config["workspace_name"]
        gcs_bucket = get_workspace_gcs_bucket(config, workspace_name)
        if gcs_bucket is None:
            cli_logger.abort("No managed GCS bucket was found. If you want to use managed GCS bucket, "
                             "you should set managed_cloud_storage equal to True when you creating workspace.")
        if "gcp_cloud_storage" not in config["provider"]:
            config["provider"]["gcp_cloud_storage"] = {}
        config["provider"]["gcp_cloud_storage"]["gcs.bucket"] = gcs_bucket.name

    return config


def _configure_iam_role(config, crm, iam):
    """Setup a gcp service account with IAM roles.

    Creates a gcp service acconut and binds IAM roles which allow it to control
    control storage/compute services. Specifically, the head node needs to have
    an IAM role that allows it to create further gce instances and store items
    in google cloud storage.

    TODO: Allow the name/id of the service account to be configured
    """
    config = copy.deepcopy(config)

    email = SERVICE_ACCOUNT_EMAIL_TEMPLATE.format(
        account_id=GCP_DEFAULT_SERVICE_ACCOUNT_ID,
        project_id=config["provider"]["project_id"])
    service_account = _get_service_account(email, config, iam)

    if service_account is None:
        cli_logger.print("Creating new service account: {}".format(GCP_DEFAULT_SERVICE_ACCOUNT_ID))

        service_account = _create_service_account(
            GCP_DEFAULT_SERVICE_ACCOUNT_ID, DEFAULT_SERVICE_ACCOUNT_CONFIG, config,
            iam)

    assert service_account is not None, "Failed to create service account"

    if config["provider"].get(HAS_TPU_PROVIDER_FIELD, False):
        roles = DEFAULT_SERVICE_ACCOUNT_ROLES + TPU_SERVICE_ACCOUNT_ROLES
    else:
        roles = DEFAULT_SERVICE_ACCOUNT_ROLES

    _add_iam_policy_binding(service_account, roles, crm)

    serviceAccounts = [{
            "email": service_account["email"],
            # NOTE: The amount of access is determined by the scope + IAM
            # role of the service account. Even if the cloud-platform scope
            # gives (scope) access to the whole cloud-platform, the service
            # account is limited by the IAM rights specified below.
            "scopes": ["https://www.googleapis.com/auth/cloud-platform"]
        }]
    worker_role_for_cloud_storage = is_worker_role_for_cloud_storage(config)
    if worker_role_for_cloud_storage:
        for key, node_type in config["available_node_types"].items():
            node_config = node_type["node_config"]
            node_config["serviceAccounts"] = serviceAccounts
    else:
        config["head_node"]["serviceAccounts"] = serviceAccounts

    return config


def _get_workspace_service_account(config, iam, service_account_id_template):
    workspace_name = config["workspace_name"]
    service_account_id = service_account_id_template.format(workspace_name)
    email = SERVICE_ACCOUNT_EMAIL_TEMPLATE.format(
        account_id=service_account_id,
        project_id=config["provider"]["project_id"])
    service_account = _get_service_account(email, config, iam)
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


def _configure_subnet(config, compute):
    """Pick a reasonable subnet if not specified by the config."""
    config = copy.deepcopy(config)

    node_configs = [
        node_type["node_config"]
        for node_type in config["available_node_types"].values()
    ]
    # Rationale: avoid subnet lookup if the network is already
    # completely manually configured

    # networkInterfaces is compute, networkConfig is TPU
    if all("networkInterfaces" in node_config or "networkConfig" in node_config
           for node_config in node_configs):
        return config

    subnets = _list_subnets(config, compute)

    if not subnets:
        raise NotImplementedError("Should be able to create subnet.")

    # TODO: make sure that we have usable subnet. Maybe call
    # compute.subnetworks().listUsable? For some reason it didn't
    # work out-of-the-box
    default_subnet = subnets[0]

    default_interfaces = [{
        "subnetwork": default_subnet["selfLink"],
        "accessConfigs": [{
            "name": "External NAT",
            "type": "ONE_TO_ONE_NAT",
        }],
    }]

    for node_config in node_configs:
        # The not applicable key will be removed during node creation

        # compute
        if "networkInterfaces" not in node_config:
            node_config["networkInterfaces"] = copy.deepcopy(
                default_interfaces)
        # TPU
        if "networkConfig" not in node_config:
            node_config["networkConfig"] = copy.deepcopy(default_interfaces)[0]
            node_config["networkConfig"].pop("accessConfigs")

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


def _list_subnets(config, compute):
    response = compute.subnetworks().list(
        project=config["provider"]["project_id"],
        region=config["provider"]["region"]).execute()

    return response["items"]


def get_subnet(config, subnetwork_name, compute):
    cli_logger.verbose("Getting the existing subnet: {}.".format(subnetwork_name))
    try:
        subnet = compute.subnetworks().get(
            project=config["provider"]["project_id"],
            region=config["provider"]["region"],
            subnetwork=subnetwork_name,
        ).execute()
        cli_logger.verbose("Successfully get the subnet: {}.".format(subnetwork_name))
        return subnet
    except Exception:
        cli_logger.verbose_error("Failed to get the subnet: {}.".format(subnetwork_name))
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


def get_workspace_gcs_bucket(config, workspace_name):
    gcs = _create_storage_client()
    region = config["provider"]["region"]
    project_id = config["provider"]["project_id"]
    bucket_name_prefix = "cloudtik-{workspace_name}-{region}-".format(
        workspace_name=workspace_name.lower(),
        region=region
    )
    for bucket in gcs.list_buckets(project=project_id):
        if bucket_name_prefix in bucket.name:
            cli_logger.verbose("Successfully get the GCS bucket: {}.".format(bucket.name))
            return bucket

    cli_logger.verbose("Failed to get the GCS bucket for workspace.")
    return None


def _create_project(project_id, crm):
    operation = crm.projects().create(body={
        "projectId": project_id,
        "name": project_id
    }).execute()

    result = wait_for_crm_operation(operation, crm)

    return result


def _get_service_account(account, config, iam):
    project_id = config["provider"]["project_id"]
    full_name = ("projects/{project_id}/serviceAccounts/{account}"
                 "".format(project_id=project_id, account=account))
    try:
        cli_logger.verbose("Getting service account: {}...".format(account))
        service_account = iam.projects().serviceAccounts().get(
            name=full_name).execute()
    except errors.HttpError as e:
        if e.resp.status != 404:
            raise
        cli_logger.verbose("Service account doesn't exist: {}...".format(account))
        service_account = None

    return service_account


def _create_service_account(account_id, account_config, config, iam):
    project_id = config["provider"]["project_id"]

    service_account = iam.projects().serviceAccounts().create(
        name="projects/{project_id}".format(project_id=project_id),
        body={
            "accountId": account_id,
            "serviceAccount": account_config,
        }).execute()

    return service_account


def _add_iam_policy_binding(service_account, roles, crm):
    """Add new IAM roles for the service account."""
    project_id = service_account["projectId"]
    email = service_account["email"]
    member_id = "serviceAccount:" + email

    policy = crm.projects().getIamPolicy(
        resource=project_id, body={}).execute()

    already_configured = True
    for role in roles:
        role_exists = False
        for binding in policy["bindings"]:
            if binding["role"] == role:
                if member_id not in binding["members"]:
                    binding["members"].append(member_id)
                    already_configured = False
                role_exists = True

        if not role_exists:
            already_configured = False
            policy["bindings"].append({
                "members": [member_id],
                "role": role,
            })

    if already_configured:
        # In some managed environments, an admin needs to grant the
        # roles, so only call setIamPolicy if needed.
        return

    result = crm.projects().setIamPolicy(
        resource=project_id, body={
            "policy": policy,
        }).execute()

    return result


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
    gcs_storage = provider_config.get("gcp_cloud_storage")
    if gcs_storage is None:
        return

    try:
        use_managed_cloud_storage = _is_use_managed_cloud_storage(provider_config)
        if use_managed_cloud_storage:
            storage_gcs = _create_storage()
        else:
            private_key_id = gcs_storage.get("gcs.service.account.private.key.id")
            if private_key_id is None:
                # The bucket may be able to accessible from roles
                # Verify through the client credential
                storage_gcs = _create_storage()
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

        storage_gcs.buckets().get(bucket=gcs_storage["gcs.bucket"]).execute()
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
