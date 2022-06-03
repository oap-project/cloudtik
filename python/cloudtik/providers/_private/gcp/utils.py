import time
import json
import logging
from typing import Any, Dict

from googleapiclient import discovery, errors

from google.cloud import storage
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials as OAuthCredentials

from cloudtik.core._private.cli_logger import cli_logger
from cloudtik.providers._private.gcp.node import (GCPNodeType, MAX_POLLS,
                                                  POLL_INTERVAL)
from cloudtik.providers._private.gcp.node import GCPNode

logger = logging.getLogger(__name__)

TPU_VERSION = "v2alpha"  # change once v2 is stable

# If there are TPU nodes in config, this field will be set
# to True in config["provider"].
HAS_TPU_PROVIDER_FIELD = "_has_tpus"


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


def get_gcs_config(provider_config, node_type_config: Dict[str, Any], node_id: str):
    config_dict = {}

    project_id = provider_config.get("project_id")
    if project_id:
        config_dict["PROJECT_ID"] = project_id

    gcs_bucket = provider_config.get("gcp_cloud_storage", {}).get("gcs.bucket")
    if gcs_bucket:
        config_dict["GCS_BUCKET"] = gcs_bucket

    gs_client_email = provider_config.get("gcp_cloud_storage", {}).get(
        "gcs.service.account.client.email")
    if gs_client_email:
        config_dict["GCS_SERVICE_ACCOUNT_CLIENT_EMAIL"] = gs_client_email

    gs_private_key_id = provider_config.get("gcp_cloud_storage", {}).get(
        "gcs.service.account.private.key.id")
    if gs_private_key_id:
        config_dict["GCS_SERVICE_ACCOUNT_PRIVATE_KEY_ID"] = gs_private_key_id

    gs_private_key = provider_config.get("gcp_cloud_storage", {}).get(
        "gcs.service.account.private.key")
    if gs_private_key:
        config_dict["GCS_SERVICE_ACCOUNT_PRIVATE_KEY"] = gs_private_key

    return config_dict


def _get_node_info(node: GCPNode):
    node_info = {"node_id": node["id"],
                 "instance_type": node["machineType"].split("/")[-1],
                 "private_ip": node.get_internal_ip(),
                 "public_ip": node.get_external_ip(),
                 "instance_status": node["status"]}
    node_info.update(node.get_labels())
    return node_info
