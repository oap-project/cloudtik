import time
import logging
from typing import Any, Dict

from googleapiclient import discovery, errors

from google.cloud import storage
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials as OAuthCredentials

from cloudtik.core._private.cli_logger import cli_logger
from cloudtik.core._private.constants import CLOUDTIK_DEFAULT_CLOUD_STORAGE_URI
from cloudtik.core._private.utils import get_storage_config_for_update, get_database_config_for_update
from cloudtik.providers._private.gcp.node import (GCPNodeType, MAX_POLLS,
                                                  POLL_INTERVAL)
from cloudtik.providers._private.gcp.node import GCPNode

logger = logging.getLogger(__name__)

TPU_VERSION = "v2alpha"  # change once v2 is stable

# If there are TPU nodes in config, this field will be set
# to True in config["provider"].
HAS_TPU_PROVIDER_FIELD = "_has_tpus"

SERVICE_ACCOUNT_EMAIL_TEMPLATE = (
    "{account_id}@{project_id}.iam.gserviceaccount.com")

GCP_GCS_BUCKET = "gcs.bucket"
GCP_DATABASE_ENDPOINT = "server_address"


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


def _create_sql_admin(gcp_credentials=None):
    return discovery.build(
        "sqladmin", "v1", credentials=gcp_credentials, cache_discovery=False)


def _create_service_networking(gcp_credentials=None):
    return discovery.build(
        "servicenetworking", "v1", credentials=gcp_credentials, cache_discovery=False)


def _create_tpu(gcp_credentials=None):
    return discovery.build(
        "tpu",
        TPU_VERSION,
        credentials=gcp_credentials,
        cache_discovery=False,
        discoveryServiceUrl="https://tpu.googleapis.com/$discovery/rest")


def _create_storage_client(project=None, gcp_credentials=None):
    return storage.Client(project=project, credentials=gcp_credentials)


def _get_gcp_credentials(provider_config):
    gcp_credentials = provider_config.get("gcp_credentials")
    if gcp_credentials is None:
        logger.debug("gcp_credentials not found in cluster yaml file. "
                     "Falling back to GOOGLE_APPLICATION_CREDENTIALS "
                     "environment variable.")
        # If gcp_credentials is None, then discovery.build will search for
        # credentials in the local environment.
        return None

    assert ("type" in gcp_credentials), \
        "gcp_credentials cluster yaml field missing 'type' field."
    assert ("credentials" in gcp_credentials), \
        "gcp_credentials cluster yaml field missing 'credentials' field."

    cred_type = gcp_credentials["type"]
    credentials_fields = gcp_credentials["credentials"]

    if cred_type == "service_account":
        credentials = service_account.Credentials.from_service_account_info(
            credentials_fields)
    elif cred_type == "oauth_token":
        # Otherwise the credentials type must be oauth_token.
        credentials = OAuthCredentials(**credentials_fields)
    else:
        credentials = None

    return credentials


def construct_clients_from_provider_config(provider_config):
    """
    Attempt to fetch and parse the JSON GCP credentials from the provider
    config yaml file.

    tpu resource (the last element of the tuple) will be None if
    `_has_tpus` in provider config is not set or False.
    """
    credentials = _get_gcp_credentials(provider_config)
    if credentials is None:
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

    tpu_resource = _create_tpu(credentials) if provider_config.get(
        HAS_TPU_PROVIDER_FIELD, False) else None

    return _create_crm(credentials), \
        _create_iam(credentials), \
        _create_compute(credentials), \
        tpu_resource


def construct_compute_client(provider_config):
    credentials = _get_gcp_credentials(provider_config)
    if credentials is None:
        return _create_compute()

    return _create_compute(credentials)


def construct_crm_client(provider_config):
    credentials = _get_gcp_credentials(provider_config)
    if credentials is None:
        return _create_crm()

    return _create_crm(credentials)


def construct_iam_client(provider_config):
    credentials = _get_gcp_credentials(provider_config)
    if credentials is None:
        return _create_iam()

    return _create_iam(credentials)


def construct_storage(provider_config):
    credentials = _get_gcp_credentials(provider_config)
    if credentials is None:
        return _create_storage()

    return _create_storage(credentials)


def construct_sql_admin(provider_config):
    credentials = _get_gcp_credentials(provider_config)
    if credentials is None:
        return _create_sql_admin()

    return _create_sql_admin(credentials)


def construct_service_networking(provider_config):
    credentials = _get_gcp_credentials(provider_config)
    if credentials is None:
        return _create_service_networking()

    return _create_service_networking(credentials)


def construct_storage_client(provider_config):
    credentials = _get_gcp_credentials(provider_config)
    project_id = provider_config.get("project_id")
    if credentials is None:
        return _create_storage_client(project_id)

    return _create_storage_client(project_id, credentials)


def wait_for_crm_operation(operation, crm):
    """Poll for cloud resource manager operation until finished."""
    cli_logger.verbose("wait_for_crm_operation: "
                       "Waiting for operation {} to finish...".format(operation["name"]))

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


def wait_for_sql_admin_operation(project_id, operation, sql_admin):
    """Poll for cloud resource manager operation until finished."""
    cli_logger.verbose("wait_for_sql_admin_operation: "
                       "Waiting for operation {} to finish...".format(operation["name"]))

    for _ in range(MAX_POLLS * 5):
        result = sql_admin.operations().get(
            project=project_id,
            operation=operation["name"]).execute()
        if "error" in result:
            raise Exception(result["error"])

        if result["status"] == "DONE":
            cli_logger.verbose("wait_for_sql_admin_operation: Operation done.")
            break

        time.sleep(POLL_INTERVAL)

    return result


def wait_for_service_networking_operation(operation, service_networking):
    """Poll for cloud resource manager operation until finished."""
    cli_logger.verbose("wait_for_service_networking_operation: "
                       "Waiting for operation {} to finish...".format(operation["name"]))

    for _ in range(MAX_POLLS * 5):
        result = service_networking.operations().get(
            name=operation["name"]).execute()
        if "error" in result:
            raise Exception(result["error"])

        if "done" in result and result["done"]:
            cli_logger.verbose("wait_for_service_networking_operation: Operation done.")
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


def get_gcp_cloud_storage_config(provider_config: Dict[str, Any]):
    if "storage" in provider_config and "gcp_cloud_storage" in provider_config["storage"]:
        return provider_config["storage"]["gcp_cloud_storage"]

    return None


def get_gcp_cloud_storage_config_for_update(provider_config: Dict[str, Any]):
    storage_config = get_storage_config_for_update(provider_config)
    if "gcp_cloud_storage" not in storage_config:
        storage_config["gcp_cloud_storage"] = {}
    return storage_config["gcp_cloud_storage"]


def export_gcp_cloud_storage_config(provider_config, config_dict: Dict[str, Any]):
    cloud_storage = get_gcp_cloud_storage_config(provider_config)
    if cloud_storage is None:
        return
    config_dict["GCP_CLOUD_STORAGE"] = True

    project_id = cloud_storage.get("project_id")
    if project_id:
        config_dict["GCP_PROJECT_ID"] = project_id

    gcs_bucket = cloud_storage.get(GCP_GCS_BUCKET)
    if gcs_bucket:
        config_dict["GCP_GCS_BUCKET"] = gcs_bucket

    gs_client_email = cloud_storage.get(
        "gcs.service.account.client.email")
    if gs_client_email:
        config_dict["GCP_GCS_SERVICE_ACCOUNT_CLIENT_EMAIL"] = gs_client_email

    gs_private_key_id = cloud_storage.get(
        "gcs.service.account.private.key.id")
    if gs_private_key_id:
        config_dict["GCP_GCS_SERVICE_ACCOUNT_PRIVATE_KEY_ID"] = gs_private_key_id

    gs_private_key = cloud_storage.get(
        "gcs.service.account.private.key")
    if gs_private_key:
        config_dict["GCP_GCS_SERVICE_ACCOUNT_PRIVATE_KEY"] = gs_private_key


def get_gcp_cloud_storage_uri(gcp_cloud_storage):
    gcs_bucket = gcp_cloud_storage.get(GCP_GCS_BUCKET)
    if gcs_bucket is None:
        return None

    return "gs://{}".format(gcs_bucket)


def get_default_gcp_cloud_storage(provider_config):
    cloud_storage = get_gcp_cloud_storage_config(provider_config)
    if cloud_storage is None:
        return None

    cloud_storage_info = {}
    cloud_storage_info.update(cloud_storage)

    cloud_storage_uri = get_gcp_cloud_storage_uri(cloud_storage)
    if cloud_storage_uri:
        cloud_storage_info[CLOUDTIK_DEFAULT_CLOUD_STORAGE_URI] = cloud_storage_uri

    return cloud_storage_info


def get_gcp_database_config(provider_config: Dict[str, Any], default=None):
    if "database" in provider_config and "gcp.database" in provider_config["database"]:
        return provider_config["database"]["gcp.database"]

    return default


def get_gcp_database_config_for_update(provider_config: Dict[str, Any]):
    database_config = get_database_config_for_update(provider_config)
    if "gcp.database" not in database_config:
        database_config["gcp.database"] = {}
    return database_config["gcp.database"]


def export_gcp_cloud_database_config(provider_config, config_dict: Dict[str, Any]):
    database_config = get_gcp_database_config(provider_config)
    if database_config is None:
        return

    database_hostname = database_config.get(GCP_DATABASE_ENDPOINT)
    if database_hostname:
        config_dict["CLOUD_DATABASE"] = True
        config_dict["CLOUD_DATABASE_HOSTNAME"] = database_hostname
        config_dict["CLOUD_DATABASE_PORT"] = database_config.get("port", 3306)
        config_dict["CLOUD_DATABASE_USERNAME"] = database_config.get("username", "root")
        config_dict["CLOUD_DATABASE_PASSWORD"] = database_config.get("password", "cloudtik")


def get_default_gcp_cloud_database(provider_config):
    cloud_database = get_gcp_database_config(provider_config)
    if cloud_database is None:
        return None

    cloud_database_info = {}
    cloud_database_info.update(cloud_database)
    cloud_database_info.pop("password")
    return cloud_database_info


def _get_node_info(node: GCPNode):
    node_info = {"node_id": node["id"],
                 "instance_type": node["machineType"].split("/")[-1],
                 "private_ip": node.get_internal_ip(),
                 "public_ip": node.get_external_ip(),
                 "instance_status": node["status"]}
    node_info.update(node.get_labels())
    return node_info


def get_gcp_project(cloud_provider, project_id):
    crm = construct_crm_client(cloud_provider)
    try:
        project = crm.projects().get(projectId=project_id).execute()
    except errors.HttpError as e:
        if e.resp.status != 403:
            raise
        project = None
    return project


def get_service_account_email(project_id, account_id):
    email = SERVICE_ACCOUNT_EMAIL_TEMPLATE.format(
            account_id=account_id,
            project_id=project_id)
    return email
