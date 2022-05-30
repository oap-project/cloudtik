from typing import Any, Dict

from cloudtik.providers._private.gcp.node import GCPNode


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
