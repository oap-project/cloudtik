from typing import Any, Dict


def get_azure_config(provider_config, node_config: Dict[str, Any], node_id: str):
    config_dict = {}

    azure_storage_type = provider_config.get("azure_cloud_storage", {}).get("azure.storage.type")
    if azure_storage_type:
        config_dict["AZURE_STORAGE_TYPE"] = azure_storage_type

    azure_storage_account = provider_config.get("azure_cloud_storage", {}).get("azure.storage.account")
    if azure_storage_account:
        config_dict["AZURE_STORAGE_ACCOUNT"] = azure_storage_account

    azure_container = provider_config.get("azure_cloud_storage", {}).get(
            "azure.container")
    if azure_container:
        config_dict["AZURE_CONTAINER"] = azure_container

    azure_account_key = provider_config.get("azure_cloud_storage", {}).get(
            "azure.account.key")
    if azure_account_key:
        config_dict["AZURE_ACCOUNT_KEY"] = azure_account_key

    user_assigned_identity_client_id = provider_config.get("azure_cloud_storage", {}).get(
        "azure.user.assigned.identity.client.id")
    if user_assigned_identity_client_id:
        config_dict["AZURE_MANAGED_IDENTITY_CLIENT_ID"] = user_assigned_identity_client_id

    user_assigned_identity_tenant_id = provider_config.get("azure_cloud_storage", {}).get(
        "azure.user.assigned.identity.tenant.id")
    if user_assigned_identity_tenant_id:
        config_dict["AZURE_MANAGED_IDENTITY_TENANT_ID"] = user_assigned_identity_tenant_id

    return config_dict


def _get_node_info(node):
    node_info = {"node_id": node["name"].split("-")[-1],
                 "instance_type": node["vm_size"],
                 "private_ip": node["internal_ip"],
                 "public_ip": node["external_ip"],
                 "instance_status": node["status"]}
    node_info.update(node["tags"])

    return node_info
