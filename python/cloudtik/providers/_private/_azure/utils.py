def get_azure_config(provider_config):
    config_dict = {}

    azure_storage_kind = provider_config.get("azure_cloud_storage", {}).get("azure.storage.kind")
    if azure_storage_kind:
        config_dict["AZURE_STORAGE_KIND"] = azure_storage_kind

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

    return config_dict
