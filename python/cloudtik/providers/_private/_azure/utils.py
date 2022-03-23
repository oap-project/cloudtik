def get_azure_config(provider_config):
    config_dict = {
        "AZURE_STORAGE_KIND": provider_config.get("azure_cloud_storage", {}).get("azure.storage.kind"),
        "AZURE_STORAGE_ACCOUNT": provider_config.get("azure_cloud_storage", {}).get("azure.storage.account"),
        "AZURE_CONTAINER": provider_config.get("azure_cloud_storage", {}).get(
            "azure.container"),
        "AZURE_ACCOUNT_KEY": provider_config.get("azure_cloud_storage", {}).get(
            "azure.account.key")}
    return config_dict
