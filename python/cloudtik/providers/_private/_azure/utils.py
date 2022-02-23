def with_azure_config(cmds, config):
    resource_group = config.get("provider").get("resource_group")
    subscription_id = config.get("provider").get("subscription_id")
    azure_storage_kind = config.get("provider").get("azure_cloud_storage").get("azure.storage.kind")
    azure_storage_account = config.get("provider").get("azure_cloud_storage").get("azure.storage.account")
    azure_container = config.get("provider").get("azure_cloud_storage").get(
        "azure.container")
    azure_account_key = config.get("provider").get("azure_cloud_storage"). \
        get("azure.account.key")
    out = []
    for cmd in cmds:
        out.append("export RESOURCE_GROUP={}; export SUBSCRIPTION_ID={}; export AZURE_STORAGE_KIND={}; "
                   "export AZURE_STORAGE_ACCOUNT={}; export AZURE_CONTAINER={}; export AZURE_ACCOUNT_KEY={}; {}".format(
                    resource_group, subscription_id, azure_storage_kind, azure_storage_account, azure_container,
                    azure_account_key, cmd))
    return out
