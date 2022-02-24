def with_azure_config(cmds, config):
    azure_storage_kind = config.get("provider").get("azure_cloud_storage").get("azure.storage.kind")
    azure_storage_account = config.get("provider").get("azure_cloud_storage").get("azure.storage.account")
    azure_container = config.get("provider").get("azure_cloud_storage").get(
        "azure.container")
    azure_account_key = config.get("provider").get("azure_cloud_storage"). \
        get("azure.account.key")
    out = []
    for cmd in cmds:
        out.append("export AZURE_STORAGE_KIND={}; export AZURE_STORAGE_ACCOUNT={}; export AZURE_CONTAINER={}; export "
                   "AZURE_ACCOUNT_KEY={}; {}".format(azure_storage_kind, azure_storage_account, azure_container,
                                                     azure_account_key, cmd))
    return out
