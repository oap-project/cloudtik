def with_azure_config(cmds, config):
    azure_storage_account = config.get("provider").get("azure_blob_storage").get("azure.storage.account")
    azure_container = config.get("provider").get("azure_blob_storage").get(
        "azure.container")
    fs_azure_account_key_blob_core_windows_net = config.get("provider").get("azure_blob_storage"). \
        get("fs.azure.account.key.blob.core.windows.net")
    out = []
    for cmd in cmds:
        out.append("export AZURE_STORAGE_ACCOUNT={}; export AZURE_CONTAINER={}; export "
                   "FS_AZURE_ACCOUNT_KEY_BLOB_CORE_WINDOWS_NET={}; {}".format(azure_storage_account, azure_container,
                                                                              fs_azure_account_key_blob_core_windows_net,
                                                                              cmd))
    return out
