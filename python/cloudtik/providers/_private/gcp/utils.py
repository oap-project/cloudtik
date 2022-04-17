def get_gcs_config(provider_config):
    config_dict = {}

    project_id = provider_config.get("project_id")
    if project_id:
        config_dict["PROJECT_ID"] = project_id

    gcp_gcs_bucket = provider_config.get("gcp_cloud_storage", {}).get("gcp.gcs.bucket")
    if gcp_gcs_bucket:
        config_dict["GCP_GCS_BUCKET"] = gcp_gcs_bucket

    gs_account_email = provider_config.get("gcp_cloud_storage", {}).get(
        "fs.gs.auth.service.account.email")
    if gs_account_email:
        config_dict["FS_GS_AUTH_SERVICE_ACCOUNT_EMAIL"] = gs_account_email

    gs_private_key_id = provider_config.get("gcp_cloud_storage", {}).get(
        "fs.gs.auth.service.account.private.key.id")
    if gs_private_key_id:
        config_dict["FS_GS_AUTH_SERVICE_ACCOUNT_PRIVATE_KEY_ID"] = gs_private_key_id

    gs_private_key = provider_config.get("gcp_cloud_storage", {}).get(
        "fs.gs.auth.service.account.private.key")
    if gs_private_key:
        config_dict["FS_GS_AUTH_SERVICE_ACCOUNT_PRIVATE_KEY"] = gs_private_key

    return config_dict
