def get_gcs_config(provider_config):
    config_dict = {"PROJECT_ID": provider_config.get("project_id"),
                   "GCP_GCS_BUCKET": provider_config.get("gcp_cloud_storage", {}).get("gcp.gcs.bucket"),
                   "FS_GS_AUTH_SERVICE_ACCOUNT_EMAIL": provider_config.get("gcp_cloud_storage", {}).get(
                       "fs.gs.auth.service.account.email"),
                   "FS_GS_AUTH_SERVICE_ACCOUNT_PRIVATE_KEY_ID": provider_config.get(
                       "gcp_cloud_storage", {}).get("fs.gs.auth.service.account.private.key.id"),
                   "FS_GS_AUTH_SERVICE_ACCOUNT_PRIVATE_KEY": provider_config.get("gcp_cloud_storage", {}).get(
                       "fs.gs.auth.service.account.private.key")}
    return config_dict
