def with_gcs_config(cmds, config):
    project_id = config.get("provider").get("project_id")
    gcp_gcs_bucket = config.get("provider").get("gcp_cloud_storage").get("gcp.gcs.bucket")
    fs_gs_auth_service_account_email = config.get("provider").get("gcp_cloud_storage").get(
        "fs.gs.auth.service.account.email")
    fs_gs_auth_service_account_private_key_id = config.get("provider").get("gcp_cloud_storage"). \
        get("fs.gs.auth.service.account.private.key.id")
    fs_gs_auth_service_account_private_key = config.get("provider").get("gcp_cloud_storage"). \
        get("fs.gs.auth.service.account.private.key")
    out = []
    for cmd in cmds:
        out.append("export PROJECT_ID={}; export GCP_GCS_BUCKET={}; export FS_GS_AUTH_SERVICE_ACCOUNT_EMAIL={}; "
                   "export FS_GS_AUTH_SERVICE_ACCOUNT_PRIVATE_KEY_ID={}; "
                   "export FS_GS_AUTH_SERVICE_ACCOUNT_PRIVATE_KEY={}; {}".format(project_id, gcp_gcs_bucket,
                                                                                 fs_gs_auth_service_account_email,
                                                                                 fs_gs_auth_service_account_private_key_id,
                                                                                 fs_gs_auth_service_account_private_key,
                                                                                 cmd))
    return out
