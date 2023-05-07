from enum import Enum

from google.cloud import container_v1

from cloudtik.providers._private.gcp.utils import _get_gcp_credentials

GCP_KUBERNETES_HEAD_IAM_SERVICE_ACCOUNT_NAME = "cloudtik-gke-{}-head"
GCP_KUBERNETES_WORKER_IAM_SERVICE_ACCOUNT_NAME = "cloudtik-gke-{}-worker"

GCP_KUBERNETES_HEAD_SERVICE_ACCOUNT_DISPLAY_NAME = "CloudTik GKE Head Service Account - {}"
GCP_KUBERNETES_WORKER_SERVICE_ACCOUNT_DISPLAY_NAME = "CloudTik GKE Worker Service Account - {}"


class AccountType(Enum):
    HEAD = 1
    WORKER = 2


def get_project_id(cloud_provider):
    return cloud_provider["project_id"]


def _get_iam_service_account_name(workspace_name, account_type: AccountType):
    if account_type == AccountType.HEAD:
        return _get_head_iam_service_account_name(workspace_name)
    else:
        return _get_worker_iam_service_account_name(workspace_name)


def _get_head_iam_service_account_name(workspace_name):
    return GCP_KUBERNETES_HEAD_IAM_SERVICE_ACCOUNT_NAME.format(workspace_name)


def _get_worker_iam_service_account_name(workspace_name):
    return GCP_KUBERNETES_WORKER_IAM_SERVICE_ACCOUNT_NAME.format(workspace_name)


def _get_iam_service_account_display_name(workspace_name, account_type: AccountType):
    if account_type == AccountType.HEAD:
        return GCP_KUBERNETES_HEAD_SERVICE_ACCOUNT_DISPLAY_NAME.format(workspace_name)
    else:
        return GCP_KUBERNETES_WORKER_SERVICE_ACCOUNT_DISPLAY_NAME.format(workspace_name)


def construct_container_client(provider_config):
    credentials = _get_gcp_credentials(provider_config)
    if credentials is None:
        return _create_container_client()

    return _create_container_client(credentials)


def _create_container_client(gcp_credentials=None):
    return container_v1.ClusterManagerClient(credentials=gcp_credentials)
