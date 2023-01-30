from enum import Enum
import uuid

from azure.common.credentials import get_cli_profile
from azure.mgmt.msi import ManagedServiceIdentityClient
from azure.mgmt.containerservice import ContainerServiceClient

from cloudtik.providers._private._azure.utils import get_client_credential

AZURE_KUBERNETES_WORKSPACE_RESOURCE_GROUP_NAME = "cloudtik-aks-{}"

AZURE_KUBERNETES_HEAD_IAM_USER_ASSIGNED_IDENTITY_NAME = "cloudtik-aks-{}-head"
AZURE_KUBERNETES_WORKER_IAM_USER_ASSIGNED_IDENTITY_NAME = "cloudtik-aks-{}-worker"

AZURE_KUBERNETES_HEAD_USER_ASSIGNED_IDENTITY_DISPLAY_NAME = "CloudTik AKS Head User Assigned Identity - {}"
AZURE_KUBERNETES_WORKER_USER_ASSIGNED_IDENTITY_DISPLAY_NAME = "CloudTik AKS Worker User Assigned Identity - {}"

AZURE_KUBERNETES_HEAD_FEDERATED_IDENTITY_CREDENTIAL_NAME = "cloudtik-fic-{}-head"
AZURE_KUBERNETES_WORKER_FEDERATED_IDENTITY_CREDENTIAL_NAME = "cloudtik-fic-{}-worker"


class AccountType(Enum):
    HEAD = 1
    WORKER = 2


def get_aks_workspace_resource_group_name(workspace_name):
    return AZURE_KUBERNETES_WORKSPACE_RESOURCE_GROUP_NAME.format(workspace_name)


def _get_iam_user_assigned_identity_name(workspace_name, account_type: AccountType):
    if account_type == AccountType.HEAD:
        return _get_head_iam_user_assigned_identity_name(workspace_name)
    else:
        return _get_worker_iam_user_assigned_identity_name(workspace_name)


def _get_head_iam_user_assigned_identity_name(workspace_name):
    return AZURE_KUBERNETES_HEAD_IAM_USER_ASSIGNED_IDENTITY_NAME.format(workspace_name)


def _get_worker_iam_user_assigned_identity_name(workspace_name):
    return AZURE_KUBERNETES_WORKER_IAM_USER_ASSIGNED_IDENTITY_NAME.format(workspace_name)


def _get_iam_user_assigned_identity_display_name(workspace_name, account_type: AccountType):
    if account_type == AccountType.HEAD:
        return AZURE_KUBERNETES_HEAD_USER_ASSIGNED_IDENTITY_DISPLAY_NAME.format(workspace_name)
    else:
        return AZURE_KUBERNETES_WORKER_USER_ASSIGNED_IDENTITY_DISPLAY_NAME.format(workspace_name)


def get_iam_role_assignment_name_for_storage_blob_data_owner(
        cloud_provider, workspace_name, account_type: AccountType):
    subscription_id = cloud_provider.get("subscription_id")
    role_type = "head" if account_type == AccountType.HEAD else "worker"
    role_assignment_name = str(uuid.uuid3(uuid.UUID(subscription_id),
                                          workspace_name + role_type + "aks_storage_blob_data_owner"))
    return role_assignment_name


def _get_federated_identity_credential_name(workspace_name, account_type: AccountType):
    if account_type == AccountType.HEAD:
        return AZURE_KUBERNETES_HEAD_FEDERATED_IDENTITY_CREDENTIAL_NAME.format(workspace_name)
    else:
        return AZURE_KUBERNETES_WORKER_FEDERATED_IDENTITY_CREDENTIAL_NAME.format(workspace_name)


def construct_container_service_client(config):
    return _construct_container_service_client(config["provider"])


def _construct_container_service_client(provider_config):
    subscription_id = provider_config.get("subscription_id")
    if subscription_id is None:
        subscription_id = get_cli_profile().get_subscription_id()
    credential = get_client_credential(provider_config)
    client = ContainerServiceClient(
        credential, subscription_id,
        api_version="2022-06-02-preview")
    return client


def _construct_manage_server_identity_client(provider_config):
    subscription_id = provider_config.get("subscription_id")
    if subscription_id is None:
        subscription_id = get_cli_profile().get_subscription_id()
    credential = get_client_credential(provider_config)
    msi_client = ManagedServiceIdentityClient(
        credential, subscription_id,
        api_version="2022-01-31-preview")
    return msi_client
