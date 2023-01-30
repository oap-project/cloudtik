from typing import Any, Dict
import time

from azure.core.exceptions import ResourceNotFoundError

from cloudtik.core._private.cli_logger import cli_logger
from cloudtik.core._private.utils import _is_use_managed_cloud_storage, _is_managed_cloud_storage
from cloudtik.core.workspace_provider import Existence
from cloudtik.providers._private._kubernetes import core_api, log_prefix
from cloudtik.providers._private._kubernetes.azure_aks.utils import get_aks_workspace_resource_group_name, \
    AccountType, _get_iam_user_assigned_identity_name, get_iam_role_assignment_name_for_storage_blob_data_owner, \
    _get_federated_identity_credential_name, _construct_container_service_client, \
    _construct_manage_server_identity_client
from cloudtik.providers._private._kubernetes.utils import _get_head_service_account_name, \
    _get_worker_service_account_name, _get_service_account
from cloudtik.providers._private._azure.config import _configure_managed_cloud_storage_from_workspace, \
    _create_managed_cloud_storage, _delete_managed_cloud_storage, \
    get_azure_managed_cloud_storage_info, _create_user_assigned_identity, _delete_user_assigned_identity, \
    _get_user_assigned_identity, _create_role_assignment_for_storage_blob_data_owner, \
    _delete_role_assignment_for_storage_blob_data_owner, _get_role_assignment_for_storage_blob_data_owner, \
    _create_resource_group, _delete_resource_group, _get_resource_group_by_name, _get_container_for_storage_account
from cloudtik.providers._private._azure.utils import export_azure_cloud_storage_config, \
    get_default_azure_cloud_storage

AZURE_KUBERNETES_ANNOTATION_NAME = "azure.workload.identity/client-id"
AZURE_KUBERNETES_ANNOTATION_VALUE = "{user_assigned_identity_client_id}"

AZURE_KUBERNETES_WORKLOAD_IDENTITY_LABEL_NAME = "azure.workload.identity/use"

AZURE_KUBERNETES_SERVICE_ACCOUNT_WORKLOAD_IDENTITY_SUBJECT = "system:serviceaccount:{namespace}:{service_account}"

AZURE_KUBERNETES_HEAD_IAM_USER_ASSIGNED_IDENTITY_INFO = "azure.kubernetes.head.iam.user_assigned_identity"
AZURE_KUBERNETES_WORKER_IAM_USER_ASSIGNED_IDENTITY_INFO = "azure.kubernetes.worker.iam.user_assigned_identity"

AZURE_KUBERNETES_NUM_CREATION_STEPS = 2
AZURE_KUBERNETES_NUM_DELETION_STEPS = 2

AZURE_KUBERNETES_IAM_ROLE_CREATION_NUM_STEPS = 4
AZURE_KUBERNETES_IAM_ROLE_DELETION_NUM_STEPS = 4

AZURE_KUBERNETES_HEAD_WORKER_FACED_NUM_STEPS = 2

AZURE_KUBERNETES_TARGET_RESOURCES = 9


def _get_service_account_name(provider_config, account_type: AccountType):
    if account_type == AccountType.HEAD:
        return _get_head_service_account_name(provider_config)
    else:
        return _get_worker_service_account_name(provider_config)


def _get_kubernetes_service_account_iam_subject(
        namespace, service_account_name):
    return AZURE_KUBERNETES_SERVICE_ACCOUNT_WORKLOAD_IDENTITY_SUBJECT.format(
        namespace=namespace, service_account=service_account_name
    )


def _get_aks_oidc_issuer_url(cloud_provider):
    # Implement the get of issuer url through container service
    managed_cluster = _get_managed_cluster(cloud_provider)
    if not managed_cluster.oidc_issuer_profile.enabled:
        raise RuntimeError("AKS cluster {} is not enabled with OIDC provider.".format(
            managed_cluster.name))

    return managed_cluster.oidc_issuer_profile.issuer_url


def _get_managed_cluster(cloud_provider):
    aks_resource_group = cloud_provider.get("aks_resource_group")
    aks_cluster_name = cloud_provider.get("aks_cluster_name")
    if not aks_resource_group:
        raise RuntimeError("AKS cluster resource group must specified with aks_resource_group key in cloud provider.")
    if not aks_cluster_name:
        raise RuntimeError("AKS cluster name must specified with aks_cluster_name key in cloud provider.")

    container_service_client = _construct_container_service_client(cloud_provider)
    cli_logger.verbose("Getting AKS cluster information: {}.{}...".format(
        aks_resource_group, aks_cluster_name))
    try:
        managed_cluster = container_service_client.managed_clusters.get(
            aks_resource_group,
            aks_cluster_name
        )
        cli_logger.verbose("Successfully get AKS cluster information: {}.".format(
            aks_resource_group, aks_cluster_name))
        return managed_cluster
    except ResourceNotFoundError as e:
        cli_logger.verbose_error(
            "Failed to get AKS cluster information: {}.{}. {}",
            aks_resource_group, aks_cluster_name, str(e))
        return None


def create_configurations_for_azure(config: Dict[str, Any], namespace, cloud_provider):
    workspace_name = config["workspace_name"]
    managed_cloud_storage = _is_managed_cloud_storage(cloud_provider)

    current_step = 1
    total_steps = AZURE_KUBERNETES_NUM_CREATION_STEPS
    if managed_cloud_storage:
        total_steps += 1

    # create resource group
    with cli_logger.group(
            "Creating resource group",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        resource_group_name = _create_aks_resource_group(cloud_provider, workspace_name)

    # Configure IAM based access for Kubernetes service accounts
    with cli_logger.group(
            "Creating IAM based access for Kubernetes",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_iam_based_access_for_kubernetes(config, namespace, cloud_provider)

    # Optionally, create managed cloud storage (Azure DataLake) if user choose to
    if managed_cloud_storage:
        with cli_logger.group(
                "Creating Azure DataLake storage",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _create_managed_cloud_storage(
                cloud_provider, workspace_name, resource_group_name)


def delete_configurations_for_azure(
        config: Dict[str, Any], namespace, cloud_provider,
        delete_managed_storage: bool = False):
    workspace_name = config["workspace_name"]
    resource_group_name = get_aks_workspace_resource_group_name(workspace_name)
    managed_cloud_storage = _is_managed_cloud_storage(cloud_provider)

    current_step = 1
    total_steps = AZURE_KUBERNETES_NUM_DELETION_STEPS
    if managed_cloud_storage and delete_managed_storage:
        total_steps += 1

    # Delete in a reverse way of creating
    if managed_cloud_storage and delete_managed_storage:
        with cli_logger.group(
                "Deleting Azure Datalake storage",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _delete_managed_cloud_storage(
                cloud_provider, workspace_name, resource_group_name)

    # Delete S3 IAM role based access for Kubernetes service accounts
    with cli_logger.group(
            "Deleting IAM role based access for Kubernetes",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_iam_based_access_for_kubernetes(config, namespace, cloud_provider)

    # delete resource group
    with cli_logger.group(
            "Deleting resource group",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_aks_resource_group(cloud_provider, workspace_name)


def configure_kubernetes_for_azure(config: Dict[str, Any], namespace, cloud_provider):
    # Optionally, if user choose to use managed cloud storage (Azure DataLake)
    # Configure the Azure DataLake container under cloud storage
    _configure_cloud_storage_for_azure(config, cloud_provider)


def _configure_cloud_storage_for_azure(config: Dict[str, Any], cloud_provider):
    use_managed_cloud_storage = _is_use_managed_cloud_storage(cloud_provider)
    if use_managed_cloud_storage:
        workspace_name = config["workspace_name"]
        resource_group_name = get_aks_workspace_resource_group_name(workspace_name)
        _configure_managed_cloud_storage_from_workspace(
            config, cloud_provider, resource_group_name)

    return config


def _create_aks_resource_group(cloud_provider, workspace_name):
    resource_group_name = get_aks_workspace_resource_group_name(workspace_name)
    resource_group = _get_resource_group_by_name(
        resource_group_name, resource_client=None, provider_config=cloud_provider)
    if resource_group is None:
        resource_group = _create_resource_group(cloud_provider, resource_group_name)
    else:
        cli_logger.print("Resource group {} for workspace already exists. Skip creation.", resource_group.name)
    return resource_group.name


def _delete_aks_resource_group(cloud_provider, workspace_name):
    resource_group_name = get_aks_workspace_resource_group_name(workspace_name)
    _delete_resource_group(cloud_provider, resource_group_name)


def _create_iam_based_access_for_kubernetes(config: Dict[str, Any], namespace, cloud_provider):
    workspace_name = config["workspace_name"]
    current_step = 1
    total_steps = AZURE_KUBERNETES_IAM_ROLE_CREATION_NUM_STEPS

    with cli_logger.group(
            "Creating user assigned identities",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_iam_user_assigned_identities(
            cloud_provider, workspace_name)

    with cli_logger.group(
            "Creating user assigned identities role binding",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_iam_user_assigned_identities_role_binding(
            cloud_provider, workspace_name)

    with cli_logger.group(
            "Creating user assigned identities binding with Kubernetes",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_iam_user_assigned_identities_binding_with_kubernetes(
            config, cloud_provider, workspace_name, namespace)

    with cli_logger.group(
            "Associating Kubernetes service accounts with IAM",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _associate_kubernetes_service_accounts_with_iam(
            config, cloud_provider, workspace_name, namespace)


def _create_iam_user_assigned_identities(cloud_provider, workspace_name):
    current_step = 1
    total_steps = AZURE_KUBERNETES_HEAD_WORKER_FACED_NUM_STEPS

    with cli_logger.group(
            "Creating head user assigned identity",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_iam_user_assigned_identity(
            cloud_provider, workspace_name, AccountType.HEAD)

    with cli_logger.group(
            "Creating worker user assigned identity",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_iam_user_assigned_identity(
            cloud_provider, workspace_name, AccountType.WORKER)


def _create_iam_user_assigned_identity(cloud_provider, workspace_name, account_type: AccountType):
    resource_group_name = get_aks_workspace_resource_group_name(workspace_name)
    iam_user_assigned_identity_name = _get_iam_user_assigned_identity_name(
        workspace_name, account_type)
    _create_user_assigned_identity(
        cloud_provider,
        resource_group_name,
        iam_user_assigned_identity_name)


def _delete_iam_user_assigned_identities(cloud_provider, workspace_name):
    current_step = 1
    total_steps = AZURE_KUBERNETES_HEAD_WORKER_FACED_NUM_STEPS

    with cli_logger.group(
            "Deleting head user assigned identity",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_iam_user_assigned_identity(
            cloud_provider, workspace_name, AccountType.HEAD)

    with cli_logger.group(
            "Deleting worker user assigned identity",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_iam_user_assigned_identity(
            cloud_provider, workspace_name, AccountType.WORKER)


def _delete_iam_user_assigned_identity(cloud_provider, workspace_name, account_type: AccountType):
    resource_group_name = get_aks_workspace_resource_group_name(workspace_name)
    iam_user_assigned_identity_name = _get_iam_user_assigned_identity_name(workspace_name, account_type)
    _delete_user_assigned_identity(
        cloud_provider, resource_group_name, iam_user_assigned_identity_name)


def _get_iam_user_assigned_identity(cloud_provider, workspace_name, account_type: AccountType):
    resource_group_name = get_aks_workspace_resource_group_name(workspace_name)
    iam_user_assigned_identity_name = _get_iam_user_assigned_identity_name(workspace_name, account_type)

    cli_logger.verbose("Getting user assigned identity: {}...",
                       iam_user_assigned_identity_name)
    user_assigned_identity = _get_user_assigned_identity(
        cloud_provider, resource_group_name, iam_user_assigned_identity_name)
    if user_assigned_identity is None:
        cli_logger.verbose_error("Failed to get user assigned identity: {}.",
                                 iam_user_assigned_identity_name)
    else:
        cli_logger.verbose("Successfully got user assigned identity: {}.",
                           iam_user_assigned_identity_name)
    return user_assigned_identity


def _create_iam_user_assigned_identities_role_binding(cloud_provider, workspace_name):
    current_step = 1
    total_steps = AZURE_KUBERNETES_HEAD_WORKER_FACED_NUM_STEPS

    with cli_logger.group(
            "Creating head user assigned identity role binding",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_iam_user_assigned_identity_role_binding(
            cloud_provider, workspace_name, AccountType.HEAD)

    with cli_logger.group(
            "Creating worker user assigned identity role binding",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_iam_user_assigned_identity_role_binding(
            cloud_provider, workspace_name, AccountType.WORKER)


def _create_iam_user_assigned_identity_role_binding(
        cloud_provider, workspace_name, account_type: AccountType):
    resource_group_name = get_aks_workspace_resource_group_name(workspace_name)
    iam_user_assigned_identity_name = _get_iam_user_assigned_identity_name(
        workspace_name, account_type)
    role_assignment_name = get_iam_role_assignment_name_for_storage_blob_data_owner(
        cloud_provider, workspace_name, account_type
    )
    user_assigned_identity = _get_user_assigned_identity(
        cloud_provider, resource_group_name, iam_user_assigned_identity_name)
    if user_assigned_identity is None:
        raise RuntimeError("No user assigned identity {} found.".format(
            iam_user_assigned_identity_name))

    # Both head and worker use the same set of roles
    _create_role_assignment_for_storage_blob_data_owner(
        cloud_provider, resource_group_name, user_assigned_identity, role_assignment_name)


def _delete_iam_user_assigned_identities_role_binding(cloud_provider, workspace_name):
    current_step = 1
    total_steps = AZURE_KUBERNETES_HEAD_WORKER_FACED_NUM_STEPS

    with cli_logger.group(
            "Deleting head user assigned identity role binding",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_iam_user_assigned_identity_role_binding(
            cloud_provider, workspace_name, AccountType.HEAD)

    with cli_logger.group(
            "Deleting worker user assigned identity role binding",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_iam_user_assigned_identity_role_binding(
            cloud_provider, workspace_name, AccountType.WORKER)


def _delete_iam_user_assigned_identity_role_binding(
        cloud_provider, workspace_name, account_type: AccountType):
    resource_group_name = get_aks_workspace_resource_group_name(workspace_name)
    iam_user_assigned_identity_name = _get_iam_user_assigned_identity_name(
        workspace_name, account_type)
    role_assignment_name = get_iam_role_assignment_name_for_storage_blob_data_owner(
        cloud_provider, workspace_name, account_type
    )
    user_assigned_identity = _get_user_assigned_identity(
        cloud_provider, resource_group_name, iam_user_assigned_identity_name)
    if user_assigned_identity is None:
        cli_logger.print(log_prefix + "No user assigned identity {} found. Skip deletion.".format(
            iam_user_assigned_identity_name))
        return

    # Both head and worker use the same set of roles
    _delete_role_assignment_for_storage_blob_data_owner(
        cloud_provider, resource_group_name, role_assignment_name)


def _has_iam_user_assigned_identity_role_binding(
        cloud_provider, workspace_name, account_type: AccountType):
    resource_group_name = get_aks_workspace_resource_group_name(workspace_name)
    iam_user_assigned_identity_name = _get_iam_user_assigned_identity_name(
        workspace_name, account_type)
    role_assignment_name = get_iam_role_assignment_name_for_storage_blob_data_owner(
        cloud_provider, workspace_name, account_type
    )

    cli_logger.verbose("Getting user assigned identity role binding: {}...",
                       iam_user_assigned_identity_name)
    result = _get_role_assignment_for_storage_blob_data_owner(
        cloud_provider, resource_group_name, role_assignment_name
    )
    cli_logger.verbose("user assigned identity role binding: {}: {}",
                       iam_user_assigned_identity_name, result)
    return result


def _create_iam_user_assigned_identities_binding_with_kubernetes(
        config, cloud_provider, workspace_name, namespace):
    current_step = 1
    total_steps = AZURE_KUBERNETES_HEAD_WORKER_FACED_NUM_STEPS

    with cli_logger.group(
            "Creating head user assigned identity role binding with Kubernetes",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_iam_user_assigned_identity_binding_with_kubernetes(
            config, cloud_provider, workspace_name,
            namespace, AccountType.HEAD)

    with cli_logger.group(
            "Creating worker user assigned identity role binding with Kubernetes",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_iam_user_assigned_identity_binding_with_kubernetes(
            config, cloud_provider, workspace_name,
            namespace, AccountType.WORKER)


def _create_iam_user_assigned_identity_binding_with_kubernetes(
        config, cloud_provider, workspace_name,
        namespace, account_type: AccountType):
    provider_config = config["provider"]
    resource_group_name = get_aks_workspace_resource_group_name(workspace_name)
    iam_user_assigned_identity_name = _get_iam_user_assigned_identity_name(
        workspace_name, account_type)

    service_account_name = _get_service_account_name(provider_config, account_type)
    subject = _get_kubernetes_service_account_iam_subject(
        namespace, service_account_name
    )

    user_assigned_identity = _get_user_assigned_identity(
        cloud_provider, resource_group_name, iam_user_assigned_identity_name)
    if user_assigned_identity is None:
        raise RuntimeError("No user assigned identity {} found.".format(
            iam_user_assigned_identity_name))

    cli_logger.print("Creating user assigned identity role binding with Kubernetes: {} -> {}...".format(
        iam_user_assigned_identity_name, service_account_name))
    federated_identity_credential_name = _get_federated_identity_credential_name(
        workspace_name, account_type)
    oidc_issuer = _get_aks_oidc_issuer_url(cloud_provider)
    _create_federated_identity_credential(
        cloud_provider,
        resource_group_name, iam_user_assigned_identity_name,
        federated_identity_credential_name,
        issuer=oidc_issuer,
        subject=subject)

    cli_logger.print("Successfully created user assigned identity role binding with Kubernetes: {} -> {}.".format(
        iam_user_assigned_identity_name, service_account_name))


def _delete_iam_user_assigned_identities_binding_with_kubernetes(
        config, cloud_provider, workspace_name, namespace):
    current_step = 1
    total_steps = AZURE_KUBERNETES_HEAD_WORKER_FACED_NUM_STEPS

    with cli_logger.group(
            "Deleting head user assigned identity role binding with Kubernetes",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_iam_user_assigned_identity_binding_with_kubernetes(
            config, cloud_provider, workspace_name,
            namespace, AccountType.HEAD)

    with cli_logger.group(
            "Deleting worker user assigned identity role binding with Kubernetes",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_iam_user_assigned_identity_binding_with_kubernetes(
            config, cloud_provider, workspace_name,
            namespace, AccountType.WORKER)


def _delete_iam_user_assigned_identity_binding_with_kubernetes(
        config, cloud_provider, workspace_name,
        namespace, account_type: AccountType):
    provider_config = config["provider"]
    resource_group_name = get_aks_workspace_resource_group_name(workspace_name)
    iam_user_assigned_identity_name = _get_iam_user_assigned_identity_name(workspace_name, account_type)
    service_account_name = _get_service_account_name(provider_config, account_type)

    user_assigned_identity = _get_user_assigned_identity(
        cloud_provider, resource_group_name, iam_user_assigned_identity_name)
    if user_assigned_identity is None:
        cli_logger.print(log_prefix + "No user assigned identity {} found. Skip deletion.".format(
            iam_user_assigned_identity_name))
        return

    cli_logger.print(
        log_prefix + "Deleting user assigned identity role binding for Kubernetes: {} -> {}".format(
            iam_user_assigned_identity_name, service_account_name))
    federated_identity_credential_name = _get_federated_identity_credential_name(
        workspace_name, account_type)

    _delete_federated_identity_credential(
        cloud_provider,
        resource_group_name, iam_user_assigned_identity_name,
        federated_identity_credential_name)
    cli_logger.print(
        log_prefix + "Successfully deleted user assigned identity role binding for Kubernetes: {} -> {}.".format(
            iam_user_assigned_identity_name, service_account_name))


def _has_iam_user_assigned_identity_binding_with_kubernetes(
        config, cloud_provider, workspace_name,
        namespace, account_type: AccountType):
    provider_config = config["provider"]
    resource_group_name = get_aks_workspace_resource_group_name(workspace_name)
    iam_user_assigned_identity_name = _get_iam_user_assigned_identity_name(
        workspace_name, account_type)
    service_account_name = _get_service_account_name(provider_config, account_type)

    cli_logger.verbose("Getting user assigned identity binding with Kubernetes: {} -> {}...",
                       iam_user_assigned_identity_name, service_account_name)
    federated_identity_credential_name = _get_federated_identity_credential_name(
        workspace_name, account_type)
    federated_identity_credential = _get_federated_identity_credential(
        cloud_provider,
        resource_group_name, iam_user_assigned_identity_name,
        federated_identity_credential_name)
    result = True if federated_identity_credential else False
    cli_logger.verbose("Getting user assigned identity binding with Kubernetes: {} -> {}: {}",
                       iam_user_assigned_identity_name, service_account_name, result)
    return result


def _create_federated_identity_credential(
        cloud_provider, resource_group_name,
        user_assigned_identity_name, federated_identity_credential_name,
        issuer,
        subject):
    # Create a federated credential with the user assigned identity to the subject
    msi_client = _construct_manage_server_identity_client(cloud_provider)

    cli_logger.verbose(
        "Creating federated identity credential: {}->{} issued by {}...",
        user_assigned_identity_name, subject, issuer)
    # Create identity
    try:
        msi_client.federated_identity_credentials.create_or_update(
            resource_group_name=resource_group_name,
            resource_name=user_assigned_identity_name,
            federated_identity_credential_resource_name=federated_identity_credential_name,
            parameters={
                "issuer": issuer,
                "subject": subject,
                "audiences": ["api://AzureADTokenExchange"]
            }
        )
        time.sleep(20)
        cli_logger.verbose(
            "Successfully created federated identity credential: {}->{}.".format(
                user_assigned_identity_name, subject))
    except Exception as e:
        cli_logger.error(
            "Failed to create federated identity credential. {}", str(e))
        raise e


def _delete_federated_identity_credential(
        cloud_provider,
        resource_group_name, user_assigned_identity_name,
        federated_identity_credential_name):
    # Delete a federated credential with the user assigned identity to the subject
    msi_client = _construct_manage_server_identity_client(cloud_provider)
    federated_identity_credential = _get_federated_identity_credential(
        cloud_provider, resource_group_name,
        user_assigned_identity_name, federated_identity_credential_name,
        msi_client
    )

    if federated_identity_credential is None:
        cli_logger.print("The federated identity credential doesn't exist: {} -> {}.".format(
            user_assigned_identity_name, federated_identity_credential_name))
        return

    cli_logger.verbose("Deleting the federated identity credential: {}->{}...".format(
        user_assigned_identity_name, federated_identity_credential_name))
    try:
        msi_client.federated_identity_credentials.delete(
            resource_group_name=resource_group_name,
            resource_name=user_assigned_identity_name,
            federated_identity_credential_resource_name=federated_identity_credential_name
        )
        cli_logger.verbose("Successfully deleted the federated identity credential: {}->{}.".format(
            user_assigned_identity_name, federated_identity_credential_name))
    except Exception as e:
        cli_logger.error(
            "Failed to delete the federated identity credential: {}->{}. {}",
            user_assigned_identity_name, federated_identity_credential_name, str(e))
        raise e


def _get_federated_identity_credential(
        cloud_provider, resource_group_name,
        user_assigned_identity_name, federated_identity_credential_name, msi_client=None):
    # Get the federated credential with the user assigned identity to the subject
    if not msi_client:
        msi_client = _construct_manage_server_identity_client(cloud_provider)

    cli_logger.verbose("Getting the federated identity credential: {}->{}.".format(
        user_assigned_identity_name, federated_identity_credential_name))
    try:
        federated_identity_credential = msi_client.federated_identity_credentials.get(
            resource_group_name=resource_group_name,
            resource_name=user_assigned_identity_name,
            federated_identity_credential_resource_name=federated_identity_credential_name
        )
        cli_logger.verbose("Successfully get the federated identity credential: {}->.".format(
            user_assigned_identity_name, federated_identity_credential_name))
        return federated_identity_credential
    except ResourceNotFoundError as e:
        cli_logger.verbose_error(
            "Failed to get the federated identity credential: {}->{}. {}",
            user_assigned_identity_name, federated_identity_credential_name, str(e))
        return None


def _associate_kubernetes_service_accounts_with_iam(
        config, cloud_provider, workspace_name, namespace):
    # Patch head service account and worker service account
    current_step = 1
    total_steps = AZURE_KUBERNETES_HEAD_WORKER_FACED_NUM_STEPS

    with cli_logger.group(
            "Patching head service account with IAM",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _patch_service_account_with_iam(
            config, cloud_provider, workspace_name, namespace,
            AccountType.HEAD
        )

    with cli_logger.group(
            "Patching head service account with IAM",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _patch_service_account_with_iam(
            config, cloud_provider, workspace_name, namespace,
            AccountType.WORKER
        )


def _patch_service_account_with_iam(
        config, cloud_provider, workspace_name, namespace, account_type: AccountType):
    provider_config = config["provider"]

    service_account_name = _get_service_account_name(provider_config, account_type)
    iam_user_assigned_identity_name = _get_iam_user_assigned_identity_name(workspace_name, account_type)
    _patch_service_account_with_iam_user_assigned_identity(
        cloud_provider,
        workspace_name,
        namespace,
        service_account_name,
        iam_user_assigned_identity_name=iam_user_assigned_identity_name
    )


def _patch_service_account_with_iam_user_assigned_identity(
        cloud_provider, workspace_name,
        namespace, name, iam_user_assigned_identity_name):
    service_account = _get_service_account(namespace=namespace, name=name)
    if service_account is None:
        cli_logger.print(log_prefix + "No service account {} found. Skip patching.".format(name))
        return

    resource_group_name = get_aks_workspace_resource_group_name(workspace_name)
    user_assigned_identity = _get_user_assigned_identity(
        cloud_provider, resource_group_name, iam_user_assigned_identity_name)
    patch = {
        "metadata": {
            "annotations": {
                AZURE_KUBERNETES_ANNOTATION_NAME: AZURE_KUBERNETES_ANNOTATION_VALUE.format(
                    user_assigned_identity_client_id=user_assigned_identity.client_id)
            },
            "labels": {
                AZURE_KUBERNETES_WORKLOAD_IDENTITY_LABEL_NAME: "true"
            }
        }
    }

    cli_logger.print(log_prefix + "Patching service account {} with IAM...".format(name))
    core_api().patch_namespaced_service_account(name, namespace, patch)
    cli_logger.print(log_prefix + "Successfully patched service account {} with IAM.".format(name))


def _patch_service_account_without_iam_user_assigned_identity(namespace, name):
    service_account = _get_service_account(namespace=namespace, name=name)
    if service_account is None:
        cli_logger.print(log_prefix + "No service account {} found. Skip patching.".format(name))
        return

    patch = {
        "metadata": {
            "annotations": {
                AZURE_KUBERNETES_ANNOTATION_NAME: None
            },
            "labels": {
                AZURE_KUBERNETES_WORKLOAD_IDENTITY_LABEL_NAME: "false"
            }
        }
    }

    cli_logger.print(log_prefix + "Patching service account {} removing IAM...".format(name))
    core_api().patch_namespaced_service_account(name, namespace, patch)
    cli_logger.print(log_prefix + "Successfully patched service account {} removing IAM.".format(name))


def _delete_iam_based_access_for_kubernetes(config: Dict[str, Any], namespace, cloud_provider):
    workspace_name = config["workspace_name"]
    current_step = 1
    total_steps = AZURE_KUBERNETES_IAM_ROLE_CREATION_NUM_STEPS

    with cli_logger.group(
            "Dissociating Kubernetes service accounts with IAM",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _dissociate_kubernetes_service_accounts_with_iam(
            config, cloud_provider, workspace_name, namespace)

    with cli_logger.group(
            "Deleting user assigned identities binding with Kubernetes",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_iam_user_assigned_identities_binding_with_kubernetes(
            config, cloud_provider, workspace_name, namespace)

    with cli_logger.group(
            "Deleting user assigned identities role binding",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_iam_user_assigned_identities_role_binding(
            cloud_provider, workspace_name)

    with cli_logger.group(
            "Deleting user assigned identities",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_iam_user_assigned_identities(
            cloud_provider, workspace_name)


def _dissociate_kubernetes_service_accounts_with_iam(config, cloud_provider, workspace_name, namespace):
    # Patch head service account and worker service account
    provider_config = config["provider"]

    current_step = 1
    total_steps = AZURE_KUBERNETES_HEAD_WORKER_FACED_NUM_STEPS

    with cli_logger.group(
            "Patching head service account without IAM role",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        head_service_account_name = _get_head_service_account_name(provider_config)
        _patch_service_account_without_iam_user_assigned_identity(
            namespace,
            head_service_account_name
        )

    with cli_logger.group(
            "Patching head service account without IAM role",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        worker_service_account_name = _get_worker_service_account_name(provider_config)
        _patch_service_account_without_iam_user_assigned_identity(
            namespace,
            worker_service_account_name
        )


def _is_service_account_associated(
        config, cloud_provider, namespace, account_type: AccountType):
    provider_config = config["provider"]
    workspace_name = config["workspace_name"]
    service_account_name = _get_service_account_name(provider_config, account_type)
    iam_user_assigned_identity_name = _get_iam_user_assigned_identity_name(
        workspace_name, account_type)

    cli_logger.verbose("Getting Kubernetes service account associated: {} -> {}...",
                       service_account_name, iam_user_assigned_identity_name)
    result = _is_service_account_associated_with_iam(
        cloud_provider,
        workspace_name,
        namespace,
        service_account_name,
        iam_user_assigned_identity_name
    )
    cli_logger.verbose("Kubernetes service account associated: {} -> {}: {}.",
                       service_account_name, iam_user_assigned_identity_name, result)

    return result


def _is_service_account_associated_with_iam(
        cloud_provider, workspace_name,
        namespace, name, iam_user_assigned_identity_name):
    service_account = _get_service_account(namespace, name)
    if service_account is None:
        return False

    # Check annotation with the account id and role_name
    resource_group_name = get_aks_workspace_resource_group_name(workspace_name)

    user_assigned_identity = _get_user_assigned_identity(
        cloud_provider, resource_group_name, iam_user_assigned_identity_name)

    annotation_name = AZURE_KUBERNETES_ANNOTATION_NAME
    annotation_value = AZURE_KUBERNETES_ANNOTATION_VALUE.format(
        user_assigned_identity_client_id=user_assigned_identity.client_id)

    annotations = service_account.metadata.annotations
    if annotations is None:
        return False
    annotated_value = annotations.get(annotation_name)
    if annotated_value is None or annotation_value != annotated_value:
        return False
    return True


def check_existence_for_azure(config: Dict[str, Any], namespace, cloud_provider):
    workspace_name = config["workspace_name"]
    resource_group_name = get_aks_workspace_resource_group_name(workspace_name)
    managed_cloud_storage = _is_managed_cloud_storage(cloud_provider)

    existing_resources = 0
    target_resources = AZURE_KUBERNETES_TARGET_RESOURCES
    if managed_cloud_storage:
        target_resources += 1

    """
         Do the work - order of operation
         1. User assigned identities (+2)
         2. User assigned identities role binding (+2)
         2. Kubernetes service accounts IAM binding (+2)
         3. Kubernetes service accounts association(+2)
    """
    resource_group_existence = False
    cloud_storage_existence = False
    resource_group = _get_resource_group_by_name(
        resource_group_name, resource_client=None, provider_config=cloud_provider)
    if resource_group is not None:
        existing_resources += 1
        resource_group_existence = True

        if _get_iam_user_assigned_identity(
                cloud_provider, workspace_name, AccountType.HEAD) is not None:
            existing_resources += 1

        if _get_iam_user_assigned_identity(
                cloud_provider, workspace_name, AccountType.WORKER) is not None:
            existing_resources += 1

        if _has_iam_user_assigned_identity_role_binding(
                cloud_provider, workspace_name, AccountType.HEAD):
            existing_resources += 1

        if _has_iam_user_assigned_identity_role_binding(
                cloud_provider, workspace_name, AccountType.WORKER):
            existing_resources += 1

        if _has_iam_user_assigned_identity_binding_with_kubernetes(
                config, cloud_provider, workspace_name,
                namespace, AccountType.HEAD):
            existing_resources += 1

        if _has_iam_user_assigned_identity_binding_with_kubernetes(
                config, cloud_provider, workspace_name,
                namespace, AccountType.WORKER):
            existing_resources += 1

        if _is_service_account_associated(
                config, cloud_provider, namespace, AccountType.HEAD):
            existing_resources += 1

        if _is_service_account_associated(
                config, cloud_provider, namespace, AccountType.WORKER):
            existing_resources += 1

        if managed_cloud_storage:
            if _get_container_for_storage_account(
                    cloud_provider, workspace_name, resource_group_name) is not None:
                existing_resources += 1
                cloud_storage_existence = True

    if existing_resources == 0 or (
            existing_resources == 1 and resource_group_existence):
        return Existence.NOT_EXIST
    elif existing_resources == target_resources:
        return Existence.COMPLETED
    else:
        if existing_resources == 2 and cloud_storage_existence:
            return Existence.STORAGE_ONLY
        return Existence.IN_COMPLETED


def get_info_for_azure(config: Dict[str, Any], namespace, cloud_provider, info):
    workspace_name = config["workspace_name"]
    resource_group_name = get_aks_workspace_resource_group_name(workspace_name)

    head_iam_user_assigned_identity_name = _get_iam_user_assigned_identity_name(workspace_name, AccountType.HEAD)
    worker_iam_user_assigned_identity_name = _get_iam_user_assigned_identity_name(workspace_name, AccountType.WORKER)

    info[AZURE_KUBERNETES_HEAD_IAM_USER_ASSIGNED_IDENTITY_INFO] = head_iam_user_assigned_identity_name
    info[AZURE_KUBERNETES_WORKER_IAM_USER_ASSIGNED_IDENTITY_INFO] = worker_iam_user_assigned_identity_name

    managed_cloud_storage = _is_managed_cloud_storage(cloud_provider)
    if managed_cloud_storage:
        get_azure_managed_cloud_storage_info(
            config, cloud_provider, resource_group_name, info)


def with_azure_environment_variables(provider_config, config_dict: Dict[str, Any]):
    export_azure_cloud_storage_config(provider_config, config_dict)


def get_default_kubernetes_cloud_storage_for_azure(provider_config):
    return get_default_azure_cloud_storage(provider_config)
