from typing import Any, Dict

from cloudtik.core._private.cli_logger import cli_logger
from cloudtik.core._private.utils import _is_use_managed_cloud_storage, _is_managed_cloud_storage, \
    _is_managed_cloud_database, _is_use_managed_cloud_database
from cloudtik.core.workspace_provider import Existence
from cloudtik.providers._private._kubernetes import core_api, log_prefix
from cloudtik.providers._private._kubernetes.gcp_gke.utils import get_project_id, \
    AccountType, _get_iam_service_account_name, _get_iam_service_account_display_name, construct_container_client
from cloudtik.providers._private._kubernetes.utils import _get_head_service_account_name, \
    _get_worker_service_account_name, _get_service_account
from cloudtik.providers._private.gcp.config import _configure_managed_cloud_storage_from_workspace, \
    _create_managed_cloud_storage, _delete_managed_cloud_storage, get_managed_gcs_bucket, _create_service_account, \
    _delete_service_account, _get_service_account_by_id, _add_iam_role_binding, WORKER_SERVICE_ACCOUNT_ROLES, \
    _remove_iam_role_binding, _has_iam_role_binding, _add_service_account_iam_role_binding, \
    _remove_service_account_iam_role_binding, _has_service_account_iam_role_binding, _get_service_account_of_project, \
    get_gcp_managed_cloud_storage_info, _create_managed_cloud_database, _delete_managed_cloud_database, \
    _configure_managed_cloud_database_from_workspace, get_managed_database_instance, get_gcp_managed_cloud_database_info
from cloudtik.providers._private.gcp.utils import get_gcp_project, construct_iam_client, construct_crm_client, \
    get_service_account_email, export_gcp_cloud_storage_config, get_default_gcp_cloud_storage, \
    export_gcp_cloud_database_config, get_default_gcp_cloud_database

GCP_KUBERNETES_ANNOTATION_NAME = "iam.gke.io/gcp-service-account"
GCP_KUBERNETES_ANNOTATION_VALUE = "{service_account}@{project_id}.iam.gserviceaccount.com"

GCP_KUBERNETES_SERVICE_ACCOUNT_WORKLOAD_IDENTITY_ROLES = ["roles/iam.workloadIdentityUser"]
GCP_KUBERNETES_SERVICE_ACCOUNT_WORKLOAD_IDENTITY_MEMBER = "serviceAccount:{}.svc.id.goog[{}/{}]"

GCP_KUBERNETES_HEAD_IAM_SERVICE_ACCOUNT_INFO = "gcp.kubernetes.head.iam.service_account"
GCP_KUBERNETES_WORKER_IAM_SERVICE_ACCOUNT_INFO = "gcp.kubernetes.worker.iam.service_account"

GCP_KUBERNETES_NUM_CREATION_STEPS = 1
GCP_KUBERNETES_NUM_DELETION_STEPS = 1
GCP_KUBERNETES_NUM_UPDATE_STEPS = 0

GCP_KUBERNETES_IAM_ROLE_CREATION_NUM_STEPS = 4
GCP_KUBERNETES_IAM_ROLE_DELETION_NUM_STEPS = 4

GCP_KUBERNETES_HEAD_WORKER_FACED_NUM_STEPS = 2

GCP_KUBERNETES_TARGET_RESOURCES = 9


def _get_service_account_name(provider_config, account_type: AccountType):
    if account_type == AccountType.HEAD:
        return _get_head_service_account_name(provider_config)
    else:
        return _get_worker_service_account_name(provider_config)


def _get_kubernetes_service_account_iam_member_id(
        project_id, namespace, service_account_name):
    return GCP_KUBERNETES_SERVICE_ACCOUNT_WORKLOAD_IDENTITY_MEMBER.format(
        project_id, namespace, service_account_name
    )


def create_configurations_for_gcp(config: Dict[str, Any], namespace, cloud_provider):
    workspace_name = config["workspace_name"]
    managed_cloud_storage = _is_managed_cloud_storage(cloud_provider)
    managed_cloud_database = _is_managed_cloud_database(cloud_provider)

    current_step = 1
    total_steps = GCP_KUBERNETES_NUM_CREATION_STEPS
    if managed_cloud_storage:
        total_steps += 1
    if managed_cloud_database:
        total_steps += 1

    # Configure IAM based access for Kubernetes service accounts
    with cli_logger.group(
            "Creating IAM based access for Kubernetes",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_iam_based_access_for_kubernetes(config, namespace, cloud_provider)

    # Optionally, create managed cloud storage (GCS bucket) if user choose to
    if managed_cloud_storage:
        with cli_logger.group(
                "Creating GCS bucket",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _create_managed_cloud_storage(cloud_provider, workspace_name)

    if managed_cloud_database:
        with cli_logger.group(
                "Creating managed cloud database",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _create_managed_cloud_database_for_gke(
                cloud_provider, workspace_name)


def _get_gke_vpc_name(cloud_provider):
    # retrieve the network name from GKE cluster information
    container_client = construct_container_client(cloud_provider)

    project_id = cloud_provider["project_id"]
    region = cloud_provider.get("region")
    if not region:
        raise RuntimeError("GKE region is not specified in cloud provider section.")
    gke_cluster_name = cloud_provider.get("gke_cluster_name")
    if not gke_cluster_name:
        raise RuntimeError("GKE cluster name is not specified in cloud provider section.")

    name = "projects/{}/locations/{}/clusters/{}".format(
        project_id, region, gke_cluster_name
    )
    try:
        cluster = container_client.get_cluster(name=name)
        return cluster.network
    except Exception as e:
        cli_logger.warning(
            "Failed to get GKE cluster: {}. {}",
            gke_cluster_name, str(e))

    vpc_name = cloud_provider.get("gke_network")
    if not vpc_name:
        raise RuntimeError("GKE network name must specified "
                           "with gke_network key in cloud provider.")
    return vpc_name


def _create_managed_cloud_database_for_gke(
        cloud_provider, workspace_name):
    vpc_name = _get_gke_vpc_name(cloud_provider)
    _create_managed_cloud_database(
        cloud_provider, workspace_name,
        vpc_name
    )


def delete_configurations_for_gcp(
        config: Dict[str, Any], namespace, cloud_provider,
        delete_managed_storage: bool = False,
        delete_managed_database: bool = False):
    workspace_name = config["workspace_name"]
    managed_cloud_storage = _is_managed_cloud_storage(cloud_provider)
    managed_cloud_database = _is_managed_cloud_database(cloud_provider)

    current_step = 1
    total_steps = GCP_KUBERNETES_NUM_DELETION_STEPS
    if managed_cloud_storage and delete_managed_storage:
        total_steps += 1
    if managed_cloud_database and delete_managed_database:
        total_steps += 1

    # Delete in a reverse way of creating
    if managed_cloud_database and delete_managed_database:
        with cli_logger.group(
                "Deleting managed cloud database",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _delete_managed_cloud_database_for_gke(
                cloud_provider, workspace_name)

    if managed_cloud_storage and delete_managed_storage:
        with cli_logger.group(
                "Deleting GCS bucket",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _delete_managed_cloud_storage(cloud_provider, workspace_name)

    # Delete GCS IAM role based access for Kubernetes service accounts
    with cli_logger.group(
            "Deleting IAM role based access for Kubernetes",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_iam_based_access_for_kubernetes(config, namespace, cloud_provider)


def _delete_managed_cloud_database_for_gke(
        cloud_provider, workspace_name,
        delete_for_update: bool = False):
    vpc_name = _get_gke_vpc_name(cloud_provider)
    _delete_managed_cloud_database(
        cloud_provider, workspace_name,
        vpc_name, delete_for_update=delete_for_update)


def update_configurations_for_gcp(
        config: Dict[str, Any], namespace, cloud_provider,
        delete_managed_storage: bool = False,
        delete_managed_database: bool = False):
    workspace_name = config["workspace_name"]
    managed_cloud_storage = _is_managed_cloud_storage(cloud_provider)
    managed_cloud_database = _is_managed_cloud_database(cloud_provider)

    current_step = 1
    total_steps = GCP_KUBERNETES_NUM_UPDATE_STEPS
    if managed_cloud_storage or delete_managed_storage:
        total_steps += 1
    if managed_cloud_database or delete_managed_database:
        total_steps += 1

    if total_steps == 0:
        cli_logger.print("No configurations needed for update. Skip update.")
        return

    if managed_cloud_storage:
        with cli_logger.group(
                "Creating managed cloud storage...",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _create_managed_cloud_storage(cloud_provider, workspace_name)
    else:
        if delete_managed_storage:
            with cli_logger.group(
                    "Deleting managed cloud storage",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _delete_managed_cloud_storage(cloud_provider, workspace_name)

    if managed_cloud_database:
        with cli_logger.group(
                "Creating managed database",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _create_managed_cloud_database_for_gke(
                cloud_provider, workspace_name)
    else:
        if delete_managed_database:
            with cli_logger.group(
                    "Deleting managed database",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _delete_managed_cloud_database_for_gke(
                    cloud_provider, workspace_name, delete_for_update=True)


def configure_kubernetes_for_gcp(config: Dict[str, Any], namespace, cloud_provider):
    # Optionally, if user choose to use managed cloud storage (gcs bucket)
    # Configure the gcs bucket under cloud storage
    _configure_cloud_storage_for_gcp(config, cloud_provider)
    _configure_cloud_database_for_gcp(config, cloud_provider)


def _configure_cloud_storage_for_gcp(config: Dict[str, Any], cloud_provider):
    use_managed_cloud_storage = _is_use_managed_cloud_storage(cloud_provider)
    if use_managed_cloud_storage:
        _configure_managed_cloud_storage_from_workspace(config, cloud_provider)

    return config


def _configure_cloud_database_for_gcp(config: Dict[str, Any], cloud_provider):
    use_managed_cloud_database = _is_use_managed_cloud_database(cloud_provider)
    if use_managed_cloud_database:
        _configure_managed_cloud_database_from_workspace(config, cloud_provider)

    return config


def _create_iam_based_access_for_kubernetes(config: Dict[str, Any], namespace, cloud_provider):
    workspace_name = config["workspace_name"]
    current_step = 1
    total_steps = GCP_KUBERNETES_IAM_ROLE_CREATION_NUM_STEPS

    with cli_logger.group(
            "Creating IAM service accounts",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_iam_service_accounts(
            cloud_provider, workspace_name)

    with cli_logger.group(
            "Creating IAM service accounts role binding",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_iam_service_accounts_role_binding(
            cloud_provider, workspace_name)

    with cli_logger.group(
            "Creating IAM service accounts binding with Kubernetes",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_iam_service_accounts_binding_with_kubernetes(
            config, cloud_provider, workspace_name, namespace)

    with cli_logger.group(
            "Associating Kubernetes service accounts with IAM",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _associate_kubernetes_service_accounts_with_iam(
            config, cloud_provider, workspace_name, namespace)


def _create_iam_service_accounts(cloud_provider, workspace_name):
    iam = construct_iam_client(cloud_provider)

    current_step = 1
    total_steps = GCP_KUBERNETES_HEAD_WORKER_FACED_NUM_STEPS

    with cli_logger.group(
            "Creating head IAM service account",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_iam_service_account(
            cloud_provider, workspace_name, iam, AccountType.HEAD)

    with cli_logger.group(
            "Creating worker IAM service account",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_iam_service_account(
            cloud_provider, workspace_name, iam, AccountType.WORKER)


def _create_iam_service_account(cloud_provider, workspace_name, iam, account_type: AccountType):
    iam_service_account_name = _get_iam_service_account_name(workspace_name, account_type)
    service_account_config = {
        "displayName": _get_iam_service_account_display_name(workspace_name, account_type),
    }

    cli_logger.print("Creating IAM service account: {}...".format(iam_service_account_name))
    _create_service_account(
        cloud_provider, iam_service_account_name,
        service_account_config, iam)
    cli_logger.print("Successfully created IAM service account: {}...".format(iam_service_account_name))


def _delete_iam_service_accounts(cloud_provider, workspace_name):
    iam = construct_iam_client(cloud_provider)

    current_step = 1
    total_steps = GCP_KUBERNETES_HEAD_WORKER_FACED_NUM_STEPS

    with cli_logger.group(
            "Deleting head IAM service account",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_iam_service_account(
            cloud_provider, workspace_name, iam, AccountType.HEAD)

    with cli_logger.group(
            "Deleting worker IAM service account",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_iam_service_account(
            cloud_provider, workspace_name, iam, AccountType.WORKER)


def _delete_iam_service_account(cloud_provider, workspace_name, iam, account_type: AccountType):
    iam_service_account_name = _get_iam_service_account_name(workspace_name, account_type)
    _delete_service_account(
        cloud_provider, iam_service_account_name, iam)


def _get_iam_service_account(cloud_provider, workspace_name, iam, account_type: AccountType):
    iam_service_account_name = _get_iam_service_account_name(workspace_name, account_type)

    cli_logger.verbose("Getting IAM service account: {}...",
                       iam_service_account_name)
    sa = _get_service_account_by_id(cloud_provider, iam_service_account_name, iam)
    if sa is None:
        cli_logger.verbose_error("Failed to get IAM service account: {}.",
                           iam_service_account_name)
    else:
        cli_logger.verbose("Successfully got IAM service account: {}.",
                           iam_service_account_name)
    return sa


def _create_iam_service_accounts_role_binding(cloud_provider, workspace_name):
    iam = construct_iam_client(cloud_provider)
    crm = construct_crm_client(cloud_provider)

    current_step = 1
    total_steps = GCP_KUBERNETES_HEAD_WORKER_FACED_NUM_STEPS

    with cli_logger.group(
            "Creating head IAM service account role binding",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_iam_service_account_role_binding(
            cloud_provider, workspace_name, iam, crm, AccountType.HEAD)

    with cli_logger.group(
            "Creating worker IAM service account role binding",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_iam_service_account_role_binding(
            cloud_provider, workspace_name, iam, crm, AccountType.WORKER)


def _create_iam_service_account_role_binding(
        cloud_provider, workspace_name, iam, crm, account_type: AccountType):
    project_id = get_project_id(cloud_provider)
    service_account_name = _get_iam_service_account_name(workspace_name, account_type)
    service_account_email = get_service_account_email(
        project_id=project_id, account_id=service_account_name)

    sa = _get_service_account_of_project(project_id, service_account_email, iam)
    if sa is None:
        raise RuntimeError("No IAM service account {} found.".format(
            service_account_name))

    cli_logger.print("Creating IAM service account role binding: {}...".format(service_account_name))
    # Both head and worker use the same set of roles
    _add_iam_role_binding(
        project_id, service_account_email, WORKER_SERVICE_ACCOUNT_ROLES, crm)

    cli_logger.print("Successfully created IAM service account role binding: {}...".format(service_account_name))


def _delete_iam_service_accounts_role_binding(cloud_provider, workspace_name):
    iam = construct_iam_client(cloud_provider)
    crm = construct_crm_client(cloud_provider)

    current_step = 1
    total_steps = GCP_KUBERNETES_HEAD_WORKER_FACED_NUM_STEPS

    with cli_logger.group(
            "Deleting head IAM service account role binding",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_iam_service_account_role_binding(
            cloud_provider, workspace_name, iam, crm, AccountType.HEAD)

    with cli_logger.group(
            "Deleting worker IAM service account role binding",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_iam_service_account_role_binding(
            cloud_provider, workspace_name, iam, crm, AccountType.WORKER)


def _delete_iam_service_account_role_binding(
        cloud_provider, workspace_name, iam, crm, account_type: AccountType):
    project_id = get_project_id(cloud_provider)
    service_account_name = _get_iam_service_account_name(workspace_name, account_type)
    service_account_email = get_service_account_email(
        project_id=project_id, account_id=service_account_name)

    sa = _get_service_account_of_project(project_id, service_account_email, iam)
    if sa is None:
        cli_logger.print(log_prefix + "No IAM service account {} found. Skip deletion.".format(
            service_account_name))
        return

    cli_logger.print(log_prefix + "Deleting IAM service account role binding: {}".format(
        service_account_name))

    # Both head and worker use the same set of roles
    _remove_iam_role_binding(
        project_id, service_account_email, WORKER_SERVICE_ACCOUNT_ROLES, crm)

    cli_logger.print(log_prefix + "Successfully deleted IAM service account role binding: {}".format(
        service_account_name))


def _has_iam_service_account_role_binding(
        cloud_provider, workspace_name, crm, account_type: AccountType):
    project_id = get_project_id(cloud_provider)
    service_account_name = _get_iam_service_account_name(workspace_name, account_type)
    service_account_email = get_service_account_email(
        project_id=project_id, account_id=service_account_name)

    cli_logger.verbose("Getting IAM service account role binding: {}...",
                       service_account_name)
    result = _has_iam_role_binding(
        project_id, service_account_email, WORKER_SERVICE_ACCOUNT_ROLES, crm)
    cli_logger.verbose("IAM service account role binding: {}: {}",
                       service_account_name, result)
    return result


def _create_iam_service_accounts_binding_with_kubernetes(
        config, cloud_provider, workspace_name, namespace):
    iam = construct_iam_client(cloud_provider)

    current_step = 1
    total_steps = GCP_KUBERNETES_HEAD_WORKER_FACED_NUM_STEPS

    with cli_logger.group(
            "Creating head IAM service account role binding with Kubernetes",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_iam_service_account_binding_with_kubernetes(
            config, cloud_provider, workspace_name, namespace,
            iam, AccountType.HEAD)

    with cli_logger.group(
            "Creating worker IAM service account role binding with Kubernetes",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_iam_service_account_binding_with_kubernetes(
            config, cloud_provider, workspace_name, namespace,
            iam, AccountType.WORKER)


def _create_iam_service_account_binding_with_kubernetes(
        config, cloud_provider, workspace_name, namespace,
        iam, account_type: AccountType):
    provider_config = config["provider"]
    project_id = get_project_id(cloud_provider)

    iam_service_account_name = _get_iam_service_account_name(workspace_name, account_type)
    iam_service_account_email = get_service_account_email(
        project_id=project_id, account_id=iam_service_account_name)

    service_account_name = _get_service_account_name(provider_config, account_type)
    member_id = _get_kubernetes_service_account_iam_member_id(
        project_id, namespace, service_account_name
    )

    sa = _get_service_account_of_project(project_id, iam_service_account_email, iam)
    if sa is None:
        raise RuntimeError("No IAM service account {} found.".format(
            iam_service_account_name))

    cli_logger.print("Creating IAM service account role binding with Kubernetes: {} -> {}...".format(
        iam_service_account_name, service_account_name))

    _add_service_account_iam_role_binding(
        project_id, iam_service_account_email,
        GCP_KUBERNETES_SERVICE_ACCOUNT_WORKLOAD_IDENTITY_ROLES,
        member_id=member_id,
        iam=iam)

    cli_logger.print("Successfully created IAM service account role binding with Kubernetes: {} -> {}.".format(
        iam_service_account_name, service_account_name))


def _delete_iam_service_accounts_binding_with_kubernetes(
        config, cloud_provider, workspace_name, namespace):
    iam = construct_iam_client(cloud_provider)

    current_step = 1
    total_steps = GCP_KUBERNETES_HEAD_WORKER_FACED_NUM_STEPS

    with cli_logger.group(
            "Deleting head IAM service account role binding with Kubernetes",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_iam_service_account_binding_with_kubernetes(
            config, cloud_provider, workspace_name, namespace,
            iam, AccountType.HEAD)

    with cli_logger.group(
            "Deleting worker IAM service account role binding with Kubernetes",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_iam_service_account_binding_with_kubernetes(
            config, cloud_provider, workspace_name, namespace,
            iam, AccountType.WORKER)


def _delete_iam_service_account_binding_with_kubernetes(
        config, cloud_provider, workspace_name, namespace,
        iam, account_type: AccountType):
    provider_config = config["provider"]
    project_id = get_project_id(cloud_provider)

    iam_service_account_name = _get_iam_service_account_name(workspace_name, account_type)
    iam_service_account_email = get_service_account_email(
        project_id=project_id, account_id=iam_service_account_name)

    service_account_name = _get_service_account_name(provider_config, account_type)
    member_id = _get_kubernetes_service_account_iam_member_id(
        project_id, namespace, service_account_name
    )

    sa = _get_service_account_of_project(project_id, iam_service_account_email, iam)
    if sa is None:
        cli_logger.print(log_prefix + "No IAM service account {} found. Skip deletion.".format(
            iam_service_account_name))
        return

    cli_logger.print(
        log_prefix + "Deleting IAM service account role binding for Kubernetes: {} -> {}".format(
            iam_service_account_name, service_account_name))
    _remove_service_account_iam_role_binding(
        project_id, iam_service_account_email,
        GCP_KUBERNETES_SERVICE_ACCOUNT_WORKLOAD_IDENTITY_ROLES,
        member_id=member_id,
        iam=iam)
    cli_logger.print(
        log_prefix + "Successfully deleted IAM service account role binding for Kubernetes: {} -> {}.".format(
            iam_service_account_name, service_account_name))


def _has_iam_service_account_binding_with_kubernetes(
        config, cloud_provider, workspace_name, namespace,
        iam, account_type: AccountType):
    provider_config = config["provider"]
    project_id = get_project_id(cloud_provider)

    iam_service_account_name = _get_iam_service_account_name(workspace_name, account_type)
    iam_service_account_email = get_service_account_email(
        project_id=project_id, account_id=iam_service_account_name)

    service_account_name = _get_service_account_name(provider_config, account_type)
    member_id = _get_kubernetes_service_account_iam_member_id(
        project_id, namespace, service_account_name
    )

    cli_logger.verbose("Getting IAM service account binding with Kubernetes: {} -> {}...",
                       iam_service_account_name, service_account_name)
    result = _has_service_account_iam_role_binding(
        project_id, iam_service_account_email,
        GCP_KUBERNETES_SERVICE_ACCOUNT_WORKLOAD_IDENTITY_ROLES,
        member_id=member_id,
        iam=iam)
    cli_logger.verbose("Getting IAM service account binding with Kubernetes: {} -> {}: {}",
                       iam_service_account_name, service_account_name, result)
    return result


def _associate_kubernetes_service_accounts_with_iam(
        config, cloud_provider, workspace_name, namespace):
    # Patch head service account and worker service account
    current_step = 1
    total_steps = GCP_KUBERNETES_HEAD_WORKER_FACED_NUM_STEPS

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
    project_id = get_project_id(cloud_provider)

    service_account_name = _get_service_account_name(provider_config, account_type)
    iam_service_account_name = _get_iam_service_account_name(workspace_name, account_type)
    _patch_service_account_with_iam_service_account(
        namespace,
        service_account_name,
        project_id=project_id,
        iam_service_account_name=iam_service_account_name
    )


def _patch_service_account_with_iam_service_account(namespace, name, project_id, iam_service_account_name):
    service_account = _get_service_account(namespace=namespace, name=name)
    if service_account is None:
        cli_logger.print(log_prefix + "No service account {} found. Skip patching.".format(name))
        return

    patch = {
        "metadata": {
            "annotations": {
                GCP_KUBERNETES_ANNOTATION_NAME: GCP_KUBERNETES_ANNOTATION_VALUE.format(
                    project_id=project_id, service_account=iam_service_account_name)
            }
        }
    }

    cli_logger.print(log_prefix + "Patching service account {} with IAM...".format(name))
    core_api().patch_namespaced_service_account(name, namespace, patch)
    cli_logger.print(log_prefix + "Successfully patched service account {} with IAM.".format(name))


def _patch_service_account_without_iam_service_account(namespace, name):
    service_account = _get_service_account(namespace=namespace, name=name)
    if service_account is None:
        cli_logger.print(log_prefix + "No service account {} found. Skip patching.".format(name))
        return

    patch = {
        "metadata": {
            "annotations": {
                GCP_KUBERNETES_ANNOTATION_NAME: None
            }
        }
    }

    cli_logger.print(log_prefix + "Patching service account {} removing IAM...".format(name))
    core_api().patch_namespaced_service_account(name, namespace, patch)
    cli_logger.print(log_prefix + "Successfully patched service account {} removing IAM.".format(name))


def _delete_iam_based_access_for_kubernetes(config: Dict[str, Any], namespace, cloud_provider):
    workspace_name = config["workspace_name"]
    current_step = 1
    total_steps = GCP_KUBERNETES_IAM_ROLE_CREATION_NUM_STEPS

    with cli_logger.group(
            "Dissociating Kubernetes service accounts with IAM",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _dissociate_kubernetes_service_accounts_with_iam(
            config, cloud_provider, workspace_name, namespace)

    with cli_logger.group(
            "Deleting IAM service accounts binding with Kubernetes",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_iam_service_accounts_binding_with_kubernetes(
            config, cloud_provider, workspace_name, namespace)

    with cli_logger.group(
            "Deleting IAM service accounts role binding",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_iam_service_accounts_role_binding(
            cloud_provider, workspace_name)

    with cli_logger.group(
            "Deleting IAM service accounts",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_iam_service_accounts(
            cloud_provider, workspace_name)


def _dissociate_kubernetes_service_accounts_with_iam(config, cloud_provider, workspace_name, namespace):
    # Patch head service account and worker service account
    provider_config = config["provider"]

    current_step = 1
    total_steps = GCP_KUBERNETES_HEAD_WORKER_FACED_NUM_STEPS

    with cli_logger.group(
            "Patching head service account without IAM role",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        head_service_account_name = _get_head_service_account_name(provider_config)
        _patch_service_account_without_iam_service_account(
            namespace,
            head_service_account_name
        )

    with cli_logger.group(
            "Patching head service account without IAM role",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        worker_service_account_name = _get_worker_service_account_name(provider_config)
        _patch_service_account_without_iam_service_account(
            namespace,
            worker_service_account_name
        )


def _is_service_account_associated(config, cloud_provider, namespace, account_type: AccountType):
    provider_config = config["provider"]
    workspace_name = config["workspace_name"]
    service_account_name = _get_service_account_name(provider_config, account_type)
    iam_service_account_name = _get_iam_service_account_name(workspace_name, account_type)

    cli_logger.verbose("Getting Kubernetes service account associated: {} -> {}...",
                       service_account_name, iam_service_account_name)
    result = _is_service_account_associated_with_iam(
        cloud_provider,
        namespace,
        service_account_name,
        iam_service_account_name
    )
    cli_logger.verbose("Kubernetes service account associated: {} -> {}: {}.",
                       service_account_name, iam_service_account_name, result)

    return result


def _is_service_account_associated_with_iam(
        cloud_provider, namespace, name, iam_service_account_name):
    service_account = _get_service_account(namespace, name)
    if service_account is None:
        return False

    # Check annotation with the account id and role_name
    project_id = get_project_id(cloud_provider)

    annotation_name = GCP_KUBERNETES_ANNOTATION_NAME
    annotation_value = GCP_KUBERNETES_ANNOTATION_VALUE.format(
                    project_id=project_id, service_account=iam_service_account_name)

    annotations = service_account.metadata.annotations
    if annotations is None:
        return False
    annotated_value = annotations.get(annotation_name)
    if annotated_value is None or annotation_value != annotated_value:
        return False
    return True


def check_existence_for_gcp(config: Dict[str, Any], namespace, cloud_provider):
    iam = construct_iam_client(cloud_provider)
    crm = construct_crm_client(cloud_provider)

    workspace_name = config["workspace_name"]
    managed_cloud_storage = _is_managed_cloud_storage(cloud_provider)
    managed_cloud_database = _is_managed_cloud_database(cloud_provider)

    existing_resources = 0
    target_resources = GCP_KUBERNETES_TARGET_RESOURCES
    if managed_cloud_storage:
        target_resources += 1
    if managed_cloud_database:
        target_resources += 1

    """
         Do the work - order of operation
         1. IAM service accounts (+2)
         2. IAM role binding (+2)
         2. Kubernetes service accounts IAM binding (+2)
         3. Kubernetes service accounts association(+2)
    """
    project_existence = False
    cloud_storage_existence = False
    cloud_database_existence = False
    project = get_gcp_project(cloud_provider, get_project_id(cloud_provider))
    if project is not None:
        existing_resources += 1
        project_existence = True

        if _get_iam_service_account(
                cloud_provider, workspace_name, iam, AccountType.HEAD) is not None:
            existing_resources += 1

        if _get_iam_service_account(
                cloud_provider, workspace_name, iam, AccountType.WORKER) is not None:
            existing_resources += 1

        if _has_iam_service_account_role_binding(
                cloud_provider, workspace_name,crm, AccountType.HEAD):
            existing_resources += 1

        if _has_iam_service_account_role_binding(
                cloud_provider, workspace_name, crm, AccountType.WORKER):
            existing_resources += 1

        if _has_iam_service_account_binding_with_kubernetes(
                config, cloud_provider, workspace_name,
                namespace, iam, AccountType.HEAD):
            existing_resources += 1

        if _has_iam_service_account_binding_with_kubernetes(
                config, cloud_provider, workspace_name,
                namespace, iam, AccountType.WORKER):
            existing_resources += 1

        if _is_service_account_associated(
                config, cloud_provider, namespace, AccountType.HEAD):
            existing_resources += 1

        if _is_service_account_associated(
                config, cloud_provider, namespace, AccountType.WORKER):
            existing_resources += 1

        if managed_cloud_storage:
            if get_managed_gcs_bucket(cloud_provider, workspace_name) is not None:
                existing_resources += 1
                cloud_storage_existence = True

        if managed_cloud_database:
            if get_managed_database_instance(
                    cloud_provider, workspace_name) is not None:
                existing_resources += 1
                cloud_database_existence = True

    if existing_resources == 0 or (
            existing_resources == 1 and project_existence):
        return Existence.NOT_EXIST
    elif existing_resources == target_resources:
        return Existence.COMPLETED
    else:
        skipped_resources = 1
        if existing_resources == skipped_resources + 1 and cloud_storage_existence:
            return Existence.STORAGE_ONLY
        elif existing_resources == skipped_resources + 1 and cloud_database_existence:
            return Existence.DATABASE_ONLY
        elif existing_resources == skipped_resources + 2 and cloud_storage_existence \
                and cloud_database_existence:
            return Existence.STORAGE_AND_DATABASE_ONLY
        return Existence.IN_COMPLETED


def get_info_for_gcp(config: Dict[str, Any], namespace, cloud_provider, info):
    workspace_name = config["workspace_name"]
    head_iam_service_account_name = _get_iam_service_account_name(workspace_name, AccountType.HEAD)
    worker_iam_service_account_name = _get_iam_service_account_name(workspace_name, AccountType.WORKER)

    info[GCP_KUBERNETES_HEAD_IAM_SERVICE_ACCOUNT_INFO] = head_iam_service_account_name
    info[GCP_KUBERNETES_WORKER_IAM_SERVICE_ACCOUNT_INFO] = worker_iam_service_account_name

    managed_cloud_storage = _is_managed_cloud_storage(cloud_provider)
    if managed_cloud_storage:
        get_gcp_managed_cloud_storage_info(config, cloud_provider, info)

    managed_cloud_database = _is_managed_cloud_database(cloud_provider)
    if managed_cloud_database:
        get_gcp_managed_cloud_database_info(config, cloud_provider, info)


def with_gcp_environment_variables(provider_config, config_dict: Dict[str, Any]):
    export_gcp_cloud_storage_config(provider_config, config_dict)
    export_gcp_cloud_database_config(provider_config, config_dict)


def get_default_kubernetes_cloud_storage_for_gcp(provider_config):
    return get_default_gcp_cloud_storage(provider_config)


def get_default_kubernetes_cloud_database_for_gcp(provider_config):
    return get_default_gcp_cloud_database(provider_config)
