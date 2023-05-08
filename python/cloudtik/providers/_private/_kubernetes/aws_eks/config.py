import json
import urllib
import urllib.parse
import urllib.request
import urllib.error
from typing import Any, Dict

import botocore

from cloudtik.core._private.cli_logger import cli_logger
from cloudtik.core._private.utils import _is_use_managed_cloud_storage, _is_managed_cloud_storage, \
    _is_managed_cloud_database, _is_use_managed_cloud_database
from cloudtik.core.workspace_provider import Existence
from cloudtik.providers._private._kubernetes import core_api, log_prefix
from cloudtik.providers._private._kubernetes.aws_eks.utils import get_root_ca_cert_thumbprint
from cloudtik.providers._private._kubernetes.utils import _get_head_service_account_name, \
    _get_worker_service_account_name, _get_service_account
from cloudtik.providers._private.aws.config import _configure_managed_cloud_storage_from_workspace, \
    _create_managed_cloud_storage, _delete_managed_cloud_storage, _get_iam_role, _delete_iam_role, \
    get_managed_s3_bucket, get_aws_managed_cloud_storage_info, get_managed_database_instance, \
    _configure_managed_cloud_database_from_workspace, _create_managed_cloud_database, _create_security_group, \
    _create_default_intra_cluster_inbound_rules, _update_inbound_rules, _get_security_group, \
    _delete_managed_cloud_database, _delete_security_group, get_aws_managed_cloud_database_info
from cloudtik.providers._private.aws.utils import _make_resource_client, _make_resource, get_current_account_id, \
    handle_boto_error, _make_client, export_aws_s3_storage_config, get_default_aws_cloud_storage, \
    export_aws_database_config, get_default_aws_cloud_database

HTTP_DEFAULT_PORT = 80
HTTPS_DEFAULT_PORT = 443

HTTPS_URL_PREFIX = "https://"

AWS_KUBERNETES_IAM_ROLE_NAME_TEMPLATE = "cloudtik-eks-{}-role"
AWS_KUBERNETES_SECURITY_GROUP_TEMPLATE = "cloudtik-eks-{}-sg"

AWS_KUBERNETES_OPEN_ID_IDENTITY_PROVIDER_ARN = "arn:aws:iam::{}:oidc-provider/{}"
AWS_KUBERNETES_ANNOTATION_NAME = "eks.amazonaws.com/role-arn"
AWS_KUBERNETES_ANNOTATION_VALUE = "arn:aws:iam::{}:role/{}"

AWS_KUBERNETES_IAM_ROLE_NAME_INFO = "aws.kubernetes.iam.role"

AWS_KUBERNETES_NUM_CREATION_STEPS = 1
AWS_KUBERNETES_NUM_DELETION_STEPS = 1
AWS_KUBERNETES_NUM_UPDATE_STEPS = 0

AWS_KUBERNETES_IAM_ROLE_CREATION_NUM_STEPS = 3
AWS_KUBERNETES_IAM_ROLE_DELETION_NUM_STEPS = 2

AWS_KUBERNETES_ASSOCIATE_CREATION_NUM_STEPS = 2
AWS_KUBERNETES_ASSOCIATE_DELETION_NUM_STEPS = 2

AWS_KUBERNETES_TARGET_RESOURCES = 4


def _check_eks_cluster_name(cloud_provider):
    if "eks_cluster_name" not in cloud_provider:
        raise ValueError("Must specify 'eks_cluster_name' in cloud provider for AWS.")


def create_configurations_for_aws(config: Dict[str, Any], namespace, cloud_provider):
    workspace_name = config["workspace_name"]
    managed_cloud_storage = _is_managed_cloud_storage(cloud_provider)
    managed_cloud_database = _is_managed_cloud_database(cloud_provider)

    current_step = 1
    total_steps = AWS_KUBERNETES_NUM_CREATION_STEPS
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

    # Optionally, create managed cloud storage (s3 bucket) if user choose to
    if managed_cloud_storage:
        with cli_logger.group(
                "Creating S3 bucket",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _create_managed_cloud_storage(cloud_provider, workspace_name)

    if managed_cloud_database:
        with cli_logger.group(
                "Creating managed cloud database",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _create_managed_cloud_database_for_eks(
                cloud_provider, workspace_name)


def _create_managed_cloud_database_for_eks(cloud_provider, workspace_name):
    current_step = 1
    total_steps = 2

    vpc_id, subnet_ids = get_eks_vpc_and_subnet_ids(cloud_provider)

    with cli_logger.group(
            "Creating database security group",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _create_database_security_group(
            cloud_provider, workspace_name, vpc_id)

    with cli_logger.group(
            "Creating managed database instance",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        security_group_id = _get_database_security_group_id(
            cloud_provider, workspace_name, vpc_id)
        _create_managed_cloud_database(
            cloud_provider, workspace_name,
            subnet_ids, security_group_id
        )


def _get_database_security_group_id(cloud_provider, workspace_name, vpc_id):
    group_name = AWS_KUBERNETES_SECURITY_GROUP_TEMPLATE.format(workspace_name)
    security_group = _get_security_group(
        cloud_provider, vpc_id, group_name)
    if security_group is None:
        return None
    return security_group.id


def _create_database_security_group(cloud_provider, workspace_name, vpc_id):
    current_step = 1
    total_steps = 2

    with cli_logger.group(
            "Creating security group for VPC",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        group_name = AWS_KUBERNETES_SECURITY_GROUP_TEMPLATE.format(workspace_name)
        security_group = _create_security_group(cloud_provider, vpc_id, group_name)

    with cli_logger.group(
            "Configuring rules for security group",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _add_security_group_rules(cloud_provider, security_group)

    return security_group


def _add_security_group_rules(cloud_provider, security_group):
    cli_logger.print("Updating rules for security group: {}...".format(
        security_group.id))
    security_group_ids = {security_group.id}
    ip_permissions = _create_default_intra_cluster_inbound_rules(security_group_ids)
    _update_inbound_rules(security_group, ip_permissions)
    cli_logger.print("Successfully updated rules for security group.")


def get_eks_vpc_and_subnet_ids(cloud_provider):
    eks_cluster_name = cloud_provider["eks_cluster_name"]
    eks_client = _make_client("eks", cloud_provider)

    eks_cluster_info = eks_client.describe_cluster(
        name=eks_cluster_name
    )

    eks_cluster = eks_cluster_info["cluster"]
    if "resourcesVpcConfig" not in eks_cluster:
        raise RuntimeError("No VPC config found for EKS cluster: {}".format(
            eks_cluster_name))

    vpc_config = eks_cluster["resourcesVpcConfig"]
    if "subnetIds" not in vpc_config or "vpcId" not in vpc_config:
        raise RuntimeError("No vpc or subnets information found for EKS cluster: {}".format(
            eks_cluster_name))

    return vpc_config["vpcId"], vpc_config["subnetIds"]


def delete_configurations_for_aws(config: Dict[str, Any], namespace, cloud_provider,
                                  delete_managed_storage: bool = False,
                                  delete_managed_database: bool = False):
    workspace_name = config["workspace_name"]
    managed_cloud_storage = _is_managed_cloud_storage(cloud_provider)
    managed_cloud_database = _is_managed_cloud_database(cloud_provider)

    current_step = 1
    total_steps = AWS_KUBERNETES_NUM_DELETION_STEPS
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
            _delete_managed_cloud_database_for_eks(cloud_provider, workspace_name)

    if managed_cloud_storage and delete_managed_storage:
        with cli_logger.group(
                "Deleting S3 bucket",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _delete_managed_cloud_storage(cloud_provider, workspace_name)

    # Delete S3 IAM role based access for Kubernetes service accounts
    with cli_logger.group(
            "Deleting IAM based access for Kubernetes",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_iam_based_access_for_kubernetes(config, namespace, cloud_provider)


def _delete_managed_cloud_database_for_eks(cloud_provider, workspace_name):
    current_step = 1
    total_steps = 2

    vpc_id, _ = get_eks_vpc_and_subnet_ids(cloud_provider)

    with cli_logger.group(
            "Deleting managed database instance",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_managed_cloud_database(
            cloud_provider, workspace_name)

    with cli_logger.group(
            "Creating database security group",
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        _delete_database_security_group(
            cloud_provider, workspace_name, vpc_id)


def _delete_database_security_group(
        cloud_provider, workspace_name, vpc_id):
    security_group_name = AWS_KUBERNETES_SECURITY_GROUP_TEMPLATE.format(workspace_name)
    _delete_security_group(
        cloud_provider, vpc_id, security_group_name
    )


def update_configurations_for_aws(
        config: Dict[str, Any], namespace, cloud_provider,
        delete_managed_storage: bool = False,
        delete_managed_database: bool = False):
    workspace_name = config["workspace_name"]
    managed_cloud_storage = _is_managed_cloud_storage(cloud_provider)
    managed_cloud_database = _is_managed_cloud_database(cloud_provider)

    current_step = 1
    total_steps = AWS_KUBERNETES_NUM_UPDATE_STEPS
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
            _create_managed_cloud_database_for_eks(
                cloud_provider, workspace_name)
    else:
        if delete_managed_database:
            with cli_logger.group(
                    "Deleting managed database",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _delete_managed_cloud_database_for_eks(
                    cloud_provider, workspace_name)


def configure_kubernetes_for_aws(config: Dict[str, Any], namespace, cloud_provider):
    # Optionally, if user choose to use managed cloud storage (s3 bucket)
    # Configure the s3 bucket under cloud storage
    _configure_cloud_storage_for_aws(config, cloud_provider)
    _configure_cloud_database_for_aws(config, cloud_provider)


def _configure_cloud_storage_for_aws(config: Dict[str, Any], cloud_provider):
    use_managed_cloud_storage = _is_use_managed_cloud_storage(cloud_provider)
    if use_managed_cloud_storage:
        _configure_managed_cloud_storage_from_workspace(config, cloud_provider)

    return config


def _configure_cloud_database_for_aws(config: Dict[str, Any], cloud_provider):
    use_managed_cloud_database = _is_use_managed_cloud_database(cloud_provider)
    if use_managed_cloud_database:
        _configure_managed_cloud_database_from_workspace(config, cloud_provider)

    return config


def _create_iam_based_access_for_kubernetes(config: Dict[str, Any], namespace, cloud_provider):
    # 1. Create an IAM OIDC provider for your cluster
    # 2. Create an IAM role and attach an IAM policy to it with the permissions that your service accounts need
    #    We recommend creating separate roles for each unique collection of permissions that pods need.
    # 3. Associate an IAM role with a service account
    workspace_name = config["workspace_name"]

    current_step = 1
    total_steps = AWS_KUBERNETES_IAM_ROLE_CREATION_NUM_STEPS

    with cli_logger.group(
            "Creating OIDC Identity Provider",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_oidc_identity_provider(cloud_provider, workspace_name)

    with cli_logger.group(
            "Creating IAM role and policy",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_oidc_iam_role_and_policy(cloud_provider, namespace)

    with cli_logger.group(
            "Associating IAM role with Kubernetes service account",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _associate_oidc_iam_role_with_service_account(config, cloud_provider, namespace)


def _create_oidc_identity_provider(cloud_provider, workspace_name):
    _check_eks_cluster_name(cloud_provider)
    eks_cluster_name = cloud_provider["eks_cluster_name"]
    oidc_provider_url = _get_eks_cluster_oidc_identity_issuer(cloud_provider, eks_cluster_name)

    oidc_identity_provider = _get_oidc_identity_provider(cloud_provider, oidc_provider_url)
    if oidc_identity_provider is not None:
        cli_logger.print("Open ID Identity Provider already exists for : {}. Skip creation.", eks_cluster_name)
        return

    client_id = "sts.amazonaws.com"
    thumbprint = get_oidc_provider_thumbprint(oidc_provider_url)

    cli_logger.print("Creating OIDC Identity Provider for: {}...", eks_cluster_name)

    iam_client = _make_resource_client("iam", cloud_provider)
    response = iam_client.create_open_id_connect_provider(
        Url=oidc_provider_url,
        ClientIDList=[
            client_id,
        ],
        ThumbprintList=[
            thumbprint,
        ],
        Tags=[
            {
                'Key': 'Name',
                'Value': "cloudtik-eks-{}".format(workspace_name)
            },
        ]
    )

    cli_logger.print("Successfully created OIDC Identity Provider for: {}.", eks_cluster_name)
    return response['OpenIDConnectProviderArn']


def get_oidc_provider_thumbprint(oidc_provider_url):
    server_name, server_port = _get_oidc_provider_server_name_and_port(oidc_provider_url)
    thumbprint = get_root_ca_cert_thumbprint(server_name, server_port)
    return thumbprint


def _get_oidc_provider_server_name_and_port(oidc_provider_url):
    # Retrieve a json from oidc_provider_url + "/.well-known/openid-configuration"
    openid_config_url = oidc_provider_url + "/.well-known/openid-configuration"
    try:
        response = urllib.request.urlopen(openid_config_url, timeout=10)
        content = response.read()
    except urllib.error.HTTPError as e:
        cli_logger.error("Failed to retrieve open id configuration. {}", str(e))
        raise e

    openid_configuration = json.loads(content)
    if "jwks_uri" not in openid_configuration:
        raise RuntimeError("No jwks_uri property found for openid configuration.")

    jwks_uri = openid_configuration["jwks_uri"]
    server_name, server_port = get_server_name_and_port(jwks_uri)
    return server_name, server_port


def get_server_name_and_port(uri):
    url_components = urllib.parse.urlparse(uri)
    netloc = url_components.netloc
    if netloc is None:
        raise RuntimeError("No server component found for OIDC provider URL.")

    server_parts = netloc.split(":")
    if len(server_parts) <= 1:
        if url_components.scheme == "https":
            return netloc, HTTPS_DEFAULT_PORT
        return netloc, HTTP_DEFAULT_PORT

    return netloc[0], int(netloc[1])


def _get_eks_cluster_oidc_identity_issuer(cloud_provider, eks_cluster_name):
    eks_client = _make_client("eks", cloud_provider)

    eks_cluster_info = eks_client.describe_cluster(
        name=eks_cluster_name
    )

    eks_cluster = eks_cluster_info["cluster"]
    if "identity" not in eks_cluster or \
            "oidc" not in eks_cluster["identity"] or \
            "issuer" not in eks_cluster["identity"]["oidc"]:
        raise RuntimeError("No OIDC provider found for EKS cluster: {}".format(eks_cluster_name))

    return eks_cluster["identity"]["oidc"]["issuer"]


def get_oidc_provider_from_url(oidc_provider_url):
    if oidc_provider_url.startswith(HTTPS_URL_PREFIX):
        oidc_provider = oidc_provider_url[len(HTTPS_URL_PREFIX):]
    else:
        oidc_provider = oidc_provider_url
    return oidc_provider


def get_oidc_provider_role_name(eks_cluster_name, namespace):
    return AWS_KUBERNETES_IAM_ROLE_NAME_TEMPLATE.format(namespace)


def _create_oidc_iam_role_and_policy(cloud_provider, namespace):
    _check_eks_cluster_name(cloud_provider)
    eks_cluster_name = cloud_provider["eks_cluster_name"]
    oidc_provider_url = _get_eks_cluster_oidc_identity_issuer(cloud_provider, eks_cluster_name)

    account_id = get_current_account_id(cloud_provider)
    oidc_provider = get_oidc_provider_from_url(oidc_provider_url)
    role_name = get_oidc_provider_role_name(eks_cluster_name, namespace)

    cli_logger.print("Creating IAM role and policy for: {}...", eks_cluster_name)
    _create_iam_role_and_policy(
        cloud_provider, role_name,
        account_id=account_id,
        oidc_provider=oidc_provider,
        namespace=namespace)
    cli_logger.print("Successfully created IAM role and policy for: {}.", eks_cluster_name)


def _create_iam_role_and_policy(
        cloud_provider, role_name, account_id, oidc_provider, namespace):
    iam = _make_resource("iam", cloud_provider)
    policy_doc = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Federated": AWS_KUBERNETES_OPEN_ID_IDENTITY_PROVIDER_ARN.format(account_id, oidc_provider)
                },
                "Action": "sts:AssumeRoleWithWebIdentity",
                "Condition": {
                    "StringEquals": {
                        "{}:aud".format(oidc_provider): "sts.amazonaws.com",
                    },
                    "StringLike": {
                        "{}:sub".format(oidc_provider): "system:serviceaccount:{}:*".format(namespace)
                    }
                }
            }
        ]
    }

    attach_policy_arns = [
        "arn:aws:iam::aws:policy/AmazonS3FullAccess"
    ]

    iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(policy_doc))
    role = _get_iam_role(role_name, cloud_provider)
    assert role is not None, "Failed to create role"

    for policy_arn in attach_policy_arns:
        role.attach_policy(PolicyArn=policy_arn)


def _associate_oidc_iam_role_with_service_account(config, cloud_provider, namespace):
    # Patch head service account and worker service account
    eks_cluster_name = cloud_provider["eks_cluster_name"]
    account_id = get_current_account_id(cloud_provider)
    role_name = get_oidc_provider_role_name(eks_cluster_name, namespace)
    provider_config = config["provider"]

    current_step = 1
    total_steps = AWS_KUBERNETES_ASSOCIATE_CREATION_NUM_STEPS

    with cli_logger.group(
            "Patching head service account with IAM role",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        head_service_account_name = _get_head_service_account_name(provider_config)
        _patch_service_account_with_iam_role(
            namespace,
            head_service_account_name,
            account_id=account_id,
            role_name=role_name
        )

    with cli_logger.group(
            "Patching head service account with IAM role",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        worker_service_account_name = _get_worker_service_account_name(provider_config)
        _patch_service_account_with_iam_role(
            namespace,
            worker_service_account_name,
            account_id=account_id,
            role_name=role_name
        )


def _patch_service_account_with_iam_role(namespace, name, account_id, role_name):
    service_account = _get_service_account(namespace=namespace, name=name)
    if service_account is None:
        cli_logger.print(log_prefix + "No service account {} found. Skip patching.".format(name))
        return

    patch = {
        "metadata": {
            "annotations": {
                AWS_KUBERNETES_ANNOTATION_NAME: AWS_KUBERNETES_ANNOTATION_VALUE.format(account_id, role_name)
            }
        }
    }

    cli_logger.print(log_prefix + "Patching service account {} with IAM role...".format(name))
    core_api().patch_namespaced_service_account(name, namespace, patch)
    cli_logger.print(log_prefix + "Successfully patched service account {} with IAM role.".format(name))


def _patch_service_account_without_iam_role(namespace, name):
    service_account = _get_service_account(namespace=namespace, name=name)
    if service_account is None:
        cli_logger.print(log_prefix + "No service account {} found. Skip patching.".format(name))
        return

    patch = {
        "metadata": {
            "annotations": {
                AWS_KUBERNETES_ANNOTATION_NAME: None
            }
        }
    }

    cli_logger.print(log_prefix + "Patching service account {} removing IAM role...".format(name))
    core_api().patch_namespaced_service_account(name, namespace, patch)
    cli_logger.print(log_prefix + "Successfully patched service account {} removing IAM role.".format(name))


def get_oidc_identity_provider(cloud_provider):
    _check_eks_cluster_name(cloud_provider)
    eks_cluster_name = cloud_provider["eks_cluster_name"]
    oidc_provider_url = _get_eks_cluster_oidc_identity_issuer(cloud_provider, eks_cluster_name)
    oidc_identity_provider = _get_oidc_identity_provider(cloud_provider, oidc_provider_url)
    if oidc_identity_provider is None:
        return None

    return oidc_identity_provider


def _get_oidc_identity_provider(cloud_provider, oidc_provider_url):
    account_id = get_current_account_id(cloud_provider)
    oidc_provider = get_oidc_provider_from_url(oidc_provider_url)
    oidc_identity_provider_arn = AWS_KUBERNETES_OPEN_ID_IDENTITY_PROVIDER_ARN.format(account_id, oidc_provider)
    iam_client = _make_resource_client("iam", cloud_provider)

    try:
        cli_logger.verbose("Getting Open ID identity provider for: {}.", oidc_provider_url)
        response = iam_client.get_open_id_connect_provider(
            OpenIDConnectProviderArn=oidc_identity_provider_arn
        )
        cli_logger.verbose("Successfully got Open ID identity provider for: {}.", oidc_provider_url)
    except botocore.exceptions.ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "NoSuchEntity":
            return None
        else:
            handle_boto_error(
                exc, "Failed to get Open ID identity provider for {} from AWS.",
                oidc_provider_url)
            raise exc

    return response


def _delete_iam_based_access_for_kubernetes(config: Dict[str, Any], namespace, cloud_provider):
    # 1. Dissociate an IAM role with service accounts
    # 2. Delete the IAM role
    current_step = 1
    total_steps = AWS_KUBERNETES_IAM_ROLE_DELETION_NUM_STEPS

    with cli_logger.group(
            "Dissociating IAM role with Kubernetes service account",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _dissociate_oidc_iam_role_with_service_account(config, cloud_provider, namespace)

    with cli_logger.group(
            "Deleting IAM role and policy",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _delete_oidc_iam_role_and_policy(cloud_provider, namespace)


def _dissociate_oidc_iam_role_with_service_account(config, cloud_provider, namespace):
    # Patch head service account and worker service account
    eks_cluster_name = cloud_provider["eks_cluster_name"]
    provider_config = config["provider"]

    current_step = 1
    total_steps = AWS_KUBERNETES_ASSOCIATE_DELETION_NUM_STEPS

    with cli_logger.group(
            "Patching head service account without IAM role",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        head_service_account_name = _get_head_service_account_name(provider_config)
        _patch_service_account_without_iam_role(
            namespace,
            head_service_account_name
        )

    with cli_logger.group(
            "Patching head service account without IAM role",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        worker_service_account_name = _get_worker_service_account_name(provider_config)
        _patch_service_account_without_iam_role(
            namespace,
            worker_service_account_name
        )


def _delete_oidc_iam_role_and_policy(cloud_provider, namespace):
    _check_eks_cluster_name(cloud_provider)
    eks_cluster_name = cloud_provider["eks_cluster_name"]
    role_name = get_oidc_provider_role_name(eks_cluster_name, namespace)

    cli_logger.print("Deleting IAM role and policy for: {}...", eks_cluster_name)
    _delete_iam_role(
        cloud_provider, role_name)
    cli_logger.print("Successfully deleted IAM role and policy for: {}.", eks_cluster_name)


def _get_oidc_iam_role(cloud_provider, namespace):
    _check_eks_cluster_name(cloud_provider)
    eks_cluster_name = cloud_provider["eks_cluster_name"]
    role_name = get_oidc_provider_role_name(eks_cluster_name, namespace)

    cli_logger.verbose("Getting Open ID IAM role: {}.", role_name)
    role = _get_iam_role(role_name, cloud_provider)
    if role is None:
        cli_logger.verbose("Open ID IAM role with the name doesn't exist: {}.", role_name)
    else:
        cli_logger.verbose("Successfully get Open ID IAM role: {}.", role_name)
    return role


def _is_head_service_account_associated(config, cloud_provider, namespace):
    provider_config = config["provider"]
    head_service_account_name = _get_head_service_account_name(provider_config)
    associated = _is_service_account_associated(
        cloud_provider,
        namespace,
        head_service_account_name,
    )
    cli_logger.verbose("Is service account {} associated with IAM role: {}.", head_service_account_name, associated)
    return associated


def _is_worker_service_account_associated(config, cloud_provider, namespace):
    provider_config = config["provider"]
    worker_service_account_name = _get_worker_service_account_name(provider_config)
    associated = _is_service_account_associated(
        cloud_provider,
        namespace,
        worker_service_account_name,
    )
    cli_logger.verbose("Is service account {} associated with IAM role: {}.", worker_service_account_name, associated)
    return associated


def _is_service_account_associated(cloud_provider, namespace, name):
    service_account = _get_service_account(namespace, name)
    if service_account is None:
        return False

    # Check annotation with the account id and role_name
    eks_cluster_name = cloud_provider["eks_cluster_name"]
    account_id = get_current_account_id(cloud_provider)
    role_name = get_oidc_provider_role_name(eks_cluster_name, namespace)

    annotation_name = AWS_KUBERNETES_ANNOTATION_NAME
    annotation_value = AWS_KUBERNETES_ANNOTATION_VALUE.format(account_id, role_name)

    annotations = service_account.metadata.annotations
    if annotations is None:
        return False
    annotated_value = annotations.get(annotation_name)
    if annotated_value is None or annotation_value != annotated_value:
        return False
    return True


def check_existence_for_aws(config: Dict[str, Any], namespace, cloud_provider):
    workspace_name = config["workspace_name"]
    managed_cloud_storage = _is_managed_cloud_storage(cloud_provider)
    managed_cloud_database = _is_managed_cloud_database(cloud_provider)

    existing_resources = 0
    target_resources = AWS_KUBERNETES_TARGET_RESOURCES
    if managed_cloud_storage:
        target_resources += 1
    if managed_cloud_database:
        target_resources += 1

    """
         Do the work - order of operation
         1. Open ID Identity provider
         2. IAM role
         3. head service association
         4. worker service association
    """
    oidc_identity_provider_existence = False
    oidc_identity_provider = get_oidc_identity_provider(cloud_provider)
    if oidc_identity_provider is not None:
        existing_resources += 1
        oidc_identity_provider_existence = True

        # Only if the oidc identity provider exist
        # Will the IAM role will exist
        if _get_oidc_iam_role(cloud_provider, namespace) is not None:
            existing_resources += 1

        if _is_head_service_account_associated(config, cloud_provider, namespace):
            existing_resources += 1

        if _is_worker_service_account_associated(config, cloud_provider, namespace):
            existing_resources += 1

    cloud_storage_existence = False
    if managed_cloud_storage:
        if get_managed_s3_bucket(cloud_provider, workspace_name) is not None:
            existing_resources += 1
            cloud_storage_existence = True

    cloud_database_existence = False
    if managed_cloud_database:
        if get_managed_database_instance(cloud_provider, workspace_name) is not None:
            existing_resources += 1
            cloud_database_existence = True

    if existing_resources == 0 or (
            existing_resources == 1 and oidc_identity_provider_existence):
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


def get_info_for_aws(config: Dict[str, Any], namespace, cloud_provider, info):
    _check_eks_cluster_name(cloud_provider)
    eks_cluster_name = cloud_provider["eks_cluster_name"]
    iam_role_name = get_oidc_provider_role_name(eks_cluster_name, namespace)
    info[AWS_KUBERNETES_IAM_ROLE_NAME_INFO] = iam_role_name

    managed_cloud_storage = _is_managed_cloud_storage(cloud_provider)
    if managed_cloud_storage:
        get_aws_managed_cloud_storage_info(config, cloud_provider, info)

    managed_cloud_database = _is_managed_cloud_database(cloud_provider)
    if managed_cloud_database:
        get_aws_managed_cloud_database_info(config, cloud_provider, info)


def with_aws_environment_variables(provider_config, config_dict: Dict[str, Any]):
    export_aws_s3_storage_config(provider_config, config_dict)
    export_aws_database_config(provider_config, config_dict)
    config_dict["AWS_WEB_IDENTITY"] = True


def get_default_kubernetes_cloud_storage_for_aws(provider_config):
    return get_default_aws_cloud_storage(provider_config)


def get_default_kubernetes_cloud_database_for_aws(provider_config):
    return get_default_aws_cloud_database(provider_config)
