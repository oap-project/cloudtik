import json
import urllib
import urllib.parse
import urllib.request
import urllib.error
from typing import Any, Dict

from cloudtik.core._private.cli_logger import cli_logger
from cloudtik.core._private.utils import _is_use_managed_cloud_storage, _is_managed_cloud_storage
from cloudtik.providers._private._kubernetes import core_api, log_prefix
from cloudtik.providers._private._kubernetes.aws_eks.eks_utils import get_root_ca_cert_thumbprint
from cloudtik.providers._private._kubernetes.utils import _get_head_service_account_name, \
    _get_worker_service_account_name
from cloudtik.providers._private.aws.config import _configure_managed_cloud_storage_from_workspace, \
    _create_managed_cloud_storage, _delete_managed_cloud_storage, _get_iam_role
from cloudtik.providers._private.aws.utils import _make_client, _make_resource, get_current_account_id

HTTP_DEFAULT_PORT = 80
HTTPS_DEFAULT_PORT = 443

HTTPS_URL_PREFIX = "https://"

AWS_KUBERNETES_IAM_ROLE_NAME_TEMPLATE = "cloudtik-eks-{}-role"

AWS_KUBERNETES_NUM_CREATION_STEPS = 2
AWS_KUBERNETES_NUM_DELETION_STEPS = 2
AWS_KUBERNETES_IAM_ROLE_NUM_STEPS = 3


def create_configurations_for_aws(config: Dict[str, Any], namespace, cloud_provider):
    workspace_name = config["workspace_name"]
    managed_cloud_storage = _is_managed_cloud_storage(cloud_provider)

    current_step = 1
    total_steps = AWS_KUBERNETES_NUM_CREATION_STEPS
    if managed_cloud_storage:
        total_steps += 1

    # Configure S3 IAM role based access for Kubernetes service accounts
    with cli_logger.group(
            "Creating IAM role based access for Kubernetes",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        _create_iam_role_based_access_for_kubernetes(config, namespace, cloud_provider)

    # Optionally, create managed cloud storage (s3 bucket) if user choose to
    if managed_cloud_storage:
        with cli_logger.group(
                "Creating S3 bucket",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _create_managed_cloud_storage(cloud_provider, workspace_name)


def delete_configurations_for_aws(config: Dict[str, Any], namespace, cloud_provider,
                                  delete_managed_storage: bool = False):
    workspace_name = config["workspace_name"]
    managed_cloud_storage = _is_managed_cloud_storage(cloud_provider)

    current_step = 1
    total_steps = AWS_KUBERNETES_NUM_DELETION_STEPS
    if managed_cloud_storage and delete_managed_storage:
        total_steps += 1

    # Delete in a reverse way of creating
    if managed_cloud_storage and delete_managed_storage:
        with cli_logger.group(
                "Deleting S3 bucket",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _delete_managed_cloud_storage(cloud_provider, workspace_name)


def configure_kubernetes_for_aws(config: Dict[str, Any], namespace, cloud_provider):
    # Optionally, if user choose to use managed cloud storage (s3 bucket)
    # Configure the s3 bucket under aws_s3_storage
    _configure_cloud_storage_for_aws(config, cloud_provider)


def _configure_cloud_storage_for_aws(config: Dict[str, Any], cloud_provider):
    use_managed_cloud_storage = _is_use_managed_cloud_storage(cloud_provider)
    if use_managed_cloud_storage:
        _configure_managed_cloud_storage_from_workspace(config, cloud_provider)

    return config


def _create_iam_role_based_access_for_kubernetes(config: Dict[str, Any], namespace, cloud_provider):
    # 1. Create an IAM OIDC provider for your cluster
    # 2. Create an IAM role and attach an IAM policy to it with the permissions that your service accounts need
    #    We recommend creating separate roles for each unique collection of permissions that pods need.
    # 3. Associate an IAM role with a service account
    workspace_name = config["workspace_name"]

    current_step = 1
    total_steps = AWS_KUBERNETES_IAM_ROLE_NUM_STEPS

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
    if "eks_cluster_name" not in cloud_provider:
        raise ValueError("Must specify 'eks_cluster_name' in cloud provider for AWS.")

    eks_cluster_name = cloud_provider["eks_cluster_name"]
    oidc_provider_url = _get_eks_cluster_oidc_identity_issuer(cloud_provider, eks_cluster_name)
    client_id = "sts.amazonaws.com"
    thumbprint = get_oidc_provider_thumbprint(oidc_provider_url)

    cli_logger.print("Creating OIDC Identity Provider for: {}...", eks_cluster_name)

    iam_client = _make_client("iam", cloud_provider)
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
    if "eks_cluster_name" not in cloud_provider:
        raise ValueError("Must specify 'eks_cluster_name' in cloud provider for AWS.")

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
    cli_logger.print("Successfully IAM role and policy for: {}.", eks_cluster_name)


def _create_iam_role_and_policy(
        cloud_provider, role_name, account_id, oidc_provider, namespace):
    iam = _make_resource("iam", cloud_provider)
    policy_doc = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Federated": "arn:aws:iam::{}:oidc-provider/{}".format(account_id, oidc_provider)
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
    total_steps = AWS_KUBERNETES_IAM_ROLE_NUM_STEPS

    with cli_logger.group(
            "Patching head service account with IAM role",
            _numbered=("[]", current_step, total_steps)):
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
        worker_service_account_name = _get_worker_service_account_name(provider_config)
        _patch_service_account_with_iam_role(
            namespace,
            worker_service_account_name,
            account_id=account_id,
            role_name=role_name
        )


def _patch_service_account_with_iam_role(namespace, name, account_id, role_name):
    patch = {
        "metadata": {
            "annotations": {
                "eks.amazonaws.com/role-arn": "arn:aws:iam::{}:role/{}".format(account_id, role_name)
            }
        }
    }

    cli_logger.print(log_prefix + "Patching service account {} with IAM role...".format(name))
    core_api().patch_namespaced_service_account(name, namespace, patch)
    cli_logger.print(log_prefix + "Successfully service account {} with IAM role.".format(name))
