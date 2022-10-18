from typing import Any, Callable, Dict

from azure.common.credentials import get_cli_profile
from azure.identity import AzureCliCredential, DefaultAzureCredential, ClientSecretCredential, CertificateCredential,\
    ManagedIdentityCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.storage import StorageManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.msi import ManagedServiceIdentityClient
from azure.mgmt.authorization import AuthorizationManagementClient

from cloudtik.core._private.utils import get_storage_config_for_update
from cloudtik.providers._private._azure.azure_identity_credential_adapter import AzureIdentityCredentialAdapter


def get_azure_sdk_function(client: Any, function_name: str) -> Callable:
    """Retrieve a callable function from Azure SDK client object.

       Newer versions of the various client SDKs renamed function names to
       have a begin_ prefix. This function supports both the old and new
       versions of the SDK by first trying the old name and falling back to
       the prefixed new name.
    """
    func = getattr(client, function_name,
                   getattr(client, f"begin_{function_name}"))
    if func is None:
        raise AttributeError(
            "'{obj}' object has no {func} or begin_{func} attribute".format(
                obj={client.__name__}, func=function_name))
    return func


def get_credential(provider_config):
    managed_identity_client_id = provider_config.get("managed_identity_client_id")
    if managed_identity_client_id is None:
        # No managed identity
        credential = DefaultAzureCredential(
            exclude_managed_identity_credential=True,
            exclude_shared_token_cache_credential=True)
    else:
        credential = DefaultAzureCredential(
            exclude_shared_token_cache_credential=True,
            managed_identity_client_id=managed_identity_client_id
        )
    return credential


def get_client_credential(provider_config):
    azure_credentials = provider_config.get("azure_credentials")
    if azure_credentials is not None:
        # azure credentials configured in the configuration file
        assert ("type" in azure_credentials), \
            "azure_credentials cluster yaml field missing 'type' field."
        assert ("credentials" in azure_credentials), \
            "azure_credentials cluster yaml field missing 'credentials' field."

        cred_type = azure_credentials["type"]
        credentials_fields = azure_credentials["credentials"]

        if cred_type == "ClientSecret":
            return ClientSecretCredential(**credentials_fields)
        elif cred_type == "ManagedIdentity":
            return ManagedIdentityCredential(**credentials_fields)

    credential = DefaultAzureCredential(
        exclude_managed_identity_credential=True,
        exclude_shared_token_cache_credential=True,
        exclude_visual_studio_code_credential=True,
        exclude_powershell_credential=True
    )
    return credential


def construct_resource_client(config):
    return _construct_resource_client(config["provider"])


def _construct_resource_client(provider_config):
    subscription_id = provider_config.get("subscription_id")
    if subscription_id is None:
        subscription_id = get_cli_profile().get_subscription_id()
    credential = get_client_credential(provider_config)
    resource_client = ResourceManagementClient(credential, subscription_id)
    return resource_client


def construct_storage_client(config):
    return _construct_storage_client(config["provider"])


def _construct_storage_client(provider_config):
    subscription_id = provider_config.get("subscription_id")
    if subscription_id is None:
        subscription_id = get_cli_profile().get_subscription_id()
    credential = get_client_credential(provider_config)
    storage_client = StorageManagementClient(credential, subscription_id)
    return storage_client


def construct_network_client(config):
    return _construct_network_client(config["provider"])


def _construct_network_client(provider_config):
    subscription_id = provider_config.get("subscription_id")
    if subscription_id is None:
        subscription_id = get_cli_profile().get_subscription_id()
    credential = get_client_credential(provider_config)
    network_client = NetworkManagementClient(credential, subscription_id)
    return network_client


def construct_compute_client(config):
    return _construct_compute_client(config["provider"])


def _construct_compute_client(provider_config):
    subscription_id = provider_config.get("subscription_id")
    if subscription_id is None:
        subscription_id = get_cli_profile().get_subscription_id()
    credential = get_client_credential(provider_config)
    compute_client = ComputeManagementClient(credential, subscription_id)
    return compute_client


def construct_manage_server_identity_client(config):
    return _construct_manage_server_identity_client(config["provider"])


def _construct_manage_server_identity_client(provider_config):
    subscription_id = provider_config.get("subscription_id")
    if subscription_id is None:
        subscription_id = get_cli_profile().get_subscription_id()
    credential = get_client_credential(provider_config)
    # It showed that we no longer need to wrapper. Will fail with wrapper: no attribute get_token
    # wrapped_credential = AzureIdentityCredentialAdapter(credential)
    msi_client = ManagedServiceIdentityClient(credential, subscription_id)
    return msi_client


def construct_authorization_client(config):
    return _construct_authorization_client(config["provider"])


def _construct_authorization_client(provider_config):
    subscription_id = provider_config.get("subscription_id")
    if subscription_id is None:
        subscription_id = get_cli_profile().get_subscription_id()
    credential = get_client_credential(provider_config)
    wrapped_credential = AzureIdentityCredentialAdapter(credential)
    authorization_client = AuthorizationManagementClient(
        credentials=wrapped_credential,
        subscription_id=subscription_id,
        api_version="2018-01-01-preview"
    )
    return authorization_client


def get_azure_cloud_storage_config(provider_config: Dict[str, Any]):
    if "storage" in provider_config and "azure_cloud_storage" in provider_config["storage"]:
        return provider_config["storage"]["azure_cloud_storage"]

    return None


def get_azure_cloud_storage_config_for_update(provider_config: Dict[str, Any]):
    storage_config = get_storage_config_for_update(provider_config)
    if "azure_cloud_storage" not in storage_config:
        storage_config["azure_cloud_storage"] = {}
    return storage_config["azure_cloud_storage"]


def export_azure_cloud_storage_config(provider_config, config_dict: Dict[str, Any]):
    cloud_storage = get_azure_cloud_storage_config(provider_config)
    if cloud_storage is None:
        return
    config_dict["AZURE_CLOUD_STORAGE"] = True

    azure_storage_type = cloud_storage.get("azure.storage.type")
    if azure_storage_type:
        config_dict["AZURE_STORAGE_TYPE"] = azure_storage_type

    azure_storage_account = cloud_storage.get("azure.storage.account")
    if azure_storage_account:
        config_dict["AZURE_STORAGE_ACCOUNT"] = azure_storage_account

    azure_container = cloud_storage.get(
        "azure.container")
    if azure_container:
        config_dict["AZURE_CONTAINER"] = azure_container

    azure_account_key = cloud_storage.get(
        "azure.account.key")
    if azure_account_key:
        config_dict["AZURE_ACCOUNT_KEY"] = azure_account_key


def _get_node_info(node):
    node_info = {"node_id": node["name"].split("-")[-1],
                 "instance_type": node["vm_size"],
                 "private_ip": node["internal_ip"],
                 "public_ip": node["external_ip"],
                 "instance_status": node["status"]}
    node_info.update(node["tags"])

    return node_info
