import logging
from functools import lru_cache
from typing import Any, Dict, List

from huaweicloudsdkcore.auth.credentials import BasicCredentials, \
    GlobalCredentials
from huaweicloudsdkcore.http.http_config import HttpConfig
from huaweicloudsdkecs.v2 import EcsClient
from huaweicloudsdkecs.v2.region.ecs_region import EcsRegion
from huaweicloudsdkeip.v2 import EipClient
from huaweicloudsdkeip.v2.region.eip_region import EipRegion
from huaweicloudsdkiam.v3 import IamClient
from huaweicloudsdkiam.v3.region.iam_region import IamRegion
from huaweicloudsdknat.v2 import NatClient
from huaweicloudsdknat.v2.region.nat_region import NatRegion
from huaweicloudsdkvpc.v2 import VpcClient
from huaweicloudsdkvpc.v2.region.vpc_region import VpcRegion
from obs import ObsClient

from cloudtik.core._private.constants import \
    CLOUDTIK_DEFAULT_CLOUD_STORAGE_URI, env_bool
from cloudtik.core._private.utils import get_storage_config_for_update

OBS_SERVICES_URL = 'https://obs.myhuaweicloud.com'
HWC_OBS_BUCKET = "obs.bucket"
HWC_SERVER_TAG_STR_FORMAT = '{}={}'
HWC_SERVER_STATUS_ACTIVE = 'ACTIVE'
HWC_SERVER_STATUS_BUILD = 'BUILD'
HWC_SERVER_STATUS_NON_TERMINATED = {HWC_SERVER_STATUS_BUILD,
                                    HWC_SERVER_STATUS_ACTIVE}

logger = logging.getLogger(__name__)


@lru_cache()
def _client_cache(region: str = None, ak: str = None, sk: str = None) -> Dict[
    str, Any]:
    client_map = {}
    credentials = BasicCredentials(ak, sk) if (ak and sk) else None
    iam_credentials = GlobalCredentials(ak, sk) if (ak and sk) else None

    # Get proxy setting, if $HWC_IGNORE_SSL_VERIFICATION is true explicitly,
    # ignore checking, in other case enable SSL verifying.
    http_config = HttpConfig.get_default_config()
    http_config.ignore_ssl_verification = env_bool(
        'HWC_IGNORE_SSL_VERIFICATION', False)

    ecs_client = EcsClient.new_builder() \
        .with_http_config(http_config) \
        .with_credentials(credentials) \
        .with_region(EcsRegion.value_of(region) if region else None) \
        .build()
    client_map['ecs'] = ecs_client

    vpc_client = VpcClient.new_builder() \
        .with_http_config(http_config) \
        .with_credentials(credentials) \
        .with_region(VpcRegion.value_of(region) if region else None) \
        .build()
    client_map['vpc'] = vpc_client

    nat_client = NatClient.new_builder() \
        .with_http_config(http_config) \
        .with_credentials(credentials) \
        .with_region(NatRegion.value_of(region) if region else None) \
        .build()
    client_map['nat'] = nat_client

    eip_client = EipClient.new_builder() \
        .with_http_config(http_config) \
        .with_credentials(credentials) \
        .with_region(EipRegion.value_of(region) if region else None) \
        .build()
    client_map['eip'] = eip_client

    iam_client = IamClient.new_builder() \
        .with_http_config(http_config) \
        .with_credentials(iam_credentials) \
        .with_region(IamRegion.value_of(region) if region else None) \
        .build()
    client_map['iam'] = iam_client

    _ssl_verify = not env_bool('HWC_IGNORE_SSL_VERIFICATION', False)
    if ak and sk:
        obs_client = ObsClient(access_key_id=ak, secret_access_key=sk,
                               server=OBS_SERVICES_URL, ssl_verify=_ssl_verify,
                               region=region)
    else:
        obs_client = ObsClient(server=OBS_SERVICES_URL, ssl_verify=_ssl_verify,
                               region=region)
    client_map['obs'] = obs_client

    return client_map


def _make_client(
        config_provider: Dict[str, Any], client_type: str, region=None):
    region = config_provider.get('region') if region is None else region
    credentials = config_provider.get('huaweicloud_credentials')
    if credentials:
        ak = credentials.get('huaweicloud_access_key')
        sk = credentials.get('huaweicloud_secret_key')
        _client_cache_map = _client_cache(region, ak, sk)
    else:
        _client_cache_map = _client_cache(region)
    return _client_cache_map[client_type]


def make_ecs_client(config: Dict[str, Any], region=None) -> Any:
    return _make_ecs_client(config['provider'], region)


def _make_ecs_client(config_provider: Dict[str, Any], region=None) -> Any:
    return _make_client(config_provider, 'ecs', region)


def make_vpc_client(config: Dict[str, Any], region=None) -> Any:
    return _make_vpc_client(config['provider'], region)


def _make_vpc_client(config_provider: Dict[str, Any], region=None) -> Any:
    return _make_client(config_provider, 'vpc', region)


def make_nat_client(config: Dict[str, Any], region=None) -> Any:
    return _make_nat_client(config['provider'], region)


def _make_nat_client(config_provider: Dict[str, Any], region=None) -> Any:
    return _make_client(config_provider, 'nat', region)


def make_eip_client(config: Dict[str, Any], region=None) -> Any:
    return _make_eip_client(config['provider'], region)


def _make_eip_client(config_provider: Dict[str, Any], region=None) -> Any:
    return _make_client(config_provider, 'eip', region)


def make_iam_client(config: Dict[str, Any], region=None) -> Any:
    return _make_iam_client(config['provider'], region)


def _make_iam_client(config_provider: Dict[str, Any], region=None) -> Any:
    return _make_client(config_provider, 'iam', region)


def make_obs_client_aksk(ak: str, sk: str, region: str = None) -> Any:
    config_provider = {}
    if ak and sk:
        aksk = {"huaweicloud_access_key": ak, "huaweicloud_secret_key": sk}
        config_provider["huaweicloud_credentials"] = aksk
    return _make_obs_client(config_provider, region=region)


def make_obs_client(config: Dict[str, Any], region=None) -> Any:
    return _make_obs_client(config["provider"], region)


def _make_obs_client(config_provider: Dict[str, Any], region=None) -> Any:
    return _make_client(config_provider, 'obs', region)


def get_huaweicloud_obs_storage_config(provider_config: Dict[str, Any]):
    if "storage" in provider_config and "huaweicloud_obs_storage" in \
            provider_config["storage"]:
        return provider_config["storage"]["huaweicloud_obs_storage"]

    return None


def get_huaweicloud_obs_storage_config_for_update(
        provider_config: Dict[str, Any]):
    storage_config = get_storage_config_for_update(provider_config)
    if "huaweicloud_obs_storage" not in storage_config:
        storage_config["huaweicloud_obs_storage"] = {}
    return storage_config["huaweicloud_obs_storage"]


def export_huaweicloud_obs_storage_config(provider_config,
                                          config_dict: Dict[str, Any]):
    cloud_storage = get_huaweicloud_obs_storage_config(provider_config)
    if cloud_storage is None:
        return
    config_dict["HUAWEICLOUD_CLOUD_STORAGE"] = True

    obs_bucket = cloud_storage.get(HWC_OBS_BUCKET)
    if obs_bucket:
        config_dict["HUAWEICLOUD_OBS_BUCKET"] = obs_bucket

    obs_access_key = cloud_storage.get("obs.access.key")
    if obs_access_key:
        config_dict["HUAWEICLOUD_OBS_ACCESS_KEY"] = obs_access_key

    obs_secret_key = cloud_storage.get("obs.secret.key")
    if obs_secret_key:
        config_dict["HUAWEICLOUD_OBS_SECRET_KEY"] = obs_secret_key


def get_huaweicloud_cloud_storage_uri(huaweicloud_cloud_storage):
    obs_bucket = huaweicloud_cloud_storage.get(HWC_OBS_BUCKET)
    if obs_bucket is None:
        return None

    return "obs://{}".format(obs_bucket)


def get_default_huaweicloud_cloud_storage(provider_config):
    cloud_storage = get_huaweicloud_obs_storage_config(provider_config)
    if cloud_storage is None:
        return None

    cloud_storage_info = {}
    cloud_storage_info.update(cloud_storage)

    cloud_storage_uri = get_huaweicloud_cloud_storage_uri(cloud_storage)
    if cloud_storage_uri:
        cloud_storage_info[
            CLOUDTIK_DEFAULT_CLOUD_STORAGE_URI] = cloud_storage_uri

    return cloud_storage_info


def flat_tags_map(tags: Dict[str, str]) -> str:
    return ','.join(
        HWC_SERVER_TAG_STR_FORMAT.format(k, v) for k, v in tags.items())


def tags_list_to_dict(tags: list) -> Dict[str, str]:
    tags_dict = {}
    for item in tags:
        k, v = item.split('=', 1)
        tags_dict[k] = v
    return tags_dict


def _get_node_private_and_public_ip(node) -> List[str]:
    private_ip = public_ip = ''
    for _, addrs in node.addresses.items():
        for addr in addrs:
            if addr.os_ext_ip_stype == 'fixed' and not private_ip:
                private_ip = addr.addr
            if addr.os_ext_ip_stype == 'floating' and not public_ip:
                public_ip = addr.addr
        if private_ip and public_ip:
            break
    return [private_ip, public_ip]


def _get_node_info(node):
    private_ip, public_ip = _get_node_private_and_public_ip(node)
    node_info = {"node_id": node.id,
                 "instance_type": node.flavor.id,
                 "private_ip": private_ip,
                 "public_ip": public_ip,
                 "instance_status": node.status}
    node_info.update(tags_list_to_dict(node.tags))
    return node_info
