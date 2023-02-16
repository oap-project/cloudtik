import logging
from functools import lru_cache
from typing import Any, Dict

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

from cloudtik.core._private.constants import env_bool

OBS_SERVICES_URL = 'https://obs.myhuaweicloud.com'

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


def make_ecs_client(config: Dict[str, Any]) -> Any:
    config_provider = config['provider']
    region = config_provider.get('region')
    credentials = config_provider.get('huaweicloud_credentials')
    if credentials:
        ak = credentials.get('huaweicloud_access_key')
        sk = credentials.get('huaweicloud_secret_key')
        _client_cache_map = _client_cache(region, ak, sk)
    else:
        _client_cache_map = _client_cache(region)
    return _client_cache_map['ecs']


def make_vpc_client(config: Dict[str, Any]) -> Any:
    config_provider = config['provider']
    region = config_provider.get('region')
    credentials = config_provider.get('huaweicloud_credentials')
    if credentials:
        ak = credentials.get('huaweicloud_access_key')
        sk = credentials.get('huaweicloud_secret_key')
        _client_cache_map = _client_cache(region, ak, sk)
    else:
        _client_cache_map = _client_cache(region)
    return _client_cache_map['vpc']


def make_nat_client(config: Dict[str, Any]) -> Any:
    config_provider = config['provider']
    region = config_provider.get('region')
    credentials = config_provider.get('huaweicloud_credentials')
    if credentials:
        ak = credentials.get('huaweicloud_access_key')
        sk = credentials.get('huaweicloud_secret_key')
        _client_cache_map = _client_cache(region, ak, sk)
    else:
        _client_cache_map = _client_cache(region)
    return _client_cache_map['nat']


def make_eip_client(config: Dict[str, Any]) -> Any:
    config_provider = config['provider']
    region = config_provider.get('region')
    credentials = config_provider.get('huaweicloud_credentials')
    if credentials:
        ak = credentials.get('huaweicloud_access_key')
        sk = credentials.get('huaweicloud_secret_key')
        _client_cache_map = _client_cache(region, ak, sk)
    else:
        _client_cache_map = _client_cache(region)
    return _client_cache_map['eip']


def make_iam_client(config: Dict[str, Any]) -> Any:
    config_provider = config['provider']
    region = config_provider.get('region')
    credentials = config_provider.get('huaweicloud_credentials')
    if credentials:
        ak = credentials.get('huaweicloud_access_key')
        sk = credentials.get('huaweicloud_secret_key')
        _client_cache_map = _client_cache(region, ak, sk)
    else:
        _client_cache_map = _client_cache(region)
    return _client_cache_map['iam']


def make_obs_client(config: Dict[str, Any]) -> Any:
    config_provider = config['provider']
    region = config_provider.get('region')
    credentials = config_provider.get('huaweicloud_credentials')
    if credentials:
        ak = credentials.get('huaweicloud_access_key')
        sk = credentials.get('huaweicloud_secret_key')
        _client_cache_map = _client_cache(region, ak, sk)
    else:
        _client_cache_map = _client_cache(region)
    return _client_cache_map['obs']
