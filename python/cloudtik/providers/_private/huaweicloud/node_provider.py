import logging
from typing import Any, Dict

from cloudtik.core._private.cli_logger import cli_logger
from cloudtik.core.node_provider import NodeProvider
from cloudtik.providers._private.huaweicloud.config import with_huaweicloud_environment_variables, \
    bootstrap_huaweicloud, post_prepare_huaweicloud, verify_obs_storage
from cloudtik.providers._private.huaweicloud.utils import get_default_huaweicloud_cloud_storage, \
    get_huaweicloud_obs_storage_config
from cloudtik.providers._private.utils import validate_config_dict

logger = logging.getLogger(__name__)


class HUAWEICLOUDNodeProvider(NodeProvider):
    def __init__(self, provider_config, cluster_name):
        NodeProvider.__init__(self, provider_config, cluster_name)
        # TODO: initialize code here

    # TODO: implement the methods for node operations

    def with_environment_variables(self, node_type_config: Dict[str, Any], node_id: str):
        """Export necessary environment variables for running node commands"""
        return with_huaweicloud_environment_variables(self.provider_config, node_type_config, node_id)

    def get_default_cloud_storage(self):
        """Return the managed cloud storage if configured."""
        return get_default_huaweicloud_cloud_storage(self.provider_config)

    @staticmethod
    def post_prepare(
            cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Fills out missing fields after the user config is merged with defaults
        This happens after prepare_config is done.
        """
        return post_prepare_huaweicloud(cluster_config)

    @staticmethod
    def validate_config(
            provider_config: Dict[str, Any]) -> None:
        """Check the provider configuration validation.
        This happens after post_prepare is done and before bootstrap_config
        """
        config_dict = {
            "region": provider_config.get("region")}
        validate_config_dict(provider_config["type"], config_dict)

        storage_config = get_huaweicloud_obs_storage_config(provider_config)
        if storage_config is not None:
            config_dict = {
                "obs.bucket": storage_config.get("obs.bucket"),
                # The access key is no longer a must since we have role access
                # "obs.access.key": storage_config.get("obs.access.key"),
                # "obs.secret.key": storage_config.get("obs.secret.key")
            }

            validate_config_dict(provider_config["type"], config_dict)

    @staticmethod
    def bootstrap_config(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Bootstraps the cluster config by adding env defaults if needed.
        This happens after validate_config is done.
        """
        return bootstrap_huaweicloud(cluster_config)

    @staticmethod
    def verify_config(
            provider_config: Dict[str, Any]) -> None:
        """Verify provider configuration. Verification usually means to check it is working.
        This happens after bootstrap_config is done.
        """
        verify_cloud_storage = provider_config.get("verify_cloud_storage", True)
        cloud_storage = get_huaweicloud_obs_storage_config(provider_config)
        if verify_cloud_storage and cloud_storage is not None:
            cli_logger.verbose("Verifying OBS storage configurations...")
            verify_obs_storage(provider_config)
            cli_logger.verbose("Successfully verified OBS storage configurations.")
