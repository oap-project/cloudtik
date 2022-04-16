import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def validate_config_dict(provider_type, config_dict: Dict[str, Any]) -> None:
    provider_config_failed = False
    for key, value in config_dict.items():
        if value is None:
            provider_config_failed = True
            logger.info("{} must be define in your yaml, please refer to config-schema.json.".format(key))
    if provider_config_failed:
        raise RuntimeError("{} provider must be provided the right config, "
                           "please refer to config-schema.json.".format(provider_type))
