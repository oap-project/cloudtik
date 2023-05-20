import copy
import logging
import json
import os
from typing import Any, Dict

import yaml

from cloudtik.core._private.concurrent_cache import ConcurrentObjectCache
from cloudtik.core._private.core_utils import _load_class

logger = logging.getLogger(__name__)

# For caching provider instantiations across API calls of one python session
_node_provider_instances = ConcurrentObjectCache()

# Minimal config for compatibility with legacy-style external configs.
MINIMAL_EXTERNAL_CONFIG = {
    "available_node_types": {
        "head.default": {},
        "worker.default": {},
    },
    "head_node_type": "head.default",
}


def _import_aws(provider_config):
    from cloudtik.providers._private.aws.node_provider import AWSNodeProvider
    return AWSNodeProvider


def _import_gcp(provider_config):
    from cloudtik.providers._private.gcp.node_provider import GCPNodeProvider
    return GCPNodeProvider


def _import_azure(provider_config):
    from cloudtik.providers._private._azure.node_provider import AzureNodeProvider
    return AzureNodeProvider


def _import_aliyun(provider_config):
    from cloudtik.providers._private.aliyun.node_provider import AliyunNodeProvider
    return AliyunNodeProvider


def _import_local(provider_config):
    from cloudtik.providers._private.local.node_provider import (
        LocalNodeProvider)
    return LocalNodeProvider


def _import_onprem(provider_config):
    from cloudtik.providers._private.onprem.node_provider import (
        OnPremNodeProvider)
    return OnPremNodeProvider


def _import_kubernetes(provider_config):
    from cloudtik.providers._private._kubernetes.node_provider import \
        KubernetesNodeProvider
    return KubernetesNodeProvider


def _import_huaweicloud(provider_config):
    from cloudtik.providers._private.huaweicloud.node_provider import \
        HUAWEICLOUDNodeProvider
    return HUAWEICLOUDNodeProvider


def _load_onprem_provider_home():
    import cloudtik.providers.onprem as onprem_provider
    return os.path.dirname(onprem_provider.__file__)


def _load_local_provider_home():
    import cloudtik.providers.local as local_provider
    return os.path.dirname(local_provider.__file__)


def _load_kubernetes_provider_home():
    import cloudtik.providers.kubernetes as kubernetes_provider
    return os.path.dirname(kubernetes_provider.__file__)


def _load_aws_provider_home():
    import cloudtik.providers.aws as aws_provider
    return os.path.dirname(aws_provider.__file__)


def _load_gcp_provider_home():
    import cloudtik.providers.gcp as gcp_provider
    return os.path.dirname(gcp_provider.__file__)


def _load_azure_provider_home():
    import cloudtik.providers.azure as azure_provider
    return os.path.dirname(azure_provider.__file__)


def _load_aliyun_provider_home():
    import cloudtik.providers.aliyun as aliyun_provider
    return os.path.dirname(aliyun_provider.__file__)


def _load_huaweicloud_provider_home():
    import cloudtik.providers.huaweicloud as huaweicloud_provider
    return os.path.dirname(huaweicloud_provider.__file__)


def _load_onprem_defaults_config():
    return os.path.join(_load_onprem_provider_home(), "defaults.yaml")


def _load_local_defaults_config():
    return os.path.join(_load_local_provider_home(), "defaults.yaml")


def _load_kubernetes_defaults_config():
    return os.path.join(_load_kubernetes_provider_home(), "defaults.yaml")


def _load_aws_defaults_config():
    return os.path.join(_load_aws_provider_home(), "defaults.yaml")


def _load_gcp_defaults_config():
    return os.path.join(_load_gcp_provider_home(), "defaults.yaml")


def _load_azure_defaults_config():
    return os.path.join(_load_azure_provider_home(), "defaults.yaml")


def _load_aliyun_defaults_config():
    return os.path.join(_load_aliyun_provider_home(), "defaults.yaml")


def _load_huaweicloud_defaults_config():
    return os.path.join(_load_huaweicloud_provider_home(), "defaults.yaml")


def _import_external(provider_config):
    provider_cls = _load_class(path=provider_config["provider_class"])
    return provider_cls


_NODE_PROVIDERS = {
    "local": _import_local,  # Run clusters on single local node
    "onprem": _import_onprem, # Run clusters with on-premise nodes
    "aws": _import_aws,
    "gcp": _import_gcp,
    "azure": _import_azure,
    "aliyun": _import_aliyun,
    "kubernetes": _import_kubernetes,
    "huaweicloud": _import_huaweicloud,
    "external": _import_external  # Import an external module
}

_PROVIDER_PRETTY_NAMES = {
    "local": "Local",
    "onprem": "On-Premise",
    "aws": "AWS",
    "gcp": "GCP",
    "azure": "Azure",
    "aliyun": "Aliyun",
    "kubernetes": "Kubernetes",
    "huaweicloud": "HUAWEI CLOUD",
    "external": "External"
}

_PROVIDER_HOMES = {
    "local": _load_local_provider_home,
    "onprem": _load_onprem_provider_home,
    "aws": _load_aws_provider_home,
    "gcp": _load_gcp_provider_home,
    "azure": _load_azure_provider_home,
    "aliyun": _load_aliyun_provider_home,
    "kubernetes": _load_kubernetes_provider_home,
    "huaweicloud": _load_huaweicloud_provider_home,
}

_DEFAULT_CONFIGS = {
    "local": _load_local_defaults_config,
    "onprem": _load_onprem_defaults_config,
    "aws": _load_aws_defaults_config,
    "gcp": _load_gcp_defaults_config,
    "azure": _load_azure_defaults_config,
    "aliyun": _load_aliyun_defaults_config,
    "kubernetes": _load_kubernetes_defaults_config,
    "huaweicloud": _load_huaweicloud_defaults_config,
}

# For caching workspace provider instantiations across API calls of one python session
_workspace_provider_instances = ConcurrentObjectCache()


def _import_aws_workspace(provider_config):
    from cloudtik.providers._private.aws.workspace_provider import AWSWorkspaceProvider
    return AWSWorkspaceProvider


def _import_gcp_workspace(provider_config):
    from cloudtik.providers._private.gcp.workspace_provider import GCPWorkspaceProvider
    return GCPWorkspaceProvider


def _import_azure_workspace(provider_config):
    from cloudtik.providers._private._azure.workspace_provider import AzureWorkspaceProvider
    return AzureWorkspaceProvider


def _import_aliyun_workspace(provider_config):
    from cloudtik.providers._private.aliyun.workspace_provider import AliyunWorkspaceProvider
    return AliyunWorkspaceProvider

def _import_local_workspace(provider_config):
    from cloudtik.providers._private.local.workspace_provider import \
        LocalWorkspaceProvider
    return LocalWorkspaceProvider


def _import_onprem_workspace(provider_config):
    from cloudtik.providers._private.onprem.workspace_provider import \
        OnPremWorkspaceProvider
    return OnPremWorkspaceProvider


def _import_kubernetes_workspace(provider_config):
    from cloudtik.providers._private._kubernetes.workspace_provider import \
        KubernetesWorkspaceProvider
    return KubernetesWorkspaceProvider


def _import_huaweicloud_workspace(provider_config):
    from cloudtik.providers._private.huaweicloud.workspace_provider import \
        HUAWEICLOUDWorkspaceProvider
    return HUAWEICLOUDWorkspaceProvider


_WORKSPACE_PROVIDERS = {
    "local": _import_local_workspace,
    "onprem": _import_onprem_workspace,
    "aws": _import_aws_workspace,
    "gcp": _import_gcp_workspace,
    "azure": _import_azure_workspace,
    "aliyun": _import_aliyun_workspace,
    "kubernetes": _import_kubernetes_workspace,
    "huaweicloud": _import_huaweicloud_workspace,
    "external": _import_external  # Import an external module
}


def _get_node_provider_cls(provider_config: Dict[str, Any]):
    """Get the node provider class for a given provider config.

    Note that this may be used by private node providers that proxy methods to
    built-in node providers, so we should maintain backwards compatibility.

    Args:
        provider_config: provider section of the cluster config.

    Returns:
        NodeProvider class
    """
    importer = _NODE_PROVIDERS.get(provider_config["type"])
    if importer is None:
        raise NotImplementedError("Unsupported node provider: {}".format(
            provider_config["type"]))
    return importer(provider_config)


def _get_node_provider(provider_config: Dict[str, Any],
                       cluster_name: str,
                       use_cache: bool = True) -> Any:
    """Get the instantiated node provider for a given provider config.

    Note that this may be used by private node providers that proxy methods to
    built-in node providers, so we should maintain backwards compatibility.

    Args:
        provider_config: provider section of the cluster config.
        cluster_name: cluster name from the cluster config.
        use_cache: whether or not to use a cached definition if available. If
            False, the returned object will also not be stored in the cache.

    Returns:
        NodeProvider
    """
    def load_node_provider(provider_config: Dict[str, Any], cluster_name: str):
        provider_cls = _get_node_provider_cls(provider_config)
        return provider_cls(provider_config, cluster_name)

    if not use_cache:
        return load_node_provider(provider_config, cluster_name)

    provider_key = (json.dumps(provider_config, sort_keys=True), cluster_name)
    return _node_provider_instances.get(
        provider_key, load_node_provider,
        provider_config=provider_config, cluster_name=cluster_name)


def _clear_provider_cache():
    _node_provider_instances.clear()


def _get_default_config(provider_config):
    """Retrieve a node provider.

    This is an INTERNAL API. It is not allowed to call this from outside.
    """
    if provider_config["type"] == "external":
        return copy.deepcopy(MINIMAL_EXTERNAL_CONFIG)
    load_config = _DEFAULT_CONFIGS.get(provider_config["type"])
    if load_config is None:
        raise NotImplementedError("Unsupported node provider: {}".format(
            provider_config["type"]))
    path_to_default = load_config()
    with open(path_to_default) as f:
        defaults = yaml.safe_load(f) or {}

    return defaults


def _get_workspace_provider_cls(provider_config: Dict[str, Any]):
    """Get the workspace provider class for a given provider config.

    Note that this may be used by private workspace providers that proxy methods to
    built-in workspace providers, so we should maintain backwards compatibility.

    Args:
        provider_config: provider section of the workspace config.

    Returns:
        WorkspaceProvider class
    """
    importer = _WORKSPACE_PROVIDERS.get(provider_config["type"])
    if importer is None:
        raise NotImplementedError("Unsupported workspace provider: {}".format(
            provider_config["type"]))
    return importer(provider_config)


def _get_workspace_provider(provider_config: Dict[str, Any],
                       workspace_name: str,
                       use_cache: bool = True) -> Any:
    """Get the instantiated workspace provider for a given provider config.

    Note that this may be used by private workspace providers that proxy methods to
    built-in workspace providers, so we should maintain backwards compatibility.

    Args:
        provider_config: provider section of the cluster config.
        workspace_name: workspace name from the cluster config.
        use_cache: whether or not to use a cached definition if available. If
            False, the returned object will also not be stored in the cache.

    Returns:
        WorkspaceProvider
    """
    def load_workspace_provider(provider_config: Dict[str, Any], workspace_name: str):
        provider_cls = _get_workspace_provider_cls(provider_config)
        return provider_cls(provider_config, workspace_name)

    if not use_cache:
        return load_workspace_provider(provider_config, workspace_name)

    provider_key = (json.dumps(provider_config, sort_keys=True), workspace_name)
    return _workspace_provider_instances.get(
        provider_key, load_workspace_provider,
        provider_config=provider_config, workspace_name=workspace_name)


def _clear_workspace_provider_cache():
    _workspace_provider_instances.clear()


def _get_default_workspace_config(provider_config):
    return _get_provider_config_object(provider_config, "workspace-defaults")


def _get_provider_config_object(provider_config, object_name: str):
    # For external provider, from the shared config object it there is one
    if provider_config["type"] == "external":
        return {"from": object_name}

    if not object_name.endswith(".yaml"):
        object_name += ".yaml"

    load_config_home = _PROVIDER_HOMES.get(provider_config["type"])
    if load_config_home is None:
        raise NotImplementedError("Unsupported provider: {}".format(
            provider_config["type"]))
    path_to_home = load_config_home()
    path_to_config_file = os.path.join(path_to_home, object_name)
    with open(path_to_config_file) as f:
        config_object = yaml.safe_load(f) or {}

    return config_object
