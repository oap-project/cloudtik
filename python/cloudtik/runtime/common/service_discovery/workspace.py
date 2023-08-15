from typing import Dict, Any

from cloudtik.core._private.service_discovery.runtime_services import is_service_discovery_runtime, \
    get_service_discovery_runtime
from cloudtik.core._private.service_discovery.utils import ServiceRegisterException
from cloudtik.core._private.utils import RUNTIME_CONFIG_KEY
from cloudtik.core._private.workspace.workspace_operator import _get_workspace_provider
from cloudtik.runtime.common.service_discovery.utils import get_service_addresses_string


def register_service_to_workspace(
        cluster_config, runtime_type,
        service_addresses,
        service_name=None):
    workspace_name = cluster_config.get("workspace_name")
    if not workspace_name:
        raise ValueError("Workspace name is missing in the cluster configuration.")
    if not service_addresses:
        raise ValueError("Must specify service addresses when registering a service.")

    runtime_config = cluster_config.get(RUNTIME_CONFIG_KEY)
    if (not is_service_discovery_runtime(runtime_type) and
            get_service_discovery_runtime(runtime_config)):
        # We don't register service to workspace for discovery when there is discovery service to use.
        # The bootstrap service discovery service should run without other services
        return

    # Service register in workspace in the format of:
    # 1. cluster-name.runtime_name
    # 2. cluster-name.runtime_name.service_name
    cluster_name = cluster_config["cluster_name"]
    if service_name:
        registry_name = "{}.{}.{}".format(
            cluster_name, runtime_type, service_name)
    else:
        registry_name = "{}.{}".format(
            cluster_name, runtime_type)

    registry_addresses = get_service_addresses_string(
        service_addresses)
    service_registry = {
        registry_name: registry_addresses
    }

    workspace_provider = _get_workspace_provider(
        cluster_config["provider"], workspace_name)

    try:
        workspace_provider.publish_global_variables(
            cluster_config, service_registry)
    except Exception as e:
        # failed to register (may because the tag limit is reached
        # or exceeding the size of key or value)
        raise ServiceRegisterException(
            "Failed to register service: {}".format(str(e)))


def query_one_service_from_workspace(
        cluster_config: Dict[str, Any], service_selector):
    workspace_name = cluster_config.get("workspace_name")
    if workspace_name is None:
        return None

    workspace_provider = _get_workspace_provider(
        cluster_config["provider"], workspace_name)
    global_variables = workspace_provider.subscribe_global_variables(
        cluster_config)

    # TODO: select service
