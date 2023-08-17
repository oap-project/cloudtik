import re
from typing import Dict, Any

from cloudtik.core._private.service_discovery.runtime_services import is_service_discovery_runtime, \
    get_service_discovery_runtime
from cloudtik.core._private.service_discovery.utils import ServiceRegisterException, SERVICE_SELECTOR_CLUSTERS, \
    SERVICE_SELECTOR_RUNTIMES, SERVICE_SELECTOR_SERVICES, SERVICE_SELECTOR_TAGS, SERVICE_SELECTOR_LABELS, \
    SERVICE_SELECTOR_EXCLUDE_LABELS, SERVICE_DISCOVERY_TAG_CLUSTER_PREFIX, \
    SERVICE_DISCOVERY_LABEL_CLUSTER, SERVICE_DISCOVERY_LABEL_RUNTIME, get_service_discovery_config, \
    is_prefer_workspace_discovery
from cloudtik.core._private.utils import RUNTIME_CONFIG_KEY
from cloudtik.core._private.workspace.workspace_operator import _get_workspace_provider
from cloudtik.runtime.common.service_discovery.utils import get_service_addresses_string, \
    get_service_addresses_from_string


def get_service_registry_name(
        cluster_name,
        runtime_type,
        service_name=None):
    # Service register in workspace in the format of:
    # 1. cluster-name.runtime_name
    # 2. cluster-name.runtime_name.service_name
    if service_name:
        registry_name = "{}.{}.{}".format(
            cluster_name, runtime_type, service_name)
    else:
        registry_name = "{}.{}".format(
            cluster_name, runtime_type)
    return registry_name


def parse_service_registry_name(registry_name):
    name_parts = registry_name.split('.')
    n = len(name_parts)
    if n == 2:
        return name_parts[0], name_parts[1], name_parts[1]
    elif n == 3:
        return name_parts[0], name_parts[1], name_parts[2]
    else:
        raise RuntimeError("Invalid service registry name")


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
    runtime_type_config = runtime_config.get(runtime_type, {})
    service_discovery_config = get_service_discovery_config(runtime_type_config)
    if (not is_prefer_workspace_discovery(service_discovery_config) and
            not is_service_discovery_runtime(runtime_type) and
            get_service_discovery_runtime(runtime_config)):
        # We don't register service to workspace for discovery when there is discovery service to use.
        # The bootstrap service discovery service should run without other services
        return

    # Service register in workspace in the format of:
    # 1. cluster-name.runtime_name
    # 2. cluster-name.runtime_name.service_name
    cluster_name = cluster_config["cluster_name"]
    registry_name = get_service_registry_name(
        cluster_name, runtime_type, service_name)

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
    if not global_variables:
        return None

    # match through the clusters, runtimes, and services if they are provided
    services = _query_one_service_registry(
        global_variables, service_selector)
    if not services:
        return None
    # return one of them
    return next(iter(services.items()))


def _query_one_service_registry(
        service_registries, service_selector):
    return _query_service_registry(
        service_registries, service_selector, first_match=True)


def _query_service_registry(
        service_registries, service_selector, first_match=False):
    # select service based on cluster_name, runtime_type, service_name
    clusters = service_selector.get(SERVICE_SELECTOR_CLUSTERS)
    runtimes = service_selector.get(SERVICE_SELECTOR_RUNTIMES)
    services = service_selector.get(SERVICE_SELECTOR_SERVICES)
    tags = service_selector.get(SERVICE_SELECTOR_TAGS)
    labels = service_selector.get(SERVICE_SELECTOR_LABELS)
    exclude_labels = service_selector.get(SERVICE_SELECTOR_EXCLUDE_LABELS)
    # TODO: support exclude_joined_labels

    matched_services = {}
    for registry_name, registry_addresses in service_registries.items():
        if _match_service_registry(
                registry_name, clusters, runtimes, services,
                tags=tags, labels=labels, exclude_labels=exclude_labels):
            matched_services[registry_name] = get_service_addresses_from_string(
                registry_addresses)
            if first_match:
                return matched_services
    return matched_services


def _match_service_registry(
        registry_name, clusters=None, runtimes=None, services=None,
        tags=None, labels=None, exclude_labels=None):
    cluster_name, runtime_type, service_name = parse_service_registry_name(
        registry_name)
    if clusters and cluster_name not in clusters:
        return False
    if runtimes and runtime_type not in runtimes:
        return False
    if services and service_name not in services:
        return False
    if tags and not _match_cluster_tag(tags, cluster_name):
        return False
    if labels and not _match_labels(
            labels, cluster_name, runtime_type):
        return False
    if exclude_labels and not _match_exclude_labels(
            exclude_labels, cluster_name, runtime_type):
        return False
    return True


def _match_cluster_tag(tags, cluster_name):
    for tag in tags:
        if tag.startswith(
                SERVICE_DISCOVERY_TAG_CLUSTER_PREFIX):
            tag_cluster_name = tag[len(SERVICE_DISCOVERY_TAG_CLUSTER_PREFIX):]
            if tag_cluster_name != cluster_name:
                return False
            break
    return True


def _match_labels(labels, cluster_name, runtime_type):
    # TODO: shall we anchor to the start and end to value by default?
    tried_labels = 0
    for label_name, label_value in labels.items():
        if label_name == SERVICE_DISCOVERY_LABEL_CLUSTER:
            pattern = re.compile(label_value)
            if not pattern.match(cluster_name):
                return False
            tried_labels += 1
        elif label_name == SERVICE_DISCOVERY_LABEL_RUNTIME:
            pattern = re.compile(label_value)
            if not pattern.match(runtime_type):
                return False
            tried_labels += 1
        if tried_labels >= 2:
            break

    return True


def _match_exclude_labels(exclude_labels, cluster_name, runtime_type):
    tried_labels = 0
    for label_name, label_value in exclude_labels.items():
        if label_name == SERVICE_DISCOVERY_LABEL_CLUSTER:
            pattern = re.compile(label_value)
            if pattern.match(cluster_name):
                return False
            tried_labels += 1
        elif label_name == SERVICE_DISCOVERY_LABEL_RUNTIME:
            pattern = re.compile(label_value)
            if pattern.match(runtime_type):
                return False
            tried_labels += 1
        if tried_labels >= 2:
            break

    return True
