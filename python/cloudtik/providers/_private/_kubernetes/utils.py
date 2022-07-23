import re

# For example cloudtik-{workspace-name}-worker-kb7w7
KUBERNETES_NAME_FIXED_MAX = 22
KUBERNETES_NAME_MAX = 256

KUBERNETES_WORKSPACE_NAME_MAX = KUBERNETES_NAME_MAX - KUBERNETES_NAME_FIXED_MAX

KUBERNETES_HEAD_SERVICE_ACCOUNT_NAME = "cloudtik-head-service-account"
KUBERNETES_WORKER_SERVICE_ACCOUNT_NAME = "cloudtik-worker-service-account"

KUBERNETES_HEAD_SERVICE_ACCOUNT_CONFIG_KEY = "head_service_account"
KUBERNETES_WORKER_SERVICE_ACCOUNT_CONFIG_KEY = "worker_service_account"


def check_kubernetes_name_format(workspace_name):
    # TODO: Improve with the correct format
    # Most resource types require a name that can be used as a DNS subdomain name as defined in RFC 1123.
    # This means the name must:
    # - contain no more than 253 characters
    # - contain only lowercase alphanumeric characters, '-' or '.'
    # - start with an alphanumeric character
    # - end with an alphanumeric character
    # '(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?'
    return bool(re.match("^(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?$", workspace_name))


def to_label_selector(tags):
    label_selector = ""
    for k, v in tags.items():
        if label_selector != "":
            label_selector += ","
        label_selector += "{}={}".format(k, v)
    return label_selector


def _get_node_info(pod):
    node_info = {"node_id": pod.metadata.name,
                 "instance_type": "Unknown",
                 "private_ip": pod.status.pod_ip,
                 "public_ip": None,
                 "instance_status": pod.status.phase}
    node_info.update(pod.metadata.labels)

    return node_info


def _get_head_service_account_name(provider_config):
    account_field = KUBERNETES_HEAD_SERVICE_ACCOUNT_CONFIG_KEY
    name = provider_config.get(account_field, {}).get("metadata", {}).get("name")
    if name is None or name == "":
        return KUBERNETES_HEAD_SERVICE_ACCOUNT_NAME
    return name


def _get_worker_service_account_name(provider_config):
    account_field = KUBERNETES_WORKER_SERVICE_ACCOUNT_CONFIG_KEY
    name = provider_config.get(account_field, {}).get("metadata", {}).get("name")
    if name is None or name == "":
        return KUBERNETES_WORKER_SERVICE_ACCOUNT_NAME
    return name
