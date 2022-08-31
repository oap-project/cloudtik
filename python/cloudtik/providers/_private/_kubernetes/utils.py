import re
import logging

from kubernetes.client.rest import ApiException

from cloudtik.core._private.cli_logger import cli_logger
from cloudtik.core.tags import CLOUDTIK_TAG_CLUSTER_NAME
from cloudtik.providers._private._kubernetes import core_api

# For example cloudtik-{workspace-name}-worker-kb7w7
KUBERNETES_NAME_FIXED_MAX = 22
KUBERNETES_NAME_MAX = 253

KUBERNETES_WORKSPACE_NAME_MAX = KUBERNETES_NAME_MAX - KUBERNETES_NAME_FIXED_MAX

KUBERNETES_HEAD_SERVICE_ACCOUNT_NAME = "cloudtik-head-service-account"
KUBERNETES_WORKER_SERVICE_ACCOUNT_NAME = "cloudtik-worker-service-account"

KUBERNETES_HEAD_SERVICE_ACCOUNT_CONFIG_KEY = "head_service_account"
KUBERNETES_WORKER_SERVICE_ACCOUNT_CONFIG_KEY = "worker_service_account"


logger = logging.getLogger(__name__)


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


def get_instance_type_for_pod(pod):
    instance_type = "Unknown"

    resources = pod.spec.containers[0].resources
    if resources is not None and resources.requests is not None:
        requests = resources.requests
        cpu = requests.get("cpu")
        memory = requests.get("memory")
        if cpu is not None and memory is not None:
            instance_type = "{} cores/{} memory".format(cpu, memory)
        elif cpu is not None:
            instance_type = "{} cores".format(cpu)
        elif memory is not None:
            instance_type = "{} memory".format(cpu)

    return instance_type


def _get_node_info(pod):
    instance_type = get_instance_type_for_pod(pod)
    node_info = {"node_id": pod.metadata.name,
                 "instance_type": instance_type,
                 "private_ip": pod.status.pod_ip,
                 "public_ip": None,
                 "instance_status": pod.status.phase}
    node_info.update(pod.metadata.labels)

    return node_info


def _get_head_service_account_name(provider_config):
    return KUBERNETES_HEAD_SERVICE_ACCOUNT_NAME


def _get_worker_service_account_name(provider_config):
    return KUBERNETES_WORKER_SERVICE_ACCOUNT_NAME


def _get_service_account(namespace, name):
    field_selector = "metadata.name={}".format(name)

    cli_logger.verbose("Getting the service account: {} {}.", namespace, name)
    accounts = core_api().list_namespaced_service_account(
        namespace, field_selector=field_selector).items
    if len(accounts) > 0:
        assert len(accounts) == 1
        cli_logger.verbose("Successfully get the service account: {} {}.", namespace, name)
        return accounts[0]
    cli_logger.verbose("Failed to get the service account: {} {}.", namespace, name)
    return None


def get_service_external_address(provider_config):
    head_service_config = provider_config["head_service"]
    service_name = head_service_config["metadata"]["name"]
    namespace = head_service_config["metadata"]["namespace"]
    service = core_api().read_namespaced_service(namespace=namespace, name=service_name)
    ingress = service.status.load_balancer.ingress[0]
    if ingress.hostname:
        return ingress.hostname
    else:
        return ingress.ip


def delete_persistent_volume_claims(pvcs, namespace):
    for pvc in pvcs:
        delete_persistent_volume_claim(pvc.metadata.name, namespace)


def delete_persistent_volume_claims_by_name(names, namespace):
    for name in names:
        delete_persistent_volume_claim(name, namespace)


def delete_persistent_volume_claim(name, namespace):
    try:
        core_api().delete_namespaced_persistent_volume_claim(
            name, namespace)
    except ApiException as e:
        if e.status == 404:
            return
        else:
            raise


def create_and_configure_pvc_for_pod(_pod_spec, data_disks, cluster_name, namespace):
    if data_disks is None or len(data_disks) == 0:
        return None

    created_pvcs = []
    for data_disk in data_disks:
        pvc_spec = {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "generateName": "cloudtik-{}-{}-".format(cluster_name, data_disk["name"])
            },
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "resources": {
                    "requests": {
                        "storage": data_disk["diskSize"]
                    }
                }
            },
        }

        if "storageClass" in data_disk:
            pvc_spec["spec"]["storageClassName"] = data_disk["storageClass"]

        tags = {CLOUDTIK_TAG_CLUSTER_NAME: cluster_name}
        pvc_spec["metadata"]["namespace"] = namespace
        if "labels" in pvc_spec["metadata"]:
            pvc_spec["metadata"]["labels"].update(tags)
        else:
            pvc_spec["metadata"]["labels"] = tags
        try:
            pvc = core_api().create_namespaced_persistent_volume_claim(namespace, pvc_spec)
            created_pvcs.append(pvc)
        except ApiException:
            logger.error("Error happened creating persistent volume claims for pod. Try clean up...")
            delete_persistent_volume_claims(created_pvcs, namespace)
            raise

    new_volumes = []
    new_mounts = []
    index = 1
    for created_pvc in created_pvcs:
        volume_name = "data-disk-{}".format(index)
        volume = {
            "name": volume_name,
            "persistentVolumeClaim": {
                "claimName": created_pvc.metadata.name,
            }
        }
        mount = {
            "mountPath": "/mnt/cloudtik/data_disk_{}".format(index),
            "name": volume_name,
        }
        new_volumes.append(volume)
        new_mounts.append(mount)
        index += 1

    # Update pod spec for volumes and mounts
    volumes = _pod_spec["spec"].get("volumes", [])
    volumes += new_volumes
    _pod_spec["spec"]["volumes"] = volumes

    for container in _pod_spec["spec"]["containers"]:
        mounts = container.get("volumeMounts", [])
        mounts += new_mounts
        container["volumeMounts"] = mounts

    return created_pvcs


def _get_data_disk_pvc(volume, cluster_name, namespace):
    if volume.persistent_volume_claim is None:
        return None

    claim_name = volume.persistent_volume_claim.claim_name
    if not claim_name.startswith(f"cloudtik-{cluster_name}-"):
        return None

    return claim_name


def get_pod_persistent_volume_claims(pod_name, cluster_name, namespace):
    pod_pvcs = []
    try:
        pod = core_api().read_namespaced_pod(pod_name, namespace)
    except ApiException as e:
        if e.status == 404:
            return pod_pvcs
        else:
            raise

    volumes = pod.spec.volumes
    if volumes is not None:
        for volume in volumes:
            pvc = _get_data_disk_pvc(volume, cluster_name, namespace)
            if pvc is not None:
                pod_pvcs.append(pvc)

    return pod_pvcs


def cleanup_orphan_pvcs(cluster_name, namespace):
    tag_filters = {CLOUDTIK_TAG_CLUSTER_NAME: cluster_name}
    label_selector = to_label_selector(tag_filters)
    pvc_list = core_api().list_namespaced_persistent_volume_claim(
        namespace,
        label_selector=label_selector)
    delete_persistent_volume_claims(pvc_list.items, namespace)


def get_pem_path_for_kubernetes(config):
    pem_file_path = "~/.ssh/cloudtik_kubernetes_{}_{}.pem".format(config["provider"]["cloud_provider"]["type"],
                                                                  config["cluster_name"])
    return pem_file_path
