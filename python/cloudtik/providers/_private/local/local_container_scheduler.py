import copy
import json
import logging
import os
import subprocess
import uuid
from threading import RLock
from types import ModuleType
from typing import Dict, Optional, Any

from cloudtik.core._private.call_context import CallContext
from cloudtik.core._private.command_executor.docker_command_executor import DockerCommandExecutor
from cloudtik.core._private.utils import DOCKER_CONFIG_KEY, AUTH_CONFIG_KEY, FILE_MOUNTS_CONFIG_KEY, \
    _merge_node_type_specific_config
from cloudtik.core.tags import CLOUDTIK_TAG_NODE_KIND, NODE_KIND_HEAD, CLOUDTIK_TAG_CLUSTER_NAME, \
    CLOUDTIK_TAG_USER_NODE_TYPE
from cloudtik.providers._private.local.config import \
    _get_provider_bridge_address, _get_request_instance_type, get_docker_scheduler_lock_path, \
    get_docker_scheduler_state_path, TAG_WORKSPACE_NAME, \
    get_docker_scheduler_state_file_name
from cloudtik.providers._private.local.local_docker_command_executor import LocalDockerCommandExecutor
from cloudtik.providers._private.local.local_scheduler import LocalScheduler
from cloudtik.providers._private.local.state_store import LocalContainerStateStore, _update_node_tags
from cloudtik.providers._private.local.utils import _get_node_info, _get_tags

logger = logging.getLogger(__name__)

MAX_CONTAINER_NAME_RETRIES = 10

STATE_MOUNT_PATH = "/cloudtik/state"
DATA_MOUNT_PATH = "/cloudtik/data"
DATA_DISK_MOUNT_PATH = "/mnt/cloudtik"

INSPECT_FORMAT = (
    '{'
    '"name":{{json .Name}},'
    '"status":{{json .State.Status}},'
    '"labels":{{json .Config.Labels}},'
    '"ip":{{range .NetworkSettings.Networks}}{{json .IPAddress}}{{end}},'
    '"cpus":{{json .HostConfig.NanoCpus}},'
    '"memory":{{json .HostConfig.Memory}}'
    '}')


def _is_head_node(tags):
    if not tags or CLOUDTIK_TAG_NODE_KIND not in tags:
        return False
    return True if tags[CLOUDTIK_TAG_NODE_KIND] == NODE_KIND_HEAD else False


def _get_merged_docker_config_from_node_config(
        docker_config, node_config):
    node_type_docker_config = node_config.get(DOCKER_CONFIG_KEY)
    return _merge_node_type_specific_config(
        docker_config, node_type_docker_config)


def _is_running(container):
    if container is None:
        return False
    # all status:
    # created
    # running
    # restarting
    # exited
    # paused
    # dead
    state = container["state"]
    return state == "created" or state == "running" or state == "restarting"


def _is_terminated(container):
    return not _is_running(container)


def _get_container_resources(container_object):
    resources = {}
    nano_cpus = container_object.get("cpus", 0)
    if nano_cpus:
        resources["CPU"] = round(nano_cpus / (10 ** 9), 2)
    memory_bytes = container_object.get("memory", 0)
    if memory_bytes:
        resources["memory"] = round(memory_bytes / (1024 * 1024 * 1024), 2)
    return resources


def _map_by_clusters(containers):
    cluster_containers = {}
    for container in containers:
        tags = container.get("tags", {})
        if CLOUDTIK_TAG_CLUSTER_NAME not in tags:
            continue
        cluster_name = tags[CLOUDTIK_TAG_CLUSTER_NAME]
        if cluster_name not in cluster_containers:
            cluster_containers[cluster_name] = []
        cluster_containers[cluster_name].append(container)
    return cluster_containers


def _apply_filters_with_state(
        nodes, tag_filters,
        state):
    state_nodes = state.get_nodes()
    matching_nodes = []
    for node in nodes:
        if not _is_running(node):
            continue
        node_id = node["name"]
        state_node = state_nodes.get(node_id)
        tags = _get_tags(state_node)
        node_tags = _get_tags(node)
        node_tags.update(tags)

        ok = True
        for k, v in tag_filters.items():
            if node_tags.get(k) != v:
                ok = False
                break
        if ok:
            # update the merged tags to container
            node["tags"] = node_tags
            matching_nodes.append(node)
    return matching_nodes


class LocalContainerScheduler(LocalScheduler):
    def __init__(self, provider_config, cluster_name):
        LocalScheduler.__init__(self, provider_config, cluster_name)

        bridge_address = _get_provider_bridge_address(provider_config)
        if bridge_address:
            # address in the form of IP:port
            address_parts = bridge_address.split(':')
            self.bridge_ip = address_parts[0]
            self.bridge_port = address_parts[1]
        else:
            self.bridge_ip = None
            self.bridge_port = None

        self.call_context = CallContext()
        self.call_context.set_call_from_api(True)

        self.docker_config = self.provider_config.get(DOCKER_CONFIG_KEY, {})
        self.file_mounts = self.provider_config.get(FILE_MOUNTS_CONFIG_KEY, {})

        self.lock = RLock()
        # Cache of node objects from the last nodes() call. This avoids
        # excessive remote requests.
        self.cached_nodes: Dict[str, Any] = {}

        # shared scheduler container for common operations
        self.scheduler_executor = self._get_scheduler_executor(
            None, docker_config=self.docker_config)

        # Use state only for cluster cases because the state is per cluster state
        if self.cluster_name:
            # workspace_name must be set in the provider config
            # when cluster name exists
            workspace_name = self.provider_config["workspace_name"]
            if self._is_in_cluster():
                self.state = self._get_state_store_in_cluster(
                    workspace_name, self.cluster_name)
            else:
                self.state = self._get_state_store(
                    workspace_name, self.cluster_name)
        else:
            self.state = None

    def create_node(self, node_config, tags, count):
        with self.lock:
            launched = 0
            while launched < count:
                # create one container
                self._start_container(node_config, tags)
                launched = launched + 1
                if count == launched:
                    return
            if launched < count:
                raise RuntimeError(
                    "No enough free nodes. {} nodes requested / {} launched.".format(
                        count, launched))

    def non_terminated_nodes(self, tag_filters):
        if tag_filters is None:
            tag_filters = {}
        if self.cluster_name:
            tag_filters[CLOUDTIK_TAG_CLUSTER_NAME] = self.cluster_name
        with self.lock:
            containers = self._list_containers(tag_filters)

            # apply the filters again
            containers = self._apply_filters(containers, tag_filters)

            # Note: All the operations use "name" as the unique node id
            self.cached_nodes = {
                container["name"]: container for container in containers
            }
            return [i["name"] for i in containers]

    def is_running(self, node_id):
        with self.lock:
            node = self._get_cached_node(node_id)
            return _is_running(node)

    def is_terminated(self, node_id):
        with self.lock:
            node = self._get_cached_node(node_id)
            return _is_terminated(node)

    def node_tags(self, node_id):
        with self.lock:
            node = self._get_cached_node(node_id)
            return _get_tags(node)

    def internal_ip(self, node_id):
        with self.lock:
            node = self._get_cached_node(node_id)
            return node.get("ip")

    def set_node_tags(self, node_id, tags):
        with self.lock:
            # update the cached node tags, although it will refresh at next non_terminated_nodes
            node = self._get_cached_node(node_id)
            _update_node_tags(node, tags)
            self._set_node_tags(node_id, tags=tags)

    def terminate_node(self, node_id):
        with self.lock:
            node = self._get_cached_node(node_id)
            self._stop_container(node_id)
            # shall we remove the node from cached node
            # the cached node list will be refreshed at next non_terminated_nodes
            # usually not problem, at least we set to "terminated"
            node["state"] = "terminated"

    def get_node_info(self, node_id):
        with self.lock:
            node = self._get_cached_node(node_id)
            if node is None:
                raise RuntimeError("Node with id {} doesn't exist.".format(node_id))
            return _get_node_info(node)

    def get_command_executor(self,
                             call_context: CallContext,
                             log_prefix: str,
                             node_id: str,
                             auth_config: Dict[str, Any],
                             cluster_name: str,
                             process_runner: ModuleType,
                             use_internal_ip: bool,
                             docker_config: Optional[Dict[str, Any]] = None
                             ) -> DockerCommandExecutor:
        # for local container scheduler, the node id is container name
        # we should avoid non-provider code handling docker "container_name" in config
        # for command executor that is not specific container related, node_id is None
        # and the following method that related to specific container should not be executed.
        if docker_config is None:
            docker_config = {}
        else:
            docker_config = copy.deepcopy(docker_config)
        docker_config["container_name"] = node_id
        common_args = {
            "log_prefix": log_prefix,
            "auth_config": auth_config,
            "cluster_name": cluster_name,
            "process_runner": process_runner,
            "use_internal_ip": use_internal_ip,
            "node_id": node_id,
            "provider": self,
        }
        if self._is_in_cluster():
            if not self.bridge_ip:
                raise RuntimeError("Missing Docker bridge SSH server IP.")
            common_args["ssh_ip"] = self.bridge_ip
            common_args["ssh_port"] = self.bridge_port
            return LocalDockerCommandExecutor(
                call_context, docker_config, True, **common_args)
        else:
            return LocalDockerCommandExecutor(
                call_context, docker_config, False, **common_args)

    def _is_in_cluster(self):
        # flag set at boostrap cluster config
        return self.provider_config.get("local_in_cluster", False)

    @staticmethod
    def _get_state_store(workspace_name, cluster_name):
        docker_scheduler_lock_path = get_docker_scheduler_lock_path(
            workspace_name, cluster_name)
        return LocalContainerStateStore(
            docker_scheduler_lock_path,
            get_docker_scheduler_state_path(
                workspace_name, cluster_name))

    @staticmethod
    def _get_state_store_in_cluster(workspace_name, cluster_name):
        docker_scheduler_lock_path = get_docker_scheduler_lock_path(
            workspace_name, cluster_name)
        docker_scheduler_state_path = os.path.join(
            STATE_MOUNT_PATH, get_docker_scheduler_state_file_name())
        return LocalContainerStateStore(
            docker_scheduler_lock_path,
            docker_scheduler_state_path)

    def _get_scheduler_executor(self, container_name, docker_config):
        log_prefix = "ContainerScheduler: "
        if self.cluster_name:
            log_prefix = log_prefix + "{}: ".format(self.cluster_name)

        # bootstrapped from config
        # ssh_user and ssh_private_key need set properly at bootstrap
        # both for working node and head node
        auth_config = self.provider_config.get(AUTH_CONFIG_KEY, {})
        return self.get_command_executor(
            self.call_context,
            log_prefix=log_prefix,
            node_id=container_name,
            auth_config=auth_config,
            cluster_name=self.cluster_name,
            process_runner=subprocess,
            use_internal_ip=True,
            docker_config=docker_config,
        )

    def _get_new_container_name(self):
        retry = 0
        while retry < MAX_CONTAINER_NAME_RETRIES:
            container_name = self._get_random_container_name()
            if not self._is_container_exists(container_name):
                return container_name
            retry += 1

        raise RuntimeError("Failed to allocate a container name.")

    def _get_random_container_name(self):
        # the container name prefix with workspace and cluster name
        # with a random string of 8 digits
        workspace_name = self.provider_config["workspace_name"].replace("-", "_")
        cluster_name = self.cluster_name.replace("-", "_")
        random_id = str(uuid.uuid1())[:8]
        return f"{workspace_name}_{cluster_name}_{random_id}"

    @staticmethod
    def _set_container_resources(node_config, docker_config):
        instance_type = _get_request_instance_type(node_config)
        if instance_type:
            cpus = instance_type.get("CPU", 0)
            if cpus:
                docker_config["cpus"] = cpus

            memory_gb = instance_type.get("memory", 0)
            if memory_gb:
                docker_config["memory"] = str(memory_gb) + "g"

    def _get_provider_cluster_state_path(self):
        # the cluster data path is set at boostrap
        return self.provider_config["state_path"]

    def _setup_docker_file_mounts(
            self, node_config, scheduler_executor):
        # Handling mounts
        # We add mounts by default the following 3 paths:
        # 1. the cluster states: to STATE_MOUNT_PATH
        # 2. the user defined data disks: from data_disk_dir #/container_name  to DATA_DISK_MOUNT_PATH/data_disk_#
        # 3. the user defined shared data dir: from data_dir to DATA_MOUNT_PATH/data_dir_name

        file_mounts = {}
        container_name = scheduler_executor.container_name

        # The bootstrap process has updated the permission
        state_path = self._get_provider_cluster_state_path()
        scheduler_executor.run(
            "mkdir -p '{path}' && chmod -R 777 '{path}'".format(
                path=state_path),
            run_env="host")
        file_mounts[STATE_MOUNT_PATH] = state_path
        data_disks = node_config.get("data_disks")
        if data_disks:
            disk_index = 1
            for data_disk in data_disks:
                host_container_data_disk = os.path.join(
                    data_disk, container_name)
                scheduler_executor.run(
                    "mkdir -p '{path}' && chmod -R a+w '{path}'".format(
                        path=host_container_data_disk),
                    run_env="host")
                # create a data disk for node
                target_data_disk = os.path.join(
                    DATA_DISK_MOUNT_PATH, "data_disk_{}".format(disk_index))
                file_mounts[target_data_disk] = host_container_data_disk
                disk_index += 1

        data_dirs = node_config.get("data_dirs")
        if data_dirs:
            for data_dir in data_dirs:
                # the bootstrap process has updated the permission
                data_dir_name = os.path.basename(data_dir)
                target_data_dir = os.path.join(
                    DATA_MOUNT_PATH, data_dir_name)
                file_mounts[target_data_dir] = data_dir

        return file_mounts

    def _start_container(self, node_config, tags):
        container_name = self._get_new_container_name()

        # check CLOUDTIK_TAG_NODE_KIND: NODE_KIND_HEAD tag for head
        is_head_node = _is_head_node(tags)
        shared_memory_ratio = node_config.get("shared_memory_ratio", 0)

        # prepare docker config
        docker_config = _get_merged_docker_config_from_node_config(
            self.docker_config, node_config)
        # make a copy before change it
        docker_config = copy.deepcopy(docker_config)
        docker_config["mounts_mapping"] = False

        # set labels
        if tags:
            docker_config["labels"] = tags

        self._set_container_resources(node_config, docker_config)

        scheduler_executor = self._get_scheduler_executor(
            container_name, docker_config=docker_config)

        file_mounts = self._setup_docker_file_mounts(
            node_config, scheduler_executor)

        scheduler_executor.start_container(
            as_head=is_head_node,
            file_mounts=file_mounts,
            shared_memory_ratio=shared_memory_ratio)

    def _get_cached_node(self, node_id: str):
        if node_id in self.cached_nodes:
            return self.cached_nodes[node_id]

        return self._get_node(node_id)

    def _get_node(self, node_id: str):
        self.non_terminated_nodes({})  # Side effect: updates cache

        with self.lock:
            if node_id in self.cached_nodes:
                return self.cached_nodes[node_id]

            node = self._get_container(container_name=node_id)
            if node is None:
                raise RuntimeError("No node found with id: {}.")
            tags = self._get_node_tags(node_id, node)
            node["tags"] = tags
            return node

    def _is_container_exists(self, container_name):
        # try cache first
        if container_name in self.cached_nodes:
            return True

        # try getting the container
        container = self._get_container(container_name)
        return True if container else False

    def _get_container(self, container_name):
        output = self.scheduler_executor.run_docker_cmd(
            "inspect --format='" + INSPECT_FORMAT + "' " + container_name + " || true")
        if not output:
            return None
        if output.startswith("Error: No such object:"):
            return None
        return self._load_container(output)

    def _stop_container(self, container_name):
        scheduler_executor = self._get_scheduler_executor(
            container_name, docker_config=self.docker_config)
        scheduler_executor.stop_container()

    def _list_containers(self, tag_filters, include_stopped=False):
        # list container tag filters only handles workspace and cluster name
        # These names are set when created which was set as docker labels
        # other filters will not be handled
        effective_filters = {}
        for name in [TAG_WORKSPACE_NAME,
                     CLOUDTIK_TAG_CLUSTER_NAME,
                     CLOUDTIK_TAG_NODE_KIND,
                     CLOUDTIK_TAG_USER_NODE_TYPE]:
            if name in tag_filters:
                effective_filters[name] = tag_filters[name]

        op_str = "container list --format '{{.Names}}'"
        if include_stopped:
            op_str += " --all"

        if effective_filters:
            for label, label_value in effective_filters.items():
                op_str += ' --filter "label={}={}"'.format(label, label_value)
        output = self.scheduler_executor.run_docker_cmd(op_str)
        if not output:
            return []

        container_names = output.splitlines()
        return self._inspect_containers(container_names)

    def _inspect_containers(self, container_names):
        containers = []
        if not container_names:
            return containers

        name_option = " ".join(container_names)
        output = self.scheduler_executor.run_docker_cmd(
            "inspect --format='" + INSPECT_FORMAT + "' " + name_option + " || true")
        if not output:
            return containers

        lines = output.splitlines()
        for line in lines:
            if line.startswith("Error: No such object:"):
                continue
            container = self._load_container(line)
            if container:
                containers.append(container)

        return containers

    @staticmethod
    def _load_container(output):
        container_object = json.loads(output)
        name = container_object.get("name")
        if name and name.startswith("/"):
            name = name[1:]
        node_ip = container_object.get("ip")
        container_status = container_object.get("status")
        tags = container_object.get("labels", {})
        resources = _get_container_resources(container_object)
        container = {
            "name": name,
            "ip": node_ip,
            "state": container_status,
            "tags": tags,
            "instance_type": resources,
            "object": container_object,
        }
        return container

    def _set_node_tags(self, node_id, tags):
        # Will use state: set labels to state
        self.state.set_node_tags(node_id, tags)

    def _get_node_tags(self, node_id, node):
        # Will use state: merge state labels with node tags
        tags = self.state.get_node_tags(node_id)
        node_tags = _get_tags(node)
        node_tags.update(tags)
        return node_tags

    def _apply_filters(self, containers, tag_filters):
        # apply the tag filters based on tags in state and containers
        return _apply_filters_with_state(containers, tag_filters, self.state)

    def _apply_filters_of_cluster(
            self, containers, tag_filters,
            workspace_name, cluster_name):
        state = self._get_state_store(workspace_name, cluster_name)
        return _apply_filters_with_state(containers, tag_filters, state)

    def list_nodes(self, workspace_name, tag_filters):
        # not use the state of this scheduler
        # List nodes that are not cluster specific, ignoring the cluster name
        # each node have the node and tags filled
        tag_filters = {} if tag_filters is None else tag_filters
        tag_filters[TAG_WORKSPACE_NAME] = workspace_name

        containers = self._list_containers(tag_filters)
        # we need to know tags of these containers
        # we sort the containers by clusters
        # for each cluster, we load its state tags
        final_containers = []
        cluster_containers = _map_by_clusters(containers)
        for cluster_name, cluster_containers in cluster_containers.items():
            filtered_containers = self._apply_filters_of_cluster(
                cluster_containers, tag_filters,
                workspace_name, cluster_name)
            final_containers += filtered_containers
        return final_containers
