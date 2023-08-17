import concurrent.futures
import copy
import json
import logging
import os
import subprocess
import uuid
from threading import RLock
from types import ModuleType
from typing import Dict, Optional, Any, List

from cloudtik.core._private.call_context import CallContext
from cloudtik.core._private.cli_logger import cli_logger
from cloudtik.core._private.command_executor.docker_command_executor import DockerCommandExecutor
from cloudtik.core._private.core_utils import get_memory_in_bytes
from cloudtik.core._private.state.file_state_store import FileStateStore
from cloudtik.core._private.utils import DOCKER_CONFIG_KEY, AUTH_CONFIG_KEY, FILE_MOUNTS_CONFIG_KEY, \
    _merge_node_type_specific_config, is_head_node_by_tags, _is_use_internal_ip
from cloudtik.core.tags import CLOUDTIK_TAG_NODE_KIND, CLOUDTIK_TAG_CLUSTER_NAME, \
    CLOUDTIK_TAG_USER_NODE_TYPE, CLOUDTIK_TAG_WORKSPACE_NAME
from cloudtik.providers._private.virtual.config import \
    _get_provider_bridge_address, _get_request_instance_type, get_virtual_scheduler_lock_path, \
    get_virtual_scheduler_state_path, \
    get_virtual_scheduler_state_file_name
from cloudtik.providers._private.virtual.virtual_docker_command_executor import VirtualDockerCommandExecutor
from cloudtik.providers._private.virtual.utils import _get_node_info, _get_tags

logger = logging.getLogger(__name__)

MAX_CONTAINER_NAME_RETRIES = 10

VIRTUAL_TAG_EXTERNAL_IP = "virtual-external-ip"

STATE_MOUNT_PATH = "/cloudtik/state"
DATA_MOUNT_PATH = "/cloudtik/data"
DATA_DISK_MOUNT_PATH = "/mnt/cloudtik"
DATA_DISK_MOUNT_PATH_PATTERN = DATA_DISK_MOUNT_PATH + "/data_disk_"

INSPECT_FORMAT = (
    '{'
    '"name":{{json .Name}},'
    '"status":{{json .State.Status}},'
    '"labels":{{json .Config.Labels}},'
    '"ip":{{range .NetworkSettings.Networks}}{{json .IPAddress}}{{end}},'
    '"cpus":{{json .HostConfig.NanoCpus}},'
    '"memory":{{json .HostConfig.Memory}},'
    '"binds":{{json .HostConfig.Binds}}'
    '}')

INSPECT_FORMAT_FOR_EXIST = (
    '{'
    '"name":{{json .Name}}'
    '}')


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


def _get_container_resources(container_object):
    resources = {}
    nano_cpus = container_object.get("cpus", 0)
    if nano_cpus:
        resources["CPU"] = round(nano_cpus / (10 ** 9), 2)
    memory_bytes = container_object.get("memory", 0)
    if memory_bytes:
        resources["memory"] = memory_bytes
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
            # external ip
            external_ip = node_tags.get(VIRTUAL_TAG_EXTERNAL_IP)
            if external_ip:
                node["external_ip"] = external_ip
            matching_nodes.append(node)
    return matching_nodes


class VirtualStateStore(FileStateStore):
    def __init__(self, lock_path, state_path):
        super().__init__(lock_path, state_path)


class VirtualContainerScheduler:
    def __init__(self, provider_config, cluster_name):
        self.provider_config = provider_config
        self.cluster_name = cluster_name

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
            self.call_context, None, docker_config=self.docker_config)

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

    def _create_node(
            self, call_context: CallContext, node_config, tags):
        return self._start_container(call_context, node_config, tags)

    def create_node(self, node_config, tags, count):
        # We should not lock here
        if count <= 0:
            return

        if count == 1:
            self._create_node(self.call_context, node_config, tags)
            return

        # for multi-thread executing of SSH command, we need a new call context
        # whose output is redirected
        call_context = self.call_context.new_call_context()
        call_context.set_output_redirected(True)
        call_context.set_allow_interactive(False)

        launched_nodes = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            indices = range(count)
            futures = {}
            for index in indices:
                futures[index] = executor.submit(
                    self._create_node,
                    call_context=call_context.new_call_context(),
                    node_config=node_config,
                    tags=tags)

            for index, future in futures.items():
                try:
                    node_id = future.result()
                except Exception as e:
                    cli_logger.error("Create node {} failed: {}", index, str(e))
                else:
                    launched_nodes[index] = node_id

        launched = len(launched_nodes)
        if launched < count:
            raise RuntimeError(
                "No enough free nodes. {} nodes requested / {} launched.".format(
                    count, launched))

    def non_terminated_nodes(self, tag_filters):
        tag_filters = {} if tag_filters is None else tag_filters
        if self.cluster_name:
            tag_filters[CLOUDTIK_TAG_CLUSTER_NAME] = self.cluster_name
        with self.lock:
            # list all containers include stopped
            containers = self._list_containers(tag_filters, True)

            # Cannot do cleanup if the tag filters has filters other than cluster name and workspace
            if len(tag_filters) == 2 and (
                    CLOUDTIK_TAG_CLUSTER_NAME in tag_filters and CLOUDTIK_TAG_WORKSPACE_NAME in tag_filters):
                all_node_ids = {container["name"] for container in containers}
                self.state.cleanup(all_node_ids)

            # apply the filters again
            containers = self._apply_filters(containers, tag_filters)

            # Note: All the operations use "name" as the unique node id
            self.cached_nodes = {
                container["name"]: container for container in containers
            }
            return [i["name"] for i in containers]

    def is_running(self, node_id):
        # always get current status
        node = self._get_node(node_id=node_id)
        return _is_running(node)

    def is_terminated(self, node_id):
        # always get current status
        return not self.is_running(node_id)

    def node_tags(self, node_id):
        with self.lock:
            node = self._get_cached_node(node_id)
            return _get_tags(node)

    def external_ip(self, node_id):
        if _is_use_internal_ip(self.provider_config):
            return None
        return self.bridge_ip

    def internal_ip(self, node_id):
        with self.lock:
            node = self._get_cached_node(node_id)
            return node.get("ip")

    def set_node_tags(self, node_id, tags):
        with self.lock:
            node = self._get_cached_node(node_id)
            self._set_node_tags(node_id, tags=tags)
            # update the cached node tags, although it will refresh at next non_terminated_nodes
            FileStateStore.update_node_tags(node, tags)

    def terminate_node(self, node_id):
        # We shall not lock here
        self._terminate_node(self.call_context, node_id)

    def _terminate_node(
            self, call_context: CallContext, node_id):
        with self.lock:
            node = self._get_cached_node(node_id)

        self._stop_container(call_context, node_id, node)

        # shall we remove the node from cached node
        # the cached node list will be refreshed at next non_terminated_nodes
        # The node may already be removed
        # usually not problem, at least we set to "terminated"
        # with self.lock:
        #   node = self._get_cached_node(node_id)
        #   node["state"] = "terminated"

    def terminate_nodes(self, node_ids: List[str]):
        if not node_ids:
            return None

        # for multi-thread executing of SSH command, we need a new call context
        # whose output is redirected
        call_context = self.call_context.new_call_context()
        call_context.set_output_redirected(True)
        call_context.set_allow_interactive(False)

        result = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}
            for node_id in node_ids:
                futures[node_id] = executor.submit(
                    self._terminate_node,
                    call_context=call_context.new_call_context(),
                    node_id=node_id)

            for node_id, future in futures.items():
                try:
                    r = future.result()
                except Exception as e:
                    result[node_id] = e
                    cli_logger.error("Terminate node {} failed: {}", node_id, str(e))
                else:
                    result[node_id] = r
        return result

    def get_node_info(self, node_id):
        with self.lock:
            node = self._get_cached_node(node_id)
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
        # for container scheduler, the node id is container name
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
            return VirtualDockerCommandExecutor(
                call_context, docker_config, True, **common_args)
        else:
            return VirtualDockerCommandExecutor(
                call_context, docker_config, False, **common_args)

    def _is_in_cluster(self):
        # flag set at boostrap cluster config
        return self.provider_config.get("virtual_in_cluster", False)

    @staticmethod
    def _get_state_store(workspace_name, cluster_name):
        virtual_scheduler_lock_path = get_virtual_scheduler_lock_path(
            workspace_name, cluster_name)
        return VirtualStateStore(
            virtual_scheduler_lock_path,
            get_virtual_scheduler_state_path(
                workspace_name, cluster_name))

    @staticmethod
    def _get_state_store_in_cluster(workspace_name, cluster_name):
        virtual_scheduler_lock_path = get_virtual_scheduler_lock_path(
            workspace_name, cluster_name)
        virtual_scheduler_state_path = os.path.join(
            STATE_MOUNT_PATH, get_virtual_scheduler_state_file_name())
        return VirtualStateStore(
            virtual_scheduler_lock_path,
            virtual_scheduler_state_path)

    def _get_scheduler_executor(
            self, call_context: CallContext, container_name, docker_config):
        log_prefix = "ContainerScheduler: "
        if self.cluster_name:
            log_prefix = log_prefix + "{}: ".format(self.cluster_name)

        # bootstrapped from config
        # ssh_user and ssh_private_key need set properly at bootstrap
        # both for working node and head node
        auth_config = self.provider_config.get(AUTH_CONFIG_KEY, {})
        return self.get_command_executor(
            call_context,
            log_prefix=log_prefix,
            node_id=container_name,
            auth_config=auth_config,
            cluster_name=self.cluster_name,
            process_runner=subprocess,
            use_internal_ip=True,
            docker_config=docker_config,
        )

    def _get_new_container_name(self, scheduler_executor):
        retry = 0
        while retry < MAX_CONTAINER_NAME_RETRIES:
            container_name = self._get_random_container_name()
            if not self._is_container_exists(scheduler_executor, container_name):
                return container_name
            retry += 1

        raise RuntimeError("Failed to allocate a container name.")

    def _get_random_container_name(self):
        # the container name prefix with workspace and cluster name
        # with a random string of 8 digits
        workspace_name = self.provider_config["workspace_name"]
        cluster_name = self.cluster_name
        random_id = str(uuid.uuid1())[:8]
        return f"{workspace_name}-{cluster_name}-{random_id}"

    @staticmethod
    def _set_container_resources(node_config, docker_config):
        instance_type = _get_request_instance_type(node_config)
        if instance_type:
            cpus = instance_type.get("CPU", 0)
            if cpus:
                docker_config["cpus"] = cpus

            memory = get_memory_in_bytes(
                instance_type.get("memory"))
            memory_mb = int(memory / (1024 * 1024))
            if memory_mb:
                docker_config["memory"] = str(memory_mb) + "m"

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

    def _start_container(
            self, call_context: CallContext, node_config, tags):
        anonymous_scheduler_executor = self._get_scheduler_executor(
            call_context, None, docker_config=self.docker_config)
        container_name = self._get_new_container_name(anonymous_scheduler_executor)

        # check CLOUDTIK_TAG_NODE_KIND: NODE_KIND_HEAD tag for head
        is_head_node = is_head_node_by_tags(tags)
        shared_memory_ratio = node_config.get("shared_memory_ratio", 0)

        # prepare docker config
        docker_config = _get_merged_docker_config_from_node_config(
            self.docker_config, node_config)
        # make a copy before change it
        docker_config = copy.deepcopy(docker_config)
        docker_config["mounts_mapping"] = False
        docker_config["ipc_mode"] = "private"

        port_mappings = node_config.get("port_mappings")
        if port_mappings:
            docker_config["port_mappings"] = copy.deepcopy(port_mappings)

        if is_head_node and not _is_use_internal_ip(self.provider_config):
            # set bridge ip as external ip store in tags
            tags = {} if tags is None else tags
            tags[VIRTUAL_TAG_EXTERNAL_IP] = self.bridge_ip

        # set labels
        if tags:
            docker_config["labels"] = tags

        self._set_container_resources(node_config, docker_config)

        scheduler_executor = self._get_scheduler_executor(
            call_context, container_name, docker_config=docker_config)

        file_mounts = self._setup_docker_file_mounts(
            node_config, scheduler_executor)

        scheduler_executor.start_container(
            as_head=is_head_node,
            file_mounts=file_mounts,
            shared_memory_ratio=shared_memory_ratio)
        return container_name

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
                raise RuntimeError("No node found with id: {}.".format(node_id))
            tags = self._get_node_tags(node_id, node)
            node["tags"] = tags
            return node

    def _is_container_exists(self, scheduler_executor, container_name):
        # try cache first
        with self.lock:
            if container_name in self.cached_nodes:
                return True

        # try getting the container
        return self._check_container_exist(
            scheduler_executor, container_name)

    @staticmethod
    def _check_container_exist(scheduler_executor, container_name):
        output = scheduler_executor.run_docker_cmd(
            "inspect --format='" + INSPECT_FORMAT_FOR_EXIST + "' " + container_name + " || true")
        if not output:
            return False
        if output.startswith("Error: No such object:"):
            return False

        try:
            container_object = json.loads(output)
        except Exception as e:
            return False

        name = container_object.get("name")
        if not name:
            return False

        if name.startswith("/"):
            name = name[1:]
        if name != container_name:
            return False
        return True

    def _get_container(self, container_name):
        output = self.scheduler_executor.run_docker_cmd(
            "inspect --format='" + INSPECT_FORMAT + "' " + container_name + " || true")
        if not output:
            return None
        if output.startswith("Error: No such object:"):
            return None
        return self._load_container(output)

    def _stop_container(
            self, call_context: CallContext, container_name, container):
        scheduler_executor = self._get_scheduler_executor(
            call_context, container_name, docker_config=self.docker_config)
        scheduler_executor.stop_container()

        delete_on_termination = self.provider_config.get(
            "data_disks.delete_on_termination", True)
        if delete_on_termination:
            self._delete_data_disks(scheduler_executor, container)

    def _delete_data_disks(self, scheduler_executor, container):
        container_object = container["object"]
        binds = container_object.get("binds")
        if not binds:
            return

        name = container["name"]
        for bind in binds:
            bind_parts = bind.split(":")
            bind_src = bind_parts[0]
            bind_dst = bind_parts[1]
            if not bind_dst.startswith(
                    DATA_DISK_MOUNT_PATH_PATTERN) or not bind_src.endswith(name):
                continue

            # delete bind src
            scheduler_executor.run(
                "sudo rm -rf '{path}'".format(path=bind_src),
                run_env="host")

    def _list_containers(self, tag_filters, include_stopped=False):
        # list container tag filters only handles workspace and cluster name
        # These names are set when created which was set as docker labels
        # other filters will not be handled
        effective_filters = {}
        for name in [CLOUDTIK_TAG_WORKSPACE_NAME,
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
        status = container_object.get("status")
        tags = container_object.get("labels", {})
        resources = _get_container_resources(container_object)
        container = {
            "name": name,
            "ip": node_ip,
            "state": status,
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
        tag_filters[CLOUDTIK_TAG_WORKSPACE_NAME] = workspace_name

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
