import base64
import copy
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import unittest
import urllib
from collections import defaultdict
from types import ModuleType

import pytest
import yaml

from jsonschema.exceptions import ValidationError
from unittest.mock import Mock
from subprocess import CalledProcessError
from threading import Thread
from typing import Dict, Callable, List, Optional, Any

from cloudtik.core._private.call_context import CallContext
from cloudtik.core._private.cli_logger import cli_logger, cf
from cloudtik.core._private.cluster.cluster_metrics_updater import ClusterMetricsUpdater
from cloudtik.core._private.cluster.cluster_operator import _should_create_new_head, _set_up_config_for_head_node, \
    POLL_INTERVAL
from cloudtik.core._private.cluster.cluster_scaler import ClusterScaler, NonTerminatedNodes
from cloudtik.core._private.cluster.event_summarizer import EventSummarizer
from cloudtik.core._private.cluster.resource_scaling_policy import ResourceScalingPolicy
from cloudtik.core._private.docker import validate_docker_config, get_docker_host_mount_location, \
    get_docker_host_mount_location_for_object
from cloudtik.core._private.event_system import global_event_system, CreateClusterEvent
from cloudtik.core._private.node.node_updater import NodeUpdater
from cloudtik.core._private.prometheus_metrics import ClusterPrometheusMetrics
from cloudtik.core._private.state.control_state import ControlState
from cloudtik.core._private.state.scaling_state import ScalingStateClient
from cloudtik.core._private.utils import prepare_config, validate_config, fillout_defaults, \
    set_node_type_min_max_workers, DOCKER_CONFIG_KEY, RUNTIME_CONFIG_KEY, get_cluster_uri, hash_launch_conf, \
    hash_runtime_conf, is_docker_enabled, get_commands_to_run, cluster_booting_completed, merge_cluster_config, \
    with_head_node_ip_environment_variables
from cloudtik.core._private.cluster import cluster_operator
from cloudtik.core._private.cluster.cluster_metrics import ClusterMetrics
from cloudtik.core._private.providers import _NODE_PROVIDERS, _get_node_provider, _PROVIDER_HOMES, \
    _load_aws_provider_home

from cloudtik.core.node_provider import NodeProvider
from cloudtik.core.tags import CLOUDTIK_TAG_NODE_KIND, CLOUDTIK_TAG_NODE_STATUS, CLOUDTIK_TAG_USER_NODE_TYPE, \
    CLOUDTIK_TAG_CLUSTER_NAME, STATUS_UNINITIALIZED, STATUS_UPDATE_FAILED, NODE_KIND_HEAD, CLOUDTIK_TAG_LAUNCH_CONFIG, \
    CLOUDTIK_TAG_NODE_NAME, CLOUDTIK_TAG_NODE_NUMBER, CLOUDTIK_TAG_HEAD_NODE_NUMBER, NODE_KIND_WORKER, STATUS_UP_TO_DATE


def mock_node_id() -> bytes:
    """Random node id to pass to cluster_metrics.update."""
    return base64.b64encode(os.urandom(10)).decode("utf-8")


def fill_in_node_ids(provider, cluster_metrics) -> None:
    """For test purposes, we sometimes need to manually fill
    these fields with mocks.
    """
    for node in provider.non_terminated_nodes({}):
        ip = provider.internal_ip(node)
        cluster_metrics.node_id_by_ip[ip] = mock_node_id()


class MockNode:
    def __init__(self, node_id, tags, node_config, node_type,
                 unique_ips=False):
        self.node_id = node_id
        self.state = "pending"
        self.tags = tags
        self.external_ip = "1.2.3.4"
        self.internal_ip = "172.0.0.{}".format(self.node_id)
        if unique_ips:
            self.external_ip = f"1.2.3.{self.node_id}"

        self.node_config = node_config
        self.node_type = node_type

    def matches(self, tags):
        for k, v in tags.items():
            if k not in self.tags or self.tags[k] != v:
                return False
        return True


class MockProcessRunner:
    def __init__(self, fail_cmds=None, cmd_to_callback=None, print_out=False):
        self.calls = []
        self.cmd_to_callback = cmd_to_callback or {
        }  # type: Dict[str, Callable]
        self.print_out = print_out
        self.fail_cmds = fail_cmds or []
        self.call_response = {}
        self.ready_to_run = threading.Event()
        self.ready_to_run.set()

        self.lock = threading.RLock()

    def check_call(self, cmd, *args, **kwargs):
        with self.lock:
            self.ready_to_run.wait()
            self.calls.append(cmd)
            if self.print_out:
                print(f">>>Process runner: Executing \n {str(cmd)}")
            for token in self.cmd_to_callback:
                if token in str(cmd):
                    # Trigger a callback if token is in cmd.
                    # Can be used to simulate background events during a node
                    # update (e.g. node disconnected).
                    callback = self.cmd_to_callback[token]
                    callback()

            for token in self.fail_cmds:
                if token in str(cmd):
                    raise CalledProcessError(1, token,
                                             "Failing command on purpose")

    def check_output(self, cmd):
        with self.lock:
            self.check_call(cmd)
            return_string = "command-output"
            key_to_shrink = None
            for pattern, response_list in self.call_response.items():
                if pattern in str(cmd):
                    return_string = response_list[0]
                    key_to_shrink = pattern
                    break
            if key_to_shrink:
                self.call_response[key_to_shrink] = self.call_response[
                                                        key_to_shrink][1:]
                if len(self.call_response[key_to_shrink]) == 0:
                    del self.call_response[key_to_shrink]

            return return_string.encode()

    def assert_has_call(self,
                        ip: str,
                        pattern: Optional[str] = None,
                        exact: Optional[List[str]] = None):
        """Checks if the given value was called by this process runner.
        NOTE: Either pattern or exact must be specified, not both!
        Args:
            ip: IP address of the node that the given call was executed on.
            pattern: RegEx that matches one specific call.
            exact: List of strings that when joined exactly match one call.
        """
        with self.lock:
            assert bool(pattern) ^ bool(exact), \
                "Must specify either a pattern or exact match."
            debug_output = ""
            if pattern is not None:
                for cmd in self.command_history():
                    if ip in cmd:
                        debug_output += cmd
                        debug_output += "\n"
                    if re.search(pattern, cmd):
                        return True
                else:
                    raise Exception(
                        f"Did not find [{pattern}] in [{debug_output}] for "
                        f"ip={ip}.\n\nFull output: {self.command_history()}")
            elif exact is not None:
                exact_cmd = " ".join(exact)
                for cmd in self.command_history():
                    if ip in cmd:
                        debug_output += cmd
                        debug_output += "\n"
                    if cmd == exact_cmd:
                        return True
                raise Exception(
                    f"Did not find [{exact_cmd}] in [{debug_output}] for "
                    f"ip={ip}.\n\nFull output: {self.command_history()}")

    def assert_not_has_call(self, ip: str, pattern: str):
        """Ensure that the given regex pattern was never called.
        """
        with self.lock:
            out = ""
            for cmd in self.command_history():
                if ip in cmd:
                    out += cmd
                    out += "\n"
            if re.search(pattern, out):
                raise Exception("Found [{}] in [{}] for {}".format(
                    pattern, out, ip))
            else:
                return True

    def clear_history(self):
        with self.lock:
            self.calls = []

    def command_history(self):
        with self.lock:
            return [" ".join(cmd) for cmd in self.calls]

    def respond_to_call(self, pattern, response_list):
        with self.lock:
            self.call_response[pattern] = response_list


class MockProvider(NodeProvider):
    def __init__(self, cache_stopped=False, unique_ips=False):
        self.mock_nodes = {}
        self.next_id = 0
        self.throw = False
        self.error_creates = False
        self.fail_creates = False
        self.ready_to_create = threading.Event()
        self.ready_to_create.set()
        self.cache_stopped = cache_stopped
        self.unique_ips = unique_ips
        self.fail_to_fetch_ip = False
        # Many of these functions are called by node_launcher or updater in
        # different threads. This can be treated as a global lock for
        # everything.
        self.lock = threading.Lock()
        self.num_non_terminated_nodes_calls = 0
        super().__init__(None, None)

    def non_terminated_nodes(self, tag_filters):
        self.num_non_terminated_nodes_calls += 1
        with self.lock:
            if self.throw:
                raise Exception("oops")
            return [
                n.node_id for n in self.mock_nodes.values()
                if n.matches(tag_filters)
                   and n.state not in ["stopped", "terminated"]
            ]

    def non_terminated_node_ips(self, tag_filters):
        with self.lock:
            if self.throw:
                raise Exception("oops")
            return [
                n.internal_ip for n in self.mock_nodes.values()
                if n.matches(tag_filters)
                   and n.state not in ["stopped", "terminated"]
            ]

    def is_running(self, node_id):
        with self.lock:
            return self.mock_nodes[node_id].state == "running"

    def is_terminated(self, node_id):
        with self.lock:
            return self.mock_nodes[node_id].state in ["stopped", "terminated"]

    def node_tags(self, node_id):
        # Don't assume that node providers can retrieve tags from
        # terminated nodes.
        if self.is_terminated(node_id):
            raise Exception(f"The node with id {node_id} has been terminated!")
        with self.lock:
            return self.mock_nodes[node_id].tags

    def internal_ip(self, node_id):
        if self.fail_to_fetch_ip:
            raise Exception("Failed to fetch ip on purpose.")
        if node_id is None:
            # Circumvent test-cases where there's no head node.
            return "mock"
        with self.lock:
            return self.mock_nodes[node_id].internal_ip

    def external_ip(self, node_id):
        with self.lock:
            return self.mock_nodes[node_id].external_ip

    def create_node(self, node_config, tags, count, _skip_wait=False):
        if self.error_creates:
            raise Exception
        if not _skip_wait:
            self.ready_to_create.wait()
        if self.fail_creates:
            return
        with self.lock:
            if self.cache_stopped:
                for node in self.mock_nodes.values():
                    if node.state == "stopped" and count > 0:
                        count -= 1
                        node.state = "pending"
                        node.tags.update(tags)
            for _ in range(count):
                self.mock_nodes[self.next_id] = MockNode(
                    self.next_id,
                    tags.copy(),
                    node_config,
                    tags.get(CLOUDTIK_TAG_USER_NODE_TYPE),
                    unique_ips=self.unique_ips)
                self.next_id += 1

    def set_node_tags(self, node_id, tags):
        with self.lock:
            self.mock_nodes[node_id].tags.update(tags)

    def terminate_node(self, node_id):
        with self.lock:
            if self.cache_stopped:
                self.mock_nodes[node_id].state = "stopped"
            else:
                self.mock_nodes[node_id].state = "terminated"

    def finish_starting_nodes(self):
        with self.lock:
            for node in self.mock_nodes.values():
                if node.state == "pending":
                    node.state = "running"

    def with_environment_variables(self, node_type_config: Dict[str, Any], node_id: str):
        return {}

    def get_node_info(self, node_id: str) -> Dict[str, str]:
        pass


class MockNodeUpdater(NodeUpdater):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def bootstrap_data_disks(self, step_numbers=(1, 1)):
        pass


class MockNodeUpdaterThread(MockNodeUpdater, Thread):
    def __init__(self, *args, **kwargs):
        Thread.__init__(self)
        MockNodeUpdater.__init__(self, *args, **kwargs)
        self.exitcode = -1


class MockClusterScaler(ClusterScaler):
    """Test Cluster Scaler constructed to verify the property that each
    Cluster Scaler update issues at most one provider.non_terminated_nodes call.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_to_find_ip_during_drain = False
        # Testing with maximum verbosity
        self.call_context.cli_logger.set_verbosity(999)

    def _publish_runtime_config(self, *args):
        return

    def _publish_runtime_configs(self):
        return

    def _update(self):
        # Only works with MockProvider
        assert isinstance(self.provider, MockProvider)
        start_calls = self.provider.num_non_terminated_nodes_calls
        super()._update()
        end_calls = self.provider.num_non_terminated_nodes_calls

        # Strict inequality if update is called twice within the throttling
        # interval `self.update_interval_s`
        assert end_calls <= start_calls + 1

    def spawn_updater(self, node_id, setup_commands, start_commands,
                      node_resources, docker_config, call_context):
        # Only works with MockProvider
        assert isinstance(self.provider, MockProvider)
        ip = self.provider.internal_ip(node_id)
        node_type = self._get_node_type(node_id)
        self.node_tracker.track(node_id, ip, node_type)
        head_node_ip = self.provider.internal_ip(
            self.non_terminated_nodes.head_id)
        runtime_hash = self.get_node_runtime_hash(node_id)
        runtime_config = self._get_node_specific_runtime_config(node_id)

        initialization_commands = self._get_node_specific_commands(
            node_id, "worker_initialization_commands")
        environment_variables = with_head_node_ip_environment_variables(
            head_node_ip)
        environment_variables = self._with_cluster_secrets(environment_variables)

        updater = MockNodeUpdaterThread(
            config=self.config,
            call_context=call_context,
            node_id=node_id,
            provider_config=self.config["provider"],
            provider=self.provider,
            auth_config=self.config["auth"],
            cluster_name=self.config["cluster_name"],
            file_mounts=self.config["file_mounts"],
            initialization_commands=initialization_commands,
            setup_commands=setup_commands,
            start_commands=start_commands,
            runtime_hash=runtime_hash,
            file_mounts_contents_hash=self.file_mounts_contents_hash,
            is_head_node=False,
            cluster_synced_files=self.config["cluster_synced_files"],
            rsync_options={
                "rsync_exclude": self.config.get("rsync_exclude"),
                "rsync_filter": self.config.get("rsync_filter")
            },
            process_runner=self.process_runner,
            use_internal_ip=True,
            docker_config=docker_config,
            node_resources=node_resources,
            runtime_config=runtime_config,
            environment_variables=environment_variables)
        updater.start()
        self.updaters[node_id] = updater


SMALL_CLUSTER = {
    "cluster_name": "default",
    "min_workers": 2,
    "max_workers": 2,
    "idle_timeout_minutes": 5,
    "provider": {
        "type": "mock",
        "region": "us-east-1",
        "availability_zone": "us-east-1a",
    },
    "docker": {
        "enabled": True,
        "image": "example",
        "container_name": "mock",
    },
    "auth": {
        "ssh_user": "ubuntu",
        "ssh_private_key": os.devnull,
    },
    "head_node": {
        "TestProp": 1,
    },
    "available_node_types": {
        "head.default": {
            "resources": {},
            "node_config": {
                "head_default_prop": 1
            }
        },
        "worker.default": {
            "min_workers": 2,
            "max_workers": 2,
            "resources": {},
            "node_config": {
                "worker_default_prop": 2
            }
        }
    },
    "file_mounts": {},
    "cluster_synced_files": [],
    "initialization_commands": ["init_cmd"],
    "setup_commands": ["setup_cmd"],
    "head_setup_commands": ["head_setup_cmd"],
    "worker_setup_commands": ["worker_setup_cmd"],
    "head_start_commands": ["head_start_cmd"],
    "worker_start_commands": ["worker_start_cmd"],
    "merged_commands": {},
    "runtime": {
        "types": ["spark", "ganglia"]
    },
}

MOCK_DEFAULT_CONFIG = {
    "cluster_name": "default",
    "max_workers": 2,
    "idle_timeout_minutes": 5,
    "provider": {
        "type": "aws",
        "region": "us-east-1",
        "availability_zone": "us-east-1a",
    },
    "docker": {
        "enabled": True,
        "image": "example",
        "container_name": "mock",
    },
    "auth": {
        "ssh_user": "ubuntu",
        "ssh_private_key": os.devnull,
    },
    "available_node_types": {
        "cloudtik.head.default": {
            "resources": {},
            "node_config": {
                "head_default_prop": 4
            }
        },
        "cloudtik.worker.default": {
            "min_workers": 0,
            "max_workers": 2,
            "resources": {},
            "node_config": {
                "worker_default_prop": 7
            }
        }
    },
    "head_node_type": "cloudtik.head.default",
    "head_node": {},
    "file_mounts": {},
    "cluster_synced_files": [],
    "initialization_commands": [],
    "setup_commands": [],
    "head_setup_commands": [],
    "worker_setup_commands": [],
    "head_start_commands": [],
    "worker_start_commands": [],
}

TYPES_A = {
    "empty_node": {
        "node_config": {
            "FooProperty": 42,
        },
        "resources": {},
        "max_workers": 0,
    },
    "m4.large": {
        "node_config": {},
        "resources": {
            "CPU": 2
        },
        "max_workers": 10,
    },
    "m4.4xlarge": {
        "node_config": {},
        "resources": {
            "CPU": 16
        },
        "max_workers": 8,
    },
    "m4.16xlarge": {
        "node_config": {},
        "resources": {
            "CPU": 64
        },
        "max_workers": 4,
    },
    "p2.xlarge": {
        "node_config": {},
        "resources": {
            "CPU": 16,
            "GPU": 1
        },
        "max_workers": 10,
    },
    "p2.8xlarge": {
        "node_config": {},
        "resources": {
            "CPU": 32,
            "GPU": 8
        },
        "max_workers": 4,
    },
}

MULTI_WORKER_CLUSTER = dict(
    SMALL_CLUSTER, **{
        "available_node_types": TYPES_A,
        "head_node_type": "empty_node"
    })


class ClusterMetricsTest(unittest.TestCase):
    def testHeartbeat(self):
        cluster_metrics = ClusterMetrics()
        cluster_metrics.update_heartbeat("1.1.1.1", mock_node_id(), None)
        cluster_metrics.update_node_resources("1.1.1.1", mock_node_id(), None, {"CPU": 2}, {"CPU": 1}, {})
        cluster_metrics.mark_active("2.2.2.2")
        assert "1.1.1.1" in cluster_metrics.last_heartbeat_time_by_ip
        assert "2.2.2.2" in cluster_metrics.last_heartbeat_time_by_ip
        assert "3.3.3.3" not in cluster_metrics.last_heartbeat_time_by_ip

    def testDebugString(self):
        cluster_metrics = ClusterMetrics()
        cluster_metrics.update_node_resources("1.1.1.1", mock_node_id(), time.time(), {"CPU": 2}, {"CPU": 0}, {})
        cluster_metrics.update_node_resources(
            "2.2.2.2", mock_node_id(), time.time(), {"CPU": 2, "GPU": 16}, {"CPU": 2, "GPU": 2}, {}
        )
        cluster_metrics.update_node_resources(
            "3.3.3.3",
            mock_node_id(),
            time.time(),
            {
                "memory": 1.05 * 1024 * 1024 * 1024,
                "object_store_memory": 2.1 * 1024 * 1024 * 1024,
            },
            {
                "memory": 0,
                "object_store_memory": 1.05 * 1024 * 1024 * 1024,
            },
            {},
        )
        debug = cluster_metrics.info_string()
        assert (
                   "ResourceUsage: 2.0/4.0 CPU, 14.0/16.0 GPU, "
                   "1.05 GiB/1.05 GiB memory, "
                   "1127428915.2/2254857830.4 object_store_memory"
               ) in debug


class CloudTikTestTimeoutException(Exception):
    pass


class CloudTikTest(unittest.TestCase):
    def setUp(self):
        _NODE_PROVIDERS["mock"] = lambda config: self.create_provider
        _PROVIDER_HOMES["mock"] = _load_aws_provider_home
        self.provider = None
        self.tmpdir = tempfile.mkdtemp()

    def waitFor(self, condition, num_retries=50, fail_msg=None):
        for _ in range(num_retries):
            if condition():
                return
            time.sleep(0.1)
        fail_msg = fail_msg or "Timed out waiting for {}".format(condition)
        raise CloudTikTestTimeoutException(fail_msg)

    def waitForNodes(self, expected, comparison=None, tag_filters=None):
        if tag_filters is None:
            tag_filters = {}

        MAX_ITER = 50
        for i in range(MAX_ITER):
            n = len(self.provider.non_terminated_nodes(tag_filters))
            if comparison is None:
                comparison = self.assertEqual
            try:
                comparison(n, expected, msg="Unexpected node quantity.")
                return
            except Exception:
                if i == MAX_ITER - 1:
                    raise
            time.sleep(.1)

    def create_provider(self, config, cluster_name):
        assert self.provider
        return self.provider

    def prepare_mock_config(self, config):
        with_defaults = fillout_defaults(config)
        merge_cluster_config(with_defaults)
        validate_docker_config(with_defaults)
        set_node_type_min_max_workers(with_defaults)
        return with_defaults

    def write_config(self, config, call_prepare_config=True):
        new_config = copy.deepcopy(config)
        if call_prepare_config:
            new_config = self.prepare_mock_config(new_config)
        path = os.path.join(self.tmpdir, "simple.yaml")
        with open(path, "w") as f:
            f.write(yaml.dump(new_config))
        return path

    def get_or_create_head_node(self, config: Dict[str, Any],
                                call_context: CallContext,
                                no_restart: bool,
                                restart_only: bool,
                                yes: bool,
                                _provider: Optional[NodeProvider] = None,
                                _runner: ModuleType = subprocess) -> None:
        """Create the cluster head node, which in turn creates the workers. Only works with MockProvider."""
        assert isinstance(self.provider, MockProvider)
        global_event_system.execute_callback(
            get_cluster_uri(config),
            CreateClusterEvent.cluster_booting_started)
        provider = (_provider or _get_node_provider(config["provider"],
                                                    config["cluster_name"]))

        config = copy.deepcopy(config)
        head_node_tags = {
            CLOUDTIK_TAG_NODE_KIND: NODE_KIND_HEAD,
        }
        nodes = provider.non_terminated_nodes(head_node_tags)
        if len(nodes) > 0:
            head_node = nodes[0]
        else:
            head_node = None

        if not head_node:
            cli_logger.confirm(
                yes,
                "No head node found. "
                "Launching a new cluster.",
                _abort=True)

        if head_node:
            if restart_only:
                cli_logger.confirm(
                    yes,
                    "Updating cluster configuration and "
                    "restarting the cluster runtime. "
                    "Setup commands will not be run due to `{}`.\n",
                    cf.bold("--restart-only"),
                    _abort=True)
            elif no_restart:
                cli_logger.print(
                    "Cluster runtime will not be restarted due "
                    "to `{}`.", cf.bold("--no-restart"))
                cli_logger.confirm(
                    yes,
                    "Updating cluster configuration and "
                    "running setup commands.",
                    _abort=True)
            else:
                cli_logger.print(
                    "Updating cluster configuration and running full setup.")
                cli_logger.confirm(
                    yes,
                    cf.bold("Cluster runtime will be restarted."),
                    _abort=True)

        cli_logger.newline()

        # The "head_node" config stores only the internal head node specific
        # configuration values generated in the runtime, for example IAM
        head_node_config = copy.deepcopy(config.get("head_node", {}))
        head_node_resources = None
        head_node_type = config.get("head_node_type")
        if head_node_type:
            head_node_tags[CLOUDTIK_TAG_USER_NODE_TYPE] = head_node_type
            head_config = config["available_node_types"][head_node_type]
            head_node_config.update(head_config["node_config"])

            # Not necessary to keep in sync with node_launcher.py
            # Keep in sync with cluster_scaler.py _node_resources
            head_node_resources = head_config.get("resources")

        launch_hash = hash_launch_conf(head_node_config, config["auth"])
        creating_new_head = _should_create_new_head(head_node, launch_hash,
                                                    head_node_type, provider)
        if creating_new_head:
            with cli_logger.group("Acquiring an up-to-date head node"):
                global_event_system.execute_callback(
                    get_cluster_uri(config),
                    CreateClusterEvent.acquiring_new_head_node)
                if head_node is not None:
                    cli_logger.confirm(
                        yes, "Relaunching the head node.", _abort=True)

                    provider.terminate_node(head_node)
                    cli_logger.print("Terminated head node {}", head_node)

                head_node_tags[CLOUDTIK_TAG_LAUNCH_CONFIG] = launch_hash
                head_node_tags[CLOUDTIK_TAG_NODE_NAME] = "cloudtik-{}-head".format(
                    config["cluster_name"])
                head_node_tags[CLOUDTIK_TAG_NODE_STATUS] = STATUS_UNINITIALIZED
                head_node_tags[CLOUDTIK_TAG_NODE_NUMBER] = str(CLOUDTIK_TAG_HEAD_NODE_NUMBER)
                provider.create_node(head_node_config, head_node_tags, 1)
                cli_logger.print("Launched a new head node")

                start = time.time()
                head_node = None
                with cli_logger.group("Fetching the new head node"):
                    while True:
                        if time.time() - start > 50:
                            cli_logger.abort("Head node fetch timed out. "
                                             "Failed to create head node.")
                        nodes = provider.non_terminated_nodes(head_node_tags)
                        if len(nodes) == 1:
                            head_node = nodes[0]
                            break
                        time.sleep(POLL_INTERVAL)
                cli_logger.newline()

        global_event_system.execute_callback(
            get_cluster_uri(config),
            CreateClusterEvent.head_node_acquired)

        with cli_logger.group(
                "Setting up head node",
                _numbered=("<>", 1, 1),
                # cf.bold(provider.node_tags(head_node)[CLOUDTIK_TAG_NODE_NAME]),
                _tags=dict()):  # add id, ARN to tags?

            # TODO: right now we always update the head node even if the
            # hash matches.
            # We could prompt the user for what they want to do here.
            # No need to pass in cluster_sync_files because we use this
            # hash to set up the head node
            (runtime_hash,
             file_mounts_contents_hash,
             runtime_hash_for_node_types) = hash_runtime_conf(
                config["file_mounts"], None, config)
            # Even we don't need controller on head, we still need config and cluster keys on head
            # because head depends a lot on the cluster config file and cluster keys to do cluster
            # operations and connect to the worker.

            # Return remote_config_file to avoid prematurely closing it.
            config, remote_config_file = _set_up_config_for_head_node(
                config, provider, no_restart)
            cli_logger.print("Prepared bootstrap config")

            if restart_only:
                # Docker may re-launch nodes, requiring setup
                # commands to be rerun.
                if is_docker_enabled(config):
                    setup_commands = get_commands_to_run(config, "head_setup_commands")
                else:
                    setup_commands = []
                start_commands = get_commands_to_run(config, "head_start_commands")
            # If user passed in --no-restart and we're not creating a new head,
            # omit start commands.
            elif no_restart and not creating_new_head:
                setup_commands = get_commands_to_run(config, "head_setup_commands")
                start_commands = []
            else:
                setup_commands = get_commands_to_run(config, "head_setup_commands")
                start_commands = get_commands_to_run(config, "head_start_commands")

            initialization_commands = get_commands_to_run(config, "head_initialization_commands")
            updater = MockNodeUpdaterThread(
                config=config,
                call_context=call_context,
                node_id=head_node,
                provider_config=config["provider"],
                provider=provider,
                auth_config=config["auth"],
                cluster_name=config["cluster_name"],
                file_mounts=config["file_mounts"],
                initialization_commands=initialization_commands,
                setup_commands=setup_commands,
                start_commands=start_commands,
                process_runner=_runner,
                runtime_hash=runtime_hash,
                file_mounts_contents_hash=file_mounts_contents_hash,
                is_head_node=True,
                node_resources=head_node_resources,
                rsync_options={
                    "rsync_exclude": config.get("rsync_exclude"),
                    "rsync_filter": config.get("rsync_filter")
                },
                docker_config=config.get(DOCKER_CONFIG_KEY),
                restart_only=restart_only,
                runtime_config=config.get(RUNTIME_CONFIG_KEY))
            updater.start()
            updater.join()

            # Refresh the node cache so we see the external ip if available
            provider.non_terminated_nodes(head_node_tags)

            if updater.exitcode != 0:
                # todo: this does not follow the mockup and is not good enough
                cli_logger.abort("Failed to setup head node.")
                sys.exit(1)

        global_event_system.execute_callback(
            get_cluster_uri(config),
            CreateClusterEvent.cluster_booting_completed, {
                "head_node_id": head_node,
            })

        cluster_booting_completed(config, head_node)

        cli_logger.newline()
        successful_msg = "Successfully started cluster: {}.".format(config["cluster_name"])
        cli_logger.success("-" * len(successful_msg))
        cli_logger.success(successful_msg)
        cli_logger.success("-" * len(successful_msg))

    def testValidateDefaultConfig(self):
        config = {"provider": {
            "type": "aws",
            "region": "us-east-1",
            "availability_zone": "us-east-1a",
        }}
        config = prepare_config(config)
        try:
            validate_config(config)
        except ValidationError:
            self.fail("Default config did not pass validation test!")

    def testValidation(self):
        """Ensures that schema validation is working."""
        config = copy.deepcopy(MOCK_DEFAULT_CONFIG)
        try:
            validate_config(config)
        except Exception:
            self.fail("Test config did not pass validation test!")

        config["blah"] = "blah"
        try:
            validate_config(config)
        except Exception:
            self.fail("Config should allow additional properties!")
        del config["blah"]

        del config["cluster_name"]
        with pytest.raises(ValidationError):
            validate_config(config)

        config = copy.deepcopy(MOCK_DEFAULT_CONFIG)
        del config["provider"]
        with pytest.raises(ValidationError):
            validate_config(config)

    def testGetRunningHeadNode(self):
        config = copy.deepcopy(SMALL_CLUSTER)
        self.provider = MockProvider()

        # Node 0 is failed.
        self.provider.create_node({}, {
            CLOUDTIK_TAG_CLUSTER_NAME: "default",
            CLOUDTIK_TAG_NODE_KIND: "head",
            CLOUDTIK_TAG_NODE_STATUS: "update-failed"
        }, 1)

        # `_allow_uninitialized_state` should return the head node
        # in the `update-failed` state.
        allow_failed = cluster_operator._get_running_head_node(
            config,
            _provider=self.provider,
            _allow_uninitialized_state=True)

        assert allow_failed == 0

        # Node 1 is okay.
        self.provider.create_node({}, {
            CLOUDTIK_TAG_CLUSTER_NAME: "default",
            CLOUDTIK_TAG_NODE_KIND: "head",
            CLOUDTIK_TAG_NODE_STATUS: "up-to-date"
        }, 1)

        node = cluster_operator._get_running_head_node(
            config,
            _provider=self.provider)

        assert node == 1

        # `_allow_uninitialized_state` should return the up-to-date head node
        # if it is present.
        optionally_failed = cluster_operator._get_running_head_node(
            config,
            _provider=self.provider,
            _allow_uninitialized_state=True)

        assert optionally_failed == 1

    def testDefaultMinMaxWorkers(self):
        config = copy.deepcopy(MOCK_DEFAULT_CONFIG)
        config = prepare_config(config)
        node_types = config["available_node_types"]
        head_node_config = node_types["cloudtik.head.default"]
        assert head_node_config["min_workers"] == 0
        assert head_node_config["max_workers"] == 0

    def testGetOrCreateHeadNode(self):
        config = copy.deepcopy(SMALL_CLUSTER)
        config = self.prepare_mock_config(config)
        head_run_option = "--kernel-memory=10g"
        standard_run_option = "--memory-swap=5g"
        config["docker"]["head_run_options"] = [head_run_option]
        config["docker"]["run_options"] = [standard_run_option]
        self.provider = MockProvider()
        runner = MockProcessRunner()
        runner.respond_to_call("json .Mounts", ["[]"])
        # Two initial calls to rsync, + 2 more calls during run_init
        runner.respond_to_call(".State.Running", ["false", "false", "false", "false"])
        runner.respond_to_call("json .Config.Env", ["[]"])

        def _create_node(node_config, tags, count, _skip_wait=False):
            assert tags[CLOUDTIK_TAG_NODE_STATUS] == STATUS_UNINITIALIZED
            if not _skip_wait:
                self.provider.ready_to_create.wait()
            if self.provider.fail_creates:
                return
            with self.provider.lock:
                if self.provider.cache_stopped:
                    for node in self.provider.mock_nodes.values():
                        if node.state == "stopped" and count > 0:
                            count -= 1
                            node.state = "pending"
                            node.tags.update(tags)
                for _ in range(count):
                    self.provider.mock_nodes[self.provider.next_id] = MockNode(
                        self.provider.next_id,
                        tags.copy(),
                        node_config,
                        tags.get(CLOUDTIK_TAG_USER_NODE_TYPE),
                        unique_ips=self.provider.unique_ips,
                    )
                    self.provider.next_id += 1

        self.provider.create_node = _create_node
        call_context = CallContext()
        call_context._allow_interactive = False
        self.get_or_create_head_node(config, call_context, no_restart=False, restart_only=False, yes=True,
                                     _provider=self.provider, _runner=runner)
        self.waitForNodes(1)
        runner.assert_has_call("1.2.3.4", "init_cmd")
        runner.assert_has_call("1.2.3.4", "head_setup_cmd")
        runner.assert_has_call("1.2.3.4", "head_start_cmd")
        runner.assert_has_call("1.2.3.4", pattern="docker run")
        runner.assert_has_call("1.2.3.4", pattern=head_run_option)
        runner.assert_has_call("1.2.3.4", pattern=standard_run_option)

        docker_mount_prefix_for_object = get_docker_host_mount_location_for_object(
            SMALL_CLUSTER["cluster_name"],
            "~/cloudtik_bootstrap_key.pem"
        )
        pattern = f"-v {docker_mount_prefix_for_object}/~/cloudtik_bootstrap_config"
        runner.assert_not_has_call(
            "1.2.3.4", pattern=pattern
        )

        pattern = f"rsync -e.*docker exec -i.*{docker_mount_prefix_for_object}/~/cloudtik_bootstrap_key.pem"
        runner.assert_has_call(
            "1.2.3.4", pattern=pattern
        )
        docker_mount_prefix_for_object = get_docker_host_mount_location_for_object(
            SMALL_CLUSTER["cluster_name"],
            "~/cloudtik_bootstrap_config.yaml"
        )
        pattern = f"rsync -e.*docker exec -i.*{docker_mount_prefix_for_object}/~/cloudtik_bootstrap_config.yaml"
        runner.assert_has_call(
            "1.2.3.4", pattern=pattern
        )
        return config

    def testGetOrCreateHeadNodeFromStopped(self):
        config = self.testGetOrCreateHeadNode()
        config["from"] = None
        self.provider.cache_stopped = True
        existing_nodes = self.provider.non_terminated_nodes({})
        assert len(existing_nodes) == 1
        self.provider.terminate_node(existing_nodes[0])
        runner = MockProcessRunner()
        runner.respond_to_call("json .Mounts", ["[]"])
        # Two initial calls to rsync, + 2 more calls during run_init
        runner.respond_to_call(".State.Running", ["false", "false", "false", "false"])
        runner.respond_to_call("json .Config.Env", ["[]"])
        call_context = CallContext()
        self.get_or_create_head_node(config, call_context, no_restart=False, restart_only=False, yes=True,
                                     _provider=self.provider, _runner=runner)
        self.waitForNodes(1)
        # Init & Setup commands must be run for Docker!
        runner.assert_has_call("1.2.3.4", "init_cmd")
        runner.assert_has_call("1.2.3.4", "head_setup_cmd")
        runner.assert_has_call("1.2.3.4", "head_start_cmd")
        runner.assert_has_call("1.2.3.4", pattern="docker run")

        docker_mount_prefix_for_object = get_docker_host_mount_location_for_object(
            SMALL_CLUSTER["cluster_name"],
            "~/cloudtik_bootstrap_key.pem"
        )
        pattern = f"-v {docker_mount_prefix_for_object}/~/cloudtik_bootstrap_config"
        runner.assert_not_has_call(
            "1.2.3.4", pattern=pattern
        )
        pattern = f"rsync -e.*docker exec -i.*{docker_mount_prefix_for_object}/~/cloudtik_bootstrap_key.pem"
        runner.assert_has_call(
            "1.2.3.4", pattern=pattern
        )
        docker_mount_prefix_for_object = get_docker_host_mount_location_for_object(
            SMALL_CLUSTER["cluster_name"],
            "~/cloudtik_bootstrap_config.yaml"
        )
        pattern = f"rsync -e.*docker exec -i.*{docker_mount_prefix_for_object}/~/cloudtik_bootstrap_config.yaml"
        runner.assert_has_call(
            "1.2.3.4", pattern=pattern
        )

        docker_mount_prefix = get_docker_host_mount_location(
            SMALL_CLUSTER["cluster_name"]
        )
        # This section of code ensures that the following order of commands are executed:
        # 1. mkdir -p {docker_mount_prefix}
        # 2. rsync bootstrap files (over ssh)
        # 3. rsync bootstrap files into container
        commands_with_mount = [
            (i, cmd)
            for i, cmd in enumerate(runner.command_history())
            if docker_mount_prefix in cmd
        ]
        rsync_commands = [x for x in commands_with_mount if "rsync --rsh" in x[1]]
        copy_into_container = [
            x
            for x in commands_with_mount
            if re.search("rsync -e.*docker exec -i", x[1])
        ]
        first_mkdir = min(x[0] for x in commands_with_mount if "mkdir" in x[1])
        docker_run_cmd_indx = [
            i for i, cmd in enumerate(runner.command_history()) if "docker run" in cmd
        ][0]
        for file_to_check in ["cloudtik_bootstrap_config.yaml", "cloudtik_bootstrap_key.pem"]:
            first_rsync = min(
                x[0] for x in rsync_commands if "cloudtik_bootstrap_config.yaml" in x[1]
            )
            first_cp = min(x[0] for x in copy_into_container if file_to_check in x[1])
            # Ensures that `mkdir -p` precedes `docker run` because Docker
            # will auto-create the folder with wrong permissions.
            assert first_mkdir < docker_run_cmd_indx
            # Ensures that the folder is created before running rsync.
            assert first_mkdir < first_rsync
            # Checks that the file is present before copying into the container
            assert first_rsync < first_cp

    def testGetOrCreateHeadNodeFromStoppedRestartOnly(self):
        config = self.testGetOrCreateHeadNode()
        config["from"] = None
        self.provider.cache_stopped = True
        existing_nodes = self.provider.non_terminated_nodes({})
        assert len(existing_nodes) == 1
        self.provider.terminate_node(existing_nodes[0])
        runner = MockProcessRunner()
        runner.respond_to_call("json .Mounts", ["[]"])
        # Two initial calls to rsync, + 2 more calls during run_init
        runner.respond_to_call(".State.Running", ["false", "false", "false", "false"])
        runner.respond_to_call("json .Config.Env", ["[]"])
        call_context = CallContext()
        self.get_or_create_head_node(config, call_context, no_restart=False, restart_only=True, yes=True,
                                     _provider=self.provider, _runner=runner)
        self.waitForNodes(1)
        runner.assert_has_call("1.2.3.4", "init_cmd")
        runner.assert_has_call("1.2.3.4", "head_start_cmd")

    def testValidateNetworkConfig(self):
        web_yaml = ("https://raw.githubusercontent.com/oap-project/cloudtik/main/python/cloudtik/templates/aws/small"
                    ".yaml")
        response = urllib.request.urlopen(web_yaml, timeout=5)
        content = response.read()
        with tempfile.TemporaryFile() as f:
            f.write(content)
            f.seek(0)
            config = yaml.safe_load(f)
        config = prepare_config(config)
        try:
            validate_config(config)
        except Exception:
            self.fail("Config did not pass validation test!")

    def ScaleUpHelper(self, disable_node_updaters):
        config = copy.deepcopy(SMALL_CLUSTER)
        config["provider"]["disable_node_updaters"] = disable_node_updaters
        config_path = self.write_config(config)
        self.provider = MockProvider()
        runner = MockProcessRunner()
        mock_metrics = Mock(spec=ClusterPrometheusMetrics())
        cluster_metrics = ClusterMetrics()
        event_summarizer = EventSummarizer()
        control_state = ControlState()
        cluster_scaler = MockClusterScaler(
            config_path,
            cluster_metrics,
            ClusterMetricsUpdater(cluster_metrics, event_summarizer, control_state),
            ResourceScalingPolicy("1.2.3.4", ScalingStateClient.create_from(ControlState())),
            max_failures=0,
            process_runner=runner,
            update_interval_s=0,
            prometheus_metrics=mock_metrics,
        )
        assert len(self.provider.non_terminated_nodes({})) == 0
        cluster_scaler.update()
        self.waitForNodes(2, tag_filters={CLOUDTIK_TAG_USER_NODE_TYPE: "worker.default"})

        assert mock_metrics.worker_create_node_time.observe.call_count == 3
        cluster_scaler.update()
        # The two cluster_scaler update iterations in this test led to two
        # observations of the update time.
        self.waitForNodes(2, tag_filters={CLOUDTIK_TAG_USER_NODE_TYPE: "worker.default"})

        # running_workers metric should be set to 2
        mock_metrics.running_workers.set.assert_called_with(3)

        if disable_node_updaters:
            # Node Updaters have NOT been invoked because they were explicitly
            # disabled.
            time.sleep(1)
            assert len(runner.calls) == 0
            # Nodes were created in uninitialized and not updated.
            self.waitForNodes(
                3, tag_filters={CLOUDTIK_TAG_NODE_STATUS: STATUS_UNINITIALIZED}
            )
        else:
            # Node Updaters have been invoked.
            self.waitFor(lambda: len(runner.calls) > 0)
            # The updates failed. Key thing is that the updates completed.
            self.waitForNodes(
                2, tag_filters={CLOUDTIK_TAG_NODE_STATUS: STATUS_UPDATE_FAILED}
            )
        assert mock_metrics.drain_node_exceptions.inc.call_count == 0

    def testScaleUp(self):
        self.ScaleUpHelper(disable_node_updaters=False)

    def testScaleUpNoUpdaters(self):
        self.ScaleUpHelper(disable_node_updaters=True)

    def testDockerFileMountsAdded(self):
        config = copy.deepcopy(SMALL_CLUSTER)
        config["file_mounts"] = {"source": "/dev/null"}
        config = self.prepare_mock_config(config)
        self.provider = MockProvider()
        runner = MockProcessRunner()
        mounts = [
            {
                "Type": "bind",
                "Source": "/sys",
                "Destination": "/sys",
                "Mode": "ro",
                "RW": False,
                "Propagation": "rprivate",
            }
        ]
        runner.respond_to_call("json .Mounts", [json.dumps(mounts)])
        # Two initial calls to rsync, +1 more call during run_init
        runner.respond_to_call(".State.Running", ["false", "false", "true", "true"])
        runner.respond_to_call("json .Config.Env", ["[]"])
        self.get_or_create_head_node(config, CallContext(), no_restart=False, restart_only=False, yes=True,
                                     _provider=self.provider, _runner=runner)
        self.waitForNodes(1)
        runner.assert_has_call("1.2.3.4", "init_cmd")
        runner.assert_has_call("1.2.3.4", "head_setup_cmd")
        runner.assert_has_call("1.2.3.4", "head_start_cmd")
        runner.assert_has_call("1.2.3.4", pattern="docker stop")
        runner.assert_has_call("1.2.3.4", pattern="docker run")

        docker_mount_prefix_for_object = get_docker_host_mount_location_for_object(
            SMALL_CLUSTER["cluster_name"],
            "~/cloudtik_bootstrap_key.pem"
        )
        pattern = f"-v {docker_mount_prefix_for_object}/~/cloudtik_bootstrap_config"
        runner.assert_not_has_call(
            "1.2.3.4", pattern=pattern
        )

        pattern = f"rsync -e.*docker exec -i.*{docker_mount_prefix_for_object}/~/cloudtik_bootstrap_key.pem"
        runner.assert_has_call(
            "1.2.3.4", pattern=pattern
        )
        docker_mount_prefix_for_object = get_docker_host_mount_location_for_object(
            SMALL_CLUSTER["cluster_name"],
            "~/cloudtik_bootstrap_config.yaml"
        )
        pattern = f"rsync -e.*docker exec -i.*{docker_mount_prefix_for_object}/~/cloudtik_bootstrap_config.yaml"
        runner.assert_has_call(
            "1.2.3.4", pattern=pattern
        )

    def testScaleDownMaxWorkers(self):
        """Tests terminating nodes due to max_nodes per type."""
        config = copy.deepcopy(MULTI_WORKER_CLUSTER)
        config["available_node_types"]["m4.large"]["min_workers"] = 3
        config["available_node_types"]["m4.large"]["max_workers"] = 3
        config["available_node_types"]["m4.large"]["resources"] = {}
        config["available_node_types"]["m4.16xlarge"]["resources"] = {}
        config["available_node_types"]["p2.xlarge"]["min_workers"] = 5
        config["available_node_types"]["p2.xlarge"]["max_workers"] = 8
        config["available_node_types"]["p2.xlarge"]["resources"] = {}
        config["available_node_types"]["p2.8xlarge"]["min_workers"] = 2
        config["available_node_types"]["p2.8xlarge"]["max_workers"] = 4
        config["available_node_types"]["p2.8xlarge"]["resources"] = {}
        config["max_workers"] = 13

        config_path = self.write_config(config)
        config = self.prepare_mock_config(config)
        self.provider = MockProvider()
        runner = MockProcessRunner()
        runner.respond_to_call("json .Config.Env", ["[]" for i in range(15)])
        self.get_or_create_head_node(config, CallContext(), no_restart=False, restart_only=False, yes=True,
                                     _provider=self.provider, _runner=runner)
        self.waitForNodes(1)
        cluster_metrics = ClusterMetrics()
        event_summarizer = EventSummarizer()
        control_state = ControlState()
        cluster_scaler = MockClusterScaler(
            config_path,
            cluster_metrics,
            ClusterMetricsUpdater(cluster_metrics, event_summarizer, control_state),
            ResourceScalingPolicy("1.2.3.4", ScalingStateClient.create_from(ControlState())),
            max_failures=0,
            max_concurrent_launches=13,
            max_launch_batch=13,
            process_runner=runner,
            update_interval_s=0,
        )
        nodes_id = cluster_scaler.provider.non_terminated_nodes({})
        for node_id in nodes_id:
            cluster_scaler.provider.terminate_node(node_id)
        cluster_scaler.update()
        self.provider = cluster_scaler.provider
        self.waitForNodes(12)
        assert cluster_scaler.pending_launches.value == 0
        assert (
                len(
                    self.provider.non_terminated_nodes(
                        {CLOUDTIK_TAG_NODE_KIND: NODE_KIND_WORKER}
                    )
                )
                == 12
        )

        # Terminate some nodes
        config["available_node_types"]["m4.large"]["min_workers"] = 2  # 3
        config["available_node_types"]["m4.large"]["max_workers"] = 2
        config["available_node_types"]["p2.8xlarge"]["min_workers"] = 0  # 2
        config["available_node_types"]["p2.8xlarge"]["max_workers"] = 0
        # And spawn one.
        config["available_node_types"]["p2.xlarge"]["min_workers"] = 6  # 5
        config["available_node_types"]["p2.xlarge"]["max_workers"] = 6
        config["from"] = None
        self.write_config(config)
        fill_in_node_ids(self.provider, cluster_metrics)
        cluster_scaler.reset(errors_fatal=False)
        cluster_scaler.update()
        self.waitFor(lambda: cluster_scaler.pending_launches.value == 0)
        self.waitForNodes(10, tag_filters={CLOUDTIK_TAG_NODE_KIND: NODE_KIND_WORKER})
        assert cluster_scaler.pending_launches.value == 0
        events = cluster_scaler.event_summarizer.summary()
        assert "Removing 1 nodes of type m4.large (max_workers_per_type)." in events
        assert "Removing 2 nodes of type p2.8xlarge (max_workers_per_type)." in events

        node_type_counts = defaultdict(int)
        for node_id in NonTerminatedNodes(self.provider).worker_ids:
            tags = self.provider.node_tags(node_id)
            if CLOUDTIK_TAG_USER_NODE_TYPE in tags:
                node_type = tags[CLOUDTIK_TAG_USER_NODE_TYPE]
                node_type_counts[node_type] += 1
        assert node_type_counts == {'empty_node': 1, 'm4.large': 2, 'p2.xlarge': 6, 'worker.default': 1}

    def testSetupCommandsWithNoNodeCaching(self):
        config = copy.deepcopy(SMALL_CLUSTER)
        config["min_workers"] = 1
        config["max_workers"] = 1
        config_path = self.write_config(config)
        self.provider = MockProvider(cache_stopped=False)
        runner = MockProcessRunner()
        runner.respond_to_call("json .Config.Env", ["[]" for i in range(2)])
        cluster_metrics = ClusterMetrics()
        event_summarizer = EventSummarizer()
        control_state = ControlState()
        cluster_scaler = MockClusterScaler(
            config_path,
            cluster_metrics,
            ClusterMetricsUpdater(cluster_metrics, event_summarizer, control_state),
            ResourceScalingPolicy("1.2.3.4", ScalingStateClient.create_from(ControlState())),
            max_failures=0,
            process_runner=runner,
            update_interval_s=0,
        )
        nodes_id = cluster_scaler.provider.non_terminated_nodes({})
        for node_id in nodes_id:
            cluster_scaler.provider.terminate_node(node_id)
        cluster_scaler.update()
        self.provider = cluster_scaler.provider
        self.waitForNodes(2)
        self.provider.finish_starting_nodes()
        cluster_scaler.update()
        self.waitForNodes(1, tag_filters={CLOUDTIK_TAG_NODE_STATUS: STATUS_UP_TO_DATE})
        if nodes_id:
            worker_id = nodes_id[-1] + 2
        else:
            worker_id = 1
        worker_ip = "172.0.0.{}".format(worker_id)
        runner.assert_has_call(worker_ip, "init_cmd")
        runner.assert_has_call(worker_ip, "setup_cmd")
        runner.assert_has_call(worker_ip, "worker_setup_cmd")
        runner.assert_has_call(worker_ip, "worker_start_cmd")

    def testSetupCommandsWithStoppedNodeCachingNoDocker(self):
        file_mount_dir = tempfile.mkdtemp()
        config = copy.deepcopy(SMALL_CLUSTER)
        del config["docker"]
        config["file_mounts"] = {"/root/test-folder": file_mount_dir}
        config["file_mounts_sync_continuously"] = True
        config["min_workers"] = 1
        config["max_workers"] = 1
        config_path = self.write_config(config)
        self.provider = MockProvider(cache_stopped=True)
        runner = MockProcessRunner()
        runner.respond_to_call("json .Config.Env", ["[]" for i in range(3)])
        cluster_metrics = ClusterMetrics()
        event_summarizer = EventSummarizer()
        control_state = ControlState()
        cluster_scaler = MockClusterScaler(
            config_path,
            cluster_metrics,
            ClusterMetricsUpdater(cluster_metrics, event_summarizer, control_state),
            ResourceScalingPolicy("1.2.3.4", ScalingStateClient.create_from(ControlState())),
            max_failures=0,
            process_runner=runner,
            update_interval_s=0,
        )
        nodes_id = cluster_scaler.provider.non_terminated_nodes({})
        for node_id in nodes_id:
            cluster_scaler.provider.terminate_node(node_id)
        cluster_scaler.update()
        self.provider = cluster_scaler.provider
        self.waitForNodes(2)
        self.provider.finish_starting_nodes()
        cluster_scaler.update()
        self.waitForNodes(1, tag_filters={CLOUDTIK_TAG_NODE_STATUS: STATUS_UP_TO_DATE})
        if nodes_id:
            worker_id = nodes_id[-1] + 2
        else:
            worker_id = 1
        worker_ip = "172.0.0.{}".format(worker_id)
        runner.assert_has_call(worker_ip, "init_cmd")
        runner.assert_has_call(worker_ip, "setup_cmd")
        runner.assert_has_call(worker_ip, "worker_setup_cmd")
        runner.assert_has_call(worker_ip, "worker_start_cmd")

        # Check the node was indeed reused
        nodes_id = self.provider.non_terminated_nodes({})
        for node_id in nodes_id:
            self.provider.terminate_node(node_id)
        cluster_scaler.update()
        self.waitForNodes(2)
        runner.clear_history()
        self.provider.finish_starting_nodes()
        cluster_scaler.update()
        self.waitForNodes(1, tag_filters={CLOUDTIK_TAG_NODE_STATUS: STATUS_UP_TO_DATE})
        runner.assert_not_has_call(worker_ip, "init_cmd")
        runner.assert_not_has_call(worker_ip, "setup_cmd")
        runner.assert_not_has_call(worker_ip, "worker_setup_cmd")
        runner.assert_has_call(worker_ip, "worker_start_cmd")

        with open(f"{file_mount_dir}/new_file", "w") as f:
            f.write("abcdefgh")

        # Check that run_init happens when file_mounts have updated
        nodes_id = self.provider.non_terminated_nodes({})
        for node_id in nodes_id:
            self.provider.terminate_node(node_id)
        cluster_scaler.update()
        self.waitForNodes(2)
        runner.clear_history()
        self.provider.finish_starting_nodes()
        cluster_scaler.update()
        self.waitForNodes(1, tag_filters={CLOUDTIK_TAG_NODE_STATUS: STATUS_UP_TO_DATE})
        runner.assert_not_has_call(worker_ip, "init_cmd")
        runner.assert_not_has_call(worker_ip, "setup_cmd")
        runner.assert_not_has_call(worker_ip, "worker_setup_cmd")
        runner.assert_has_call(worker_ip, "worker_start_cmd")

        runner.clear_history()
        cluster_scaler.update()
        runner.assert_not_has_call(worker_ip, "setup_cmd")

        # We did not start any other nodes
        next_ip = "172.0.0.{}".format(worker_id + 1)
        runner.assert_not_has_call(next_ip, " ")

    def testSetupCommandsWithStoppedNodeCachingDocker(self):
        # NOTE(ilr) Setup & Init commands **should** run with stopped nodes
        # when Docker is in use.
        file_mount_dir = tempfile.mkdtemp()
        config = copy.deepcopy(SMALL_CLUSTER)
        config["file_mounts"] = {"/root/test-folder": file_mount_dir}
        config["file_mounts_sync_continuously"] = True
        config["min_workers"] = 1
        config["max_workers"] = 1
        config_path = self.write_config(config)
        self.provider = MockProvider(cache_stopped=True)
        runner = MockProcessRunner()
        runner.respond_to_call("json .Config.Env", ["[]" for i in range(3)])
        cluster_metrics = ClusterMetrics()
        event_summarizer = EventSummarizer()
        control_state = ControlState()
        cluster_scaler = MockClusterScaler(
            config_path,
            cluster_metrics,
            ClusterMetricsUpdater(cluster_metrics, event_summarizer, control_state),
            ResourceScalingPolicy("1.2.3.4", ScalingStateClient.create_from(ControlState())),
            max_failures=0,
            process_runner=runner,
            update_interval_s=0,
        )
        cluster_scaler.update()
        self.provider = cluster_scaler.provider
        self.waitForNodes(2)
        self.provider.finish_starting_nodes()
        cluster_scaler.update()
        self.waitForNodes(1, tag_filters={CLOUDTIK_TAG_NODE_STATUS: STATUS_UP_TO_DATE})
        nodes_id = self.provider.non_terminated_nodes({})
        if nodes_id:
            worker_id = nodes_id[-1] + 2
        else:
            worker_id = 1
        worker_ip = "172.0.0.{}".format(worker_id)
        runner.assert_has_call(worker_ip, "init_cmd")
        runner.assert_has_call(worker_ip, "setup_cmd")
        runner.assert_has_call(worker_ip, "worker_setup_cmd")
        runner.assert_has_call(worker_ip, "worker_start_cmd")
        runner.assert_has_call(worker_ip, "docker run")

        # Check the node was indeed reused
        nodes_id = self.provider.non_terminated_nodes({})
        for node_id in nodes_id:
            self.provider.terminate_node(node_id)
        cluster_scaler.update()
        self.waitForNodes(2)
        runner.clear_history()
        self.provider.finish_starting_nodes()
        cluster_scaler.update()
        self.waitForNodes(1, tag_filters={CLOUDTIK_TAG_NODE_STATUS: STATUS_UP_TO_DATE})
        # These all must happen when the node is stopped and resued
        runner.assert_has_call(worker_ip, "init_cmd")
        runner.assert_has_call(worker_ip, "setup_cmd")
        runner.assert_has_call(worker_ip, "worker_setup_cmd")
        runner.assert_has_call(worker_ip, "worker_start_cmd")
        runner.assert_has_call(worker_ip, "docker run")

        with open(f"{file_mount_dir}/new_file", "w") as f:
            f.write("abcdefgh")

        # Check that run_init happens when file_mounts have updated
        nodes_id = self.provider.non_terminated_nodes({})
        for node_id in nodes_id:
            self.provider.terminate_node(node_id)
        cluster_scaler.update()
        self.waitForNodes(2)
        runner.clear_history()
        self.provider.finish_starting_nodes()
        cluster_scaler.update()
        self.waitForNodes(1, tag_filters={CLOUDTIK_TAG_NODE_STATUS: STATUS_UP_TO_DATE})
        runner.assert_has_call(worker_ip, "init_cmd")
        runner.assert_has_call(worker_ip, "setup_cmd")
        runner.assert_has_call(worker_ip, "worker_setup_cmd")
        runner.assert_has_call(worker_ip, "worker_start_cmd")
        runner.assert_has_call(worker_ip, "docker run")

        docker_run_cmd_indx = [
            i for i, cmd in enumerate(runner.command_history()) if "docker run" in cmd
        ][0]
        mkdir_cmd_indx = [
            i for i, cmd in enumerate(runner.command_history()) if "mkdir -p" in cmd
        ][0]
        assert mkdir_cmd_indx < docker_run_cmd_indx
        runner.clear_history()
        cluster_scaler.update()
        runner.assert_not_has_call(worker_ip, "setup_cmd")

        # We did not start any other nodes
        next_ip = "172.0.0.{}".format(worker_id + 1)
        runner.assert_not_has_call(next_ip, " ")

    def testAutodetectResources(self):
        self.provider = MockProvider()
        config = copy.deepcopy(SMALL_CLUSTER)
        config_path = self.write_config(config)
        runner = MockProcessRunner()
        proc_meminfo = """
MemTotal:       16396056 kB
MemFree:        12869528 kB
MemAvailable:   33000000 kB
        """
        runner.respond_to_call("cat /proc/meminfo", 2 * [proc_meminfo])
        runner.respond_to_call(".Runtimes", 2 * ["nvidia-container-runtime"])
        runner.respond_to_call("nvidia-smi", 2 * ["works"])
        runner.respond_to_call("json .Config.Env", 2 * ["[]"])
        cluster_metrics = ClusterMetrics()
        event_summarizer = EventSummarizer()
        control_state = ControlState()
        cluster_scaler = MockClusterScaler(
            config_path,
            cluster_metrics,
            ClusterMetricsUpdater(cluster_metrics, event_summarizer, control_state),
            ResourceScalingPolicy("1.2.3.4", ScalingStateClient.create_from(ControlState())),
            max_failures=0,
            process_runner=runner,
            update_interval_s=0,
        )
        nodes_id = cluster_scaler.provider.non_terminated_nodes({})
        for node_id in nodes_id:
            cluster_scaler.provider.terminate_node(node_id)
        cluster_scaler.update()
        self.provider = cluster_scaler.provider
        self.waitForNodes(2, tag_filters={CLOUDTIK_TAG_USER_NODE_TYPE: "worker.default"})
        self.provider.finish_starting_nodes()
        cluster_scaler.update()
        self.waitForNodes(2, tag_filters={CLOUDTIK_TAG_NODE_STATUS: STATUS_UP_TO_DATE})
        cluster_scaler.update()
        if nodes_id:
            worker_id = nodes_id[-1] + 2
        else:
            worker_id = 1
        worker_ip = "172.0.0.{}".format(worker_id)
        runner.assert_has_call(worker_ip, pattern="--runtime=nvidia")

    def testDockerImageExistsBeforeInspect(self):
        config = copy.deepcopy(SMALL_CLUSTER)
        config["min_workers"] = 1
        config["max_workers"] = 1
        config["docker"]["pull_before_run"] = False
        config_path = self.write_config(config)
        self.provider = MockProvider()
        runner = MockProcessRunner()
        runner.respond_to_call("json .Config.Env", ["[]" for i in range(1)])
        cluster_metrics = ClusterMetrics()
        event_summarizer = EventSummarizer()
        control_state = ControlState()
        cluster_scaler = MockClusterScaler(
            config_path,
            cluster_metrics,
            ClusterMetricsUpdater(cluster_metrics, event_summarizer, control_state),
            ResourceScalingPolicy("1.2.3.4", ScalingStateClient.create_from(ControlState())),
            max_failures=0,
            process_runner=runner,
            update_interval_s=0,
        )
        nodes_id = cluster_scaler.provider.non_terminated_nodes({})
        for node_id in nodes_id:
            cluster_scaler.provider.terminate_node(node_id)
        cluster_scaler.update()
        cluster_scaler.update()
        self.provider = cluster_scaler.provider
        self.waitForNodes(2)
        self.provider.finish_starting_nodes()
        cluster_scaler.update()
        self.waitForNodes(1, tag_filters={CLOUDTIK_TAG_NODE_STATUS: STATUS_UP_TO_DATE})
        first_pull = [
            (i, cmd)
            for i, cmd in enumerate(runner.command_history())
            if "docker pull" in cmd
        ]
        first_targeted_inspect = [
            (i, cmd)
            for i, cmd in enumerate(runner.command_history())
            if "docker inspect -f" in cmd
        ]

        # This checks for the bug mentioned #13128 where the image is inspected
        # before the image is present.
        assert min(x[0] for x in first_pull) < min(x[0] for x in first_targeted_inspect)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
