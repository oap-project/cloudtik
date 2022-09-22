from typing import Any, Dict
import subprocess
from types import ModuleType

from cloudtik.core._private.call_context import CallContext
from cloudtik.core._private.node.node_updater import NodeUpdaterThread
from cloudtik.core._private.providers import _get_node_provider
from cloudtik.core._private.utils import get_running_head_node, _get_node_specific_runtime_config, \
    _get_node_specific_docker_config


def create_node_updater_for_exec(config,
                                 call_context: CallContext,
                                 node_id,
                                 provider,
                                 start_commands,
                                 is_head_node: bool = False,
                                 use_internal_ip: bool = False,
                                 runtime_config: Dict[str, Any] = None,
                                 process_runner: ModuleType = subprocess):
    if runtime_config is None:
        runtime_config = _get_node_specific_runtime_config(
            config, provider, node_id)
    docker_config = _get_node_specific_docker_config(
            config, provider, node_id)
    updater = NodeUpdaterThread(
        config=config,
        call_context=call_context,
        node_id=node_id,
        provider_config=config["provider"],
        provider=provider,
        auth_config=config["auth"],
        cluster_name=config["cluster_name"],
        file_mounts=config["file_mounts"],
        initialization_commands=[],
        setup_commands=[],
        start_commands=start_commands,
        runtime_hash="",
        file_mounts_contents_hash="",
        is_head_node=is_head_node,
        process_runner=process_runner,
        use_internal_ip=use_internal_ip,
        rsync_options={
            "rsync_exclude": config.get("rsync_exclude"),
            "rsync_filter": config.get("rsync_filter")
        },
        docker_config=docker_config,
        runtime_config=runtime_config)
    return updater


def run_on_cluster(
        config: Dict[str, Any],
        call_context: CallContext,
        cmd: str = None,
        run_env: str = "auto",
        with_output: bool = False,
        _allow_uninitialized_state: bool = False) -> str:
    # We default this to True to maintain backwards-compatibility
    # In the future we would want to support disabling login-shells
    # and interactivity.
    call_context.set_allow_interactive(True)

    use_internal_ip = config.get("bootstrapped", False)

    provider = _get_node_provider(config["provider"], config["cluster_name"])
    head_node = get_running_head_node(
        config,
        _allow_uninitialized_state=_allow_uninitialized_state)
    updater = create_node_updater_for_exec(
        config=config,
        call_context=call_context,
        node_id=head_node,
        provider=provider,
        start_commands=[],
        is_head_node=True,
        use_internal_ip=use_internal_ip)

    exec_out = updater.cmd_executor.run(
        cmd,
        with_output=with_output,
        run_env=run_env)
    if with_output:
        return exec_out.decode(encoding="utf-8")
    else:
        return exec_out
