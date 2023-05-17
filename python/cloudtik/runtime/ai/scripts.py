from shlex import quote

import click
import os
import logging

from cloudtik.core._private import constants
from cloudtik.core._private import logging_utils
from cloudtik.core._private.cli_logger import (cli_logger)
from cloudtik.core._private.utils import run_bash_scripts, run_system_command, with_script_args

from cloudtik.runtime.ai.utils import RUNTIME_ROOT_PATH

RUNTIME_SCRIPTS_PATH = os.path.join(
    RUNTIME_ROOT_PATH, "scripts")

SERVICES_SCRIPT_PATH = os.path.join(RUNTIME_SCRIPTS_PATH, "services.sh")

logger = logging.getLogger(__name__)


def run_services_command(command: str, script_args):
    run_bash_scripts(command, SERVICES_SCRIPT_PATH, script_args)


@click.group()
@click.option(
    "--logging-level",
    required=False,
    default=constants.LOGGER_LEVEL_INFO,
    type=str,
    help=constants.LOGGER_LEVEL_HELP)
@click.option(
    "--logging-format",
    required=False,
    default=constants.LOGGER_FORMAT,
    type=str,
    help=constants.LOGGER_FORMAT_HELP)
@click.version_option()
def cli(logging_level, logging_format):
    level = logging.getLevelName(logging_level.upper())
    logging_utils.setup_logger(level, logging_format)
    cli_logger.set_format(format_tmpl=logging_format)


@click.command(context_settings={"ignore_unknown_options": True})
@click.option(
    "--head",
    is_flag=True,
    default=False,
    help="provide this argument for the head node")
@click.argument("script_args", nargs=-1)
def install(head, script_args):
    install_script_path = os.path.join(RUNTIME_SCRIPTS_PATH, "install.sh")
    cmds = [
        "bash",
        quote(install_script_path),
    ]

    if head:
        cmds += ["--head"]
    with_script_args(cmds, script_args)
    final_cmd = " ".join(cmds)

    run_system_command(final_cmd)


@click.command(context_settings={"ignore_unknown_options": True})
@click.option(
    "--head",
    is_flag=True,
    default=False,
    help="provide this argument for the head node")
@click.option(
    '--head_address',
    required=False,
    type=str,
    default="",
    help="the head ip ")
@click.argument("script_args", nargs=-1)
def configure(head, head_address, script_args):
    shell_path = os.path.join(RUNTIME_SCRIPTS_PATH, "configure.sh")
    cmds = [
        "bash",
        quote(shell_path),
    ]

    if head:
        cmds += ["--head"]
    if head_address:
        cmds += ["--head_address={}".format(head_address)]

    with_script_args(cmds, script_args)

    final_cmd = " ".join(cmds)
    run_system_command(final_cmd)


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument("command", required=True, type=str)
@click.argument("script_args", nargs=-1)
def services(command, script_args):
    run_services_command(command, script_args)


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument("script_args", nargs=-1)
def start_head(script_args):
    run_services_command("start-head", script_args)


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument("script_args", nargs=-1)
def start_worker(script_args):
    run_services_command("start-worker", script_args)


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument("script_args", nargs=-1)
def stop_head(script_args):
    run_services_command("stop-head", script_args)


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument("script_args", nargs=-1)
def stop_worker(script_args):
    run_services_command("stop-worker", script_args)


cli.add_command(install)
cli.add_command(configure)
cli.add_command(services)
cli.add_command(start_head)
cli.add_command(start_worker)
cli.add_command(stop_head)
cli.add_command(stop_worker)


def main():
    return cli()


if __name__ == "__main__":
    main()
