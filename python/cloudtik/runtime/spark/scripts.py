import click
import os
import logging
from cloudtik.core._private import constants

from cloudtik.core._private import logging_utils

from cloudtik.core._private.cli_logger import (cli_logger)


CLOUDTIK_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

CLOUDTIK_RUNTIME_SCRIPTS_PATH = os.path.join(
    CLOUDTIK_PATH, "runtime/spark/scripts/")

HADOOP_DAEMON_PATH = os.path.join(CLOUDTIK_RUNTIME_SCRIPTS_PATH, "hadoop-daemon.sh")

logger = logging.getLogger(__name__)


@click.group()
@click.option(
    "--logging-level",
    required=False,
    default=constants.LOGGER_LEVEL,
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


@click.command()
@click.option(
    '--master',
    required=False,
    type=str,
    default="",
    help="the master name or ip ")
def update_config(master):
    shell_path = os.path.join(CLOUDTIK_RUNTIME_SCRIPTS_PATH, "update-config.sh")
    os.system("bash {} {}".format(shell_path,master))


@click.command()
def start_head():
    os.system("bash {} start-head".format(HADOOP_DAEMON_PATH))

@click.command()
def start_worker():
    os.system("bash {} start-worker".format(HADOOP_DAEMON_PATH))

@click.command()
def stop_head():
    os.system("bash {} stop-head".format(HADOOP_DAEMON_PATH))

@click.command()
def stop_worker():
    os.system("bash {} stop-worker".format(HADOOP_DAEMON_PATH))

cli.add_command(update_config)
cli.add_command(start_head)
cli.add_command(start_worker)
cli.add_command(stop_head)
cli.add_command(stop_worker)


def main():
    return cli()


if __name__ == "__main__":
    main()
