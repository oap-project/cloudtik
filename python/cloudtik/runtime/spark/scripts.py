import logging

import click

from cloudtik.core._private import constants
from cloudtik.core._private import logging_utils
from cloudtik.core._private.cli_logger import (cli_logger, add_click_logging_options)
from cloudtik.core._private.cluster.cluster_config import _load_cluster_config
from cloudtik.runtime.spark.utils import print_request_rest_applications, \
    print_request_rest_yarn, get_runtime_default_storage

logger = logging.getLogger(__name__)


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


@click.command()
@click.argument("cluster_config_file", required=True, type=str)
@click.option(
    "--cluster-name",
    "-n",
    required=False,
    type=str,
    help="Override the configured cluster name.")
@click.option(
    "--endpoint",
    required=False,
    type=str,
    help="The resource endpoint for the history server rest API")
@add_click_logging_options
def applications(cluster_config_file, cluster_name, endpoint):
    print_request_rest_applications(cluster_config_file, cluster_name, endpoint)


@click.command()
@click.argument("cluster_config_file", required=True, type=str)
@click.option(
    "--cluster-name",
    "-n",
    required=False,
    type=str,
    help="Override the configured cluster name.")
@click.option(
    "--endpoint",
    required=False,
    type=str,
    help="The resource endpoint for the YARN rest API")
@add_click_logging_options
def yarn(cluster_config_file, cluster_name, endpoint):
    print_request_rest_yarn(cluster_config_file, cluster_name, endpoint)


@click.command()
@click.argument("cluster_config_file", required=True, type=str)
@click.option(
    "--cluster-name",
    "-n",
    required=False,
    type=str,
    help="Override the configured cluster name.")
@click.option(
    "--default-storage",
    is_flag=True,
    default=False,
    help="Show the default storage of the cluster.")
@add_click_logging_options
def info(cluster_config_file, cluster_name, default_storage):
    config = _load_cluster_config(cluster_config_file, cluster_name)
    if default_storage:
        # show default storage
        default_storage_uri = get_runtime_default_storage(config)
        if default_storage_uri:
            click.echo(default_storage_uri)


cli.add_command(applications)
cli.add_command(yarn)
cli.add_command(info)


def main():
    return cli()


if __name__ == "__main__":
    main()
