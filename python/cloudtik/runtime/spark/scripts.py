import click
import os
import logging

from cloudtik.core._private import constants
from cloudtik.core._private import logging_utils
from cloudtik.core._private.cli_logger import (cli_logger)

from shlex import quote

from cloudtik.runtime.spark.utils import RUNTIME_ROOT_PATH, update_spark_configurations

RUNTIME_SCRIPTS_PATH = os.path.join(
    RUNTIME_ROOT_PATH, "scripts")

SERVICES_SCRIPT_PATH = os.path.join(RUNTIME_SCRIPTS_PATH, "services.sh")

logger = logging.getLogger(__name__)


def run_system_command(cmd: str):
    result = os.system(cmd)
    if result != 0:
        raise RuntimeError(f"Error happened in running: {cmd}")


def run_services_command(command: str, script_args):
    cmds = [
        "bash",
        SERVICES_SCRIPT_PATH,
    ]

    cmds += [command]
    if script_args:
        cmds += list(script_args)
    final_cmd = " ".join(cmds)

    run_system_command(final_cmd)


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


@click.command(context_settings={"ignore_unknown_options": True})
@click.option(
    "--head",
    is_flag=True,
    default=False,
    help="provide this argument for the head node")
@click.option(
    '--provider',
    required=True,
    type=str,
    help="the provider of cluster ")
@click.argument("script_args", nargs=-1)
def install(head, provider, script_args):
    install_script_path = os.path.join(RUNTIME_SCRIPTS_PATH, "install.sh")
    cmds = [
        "bash",
        install_script_path,
    ]

    if head:
        cmds += ["--head"]
    if provider:
        cmds += ["--provider={}".format(provider)]
    if script_args:
        cmds += list(script_args)
    final_cmd = " ".join(cmds)

    run_system_command(final_cmd)


@click.command(context_settings={"ignore_unknown_options": True})
@click.option(
    "--head",
    is_flag=True,
    default=False,
    help="provide this argument for the head node")
@click.option(
    '--provider',
    required=True,
    type=str,
    help="the provider of cluster ")
@click.option(
    '--head_address',
    required=False,
    type=str,
    default="",
    help="the head ip ")
@click.option(
    '--aws_s3_bucket',
    required=False,
    type=str,
    default="",
    help="the bucket name of s3")
@click.option(
    '--aws_s3_access_key_id',
    required=False,
    type=str,
    default="",
    help="the access key id of s3")
@click.option(
    '--aws_s3_secret_access_key',
    required=False,
    type=str,
    default="",
    help="the secret access key of s3")
@click.option(
    '--project_id',
    required=False,
    type=str,
    default="",
    help="gcp project id")
@click.option(
    '--gcs_bucket',
    required=False,
    type=str,
    default="",
    help="gcp cloud storage bucket name")
@click.option(
    '--gcs_service_account_client_email',
    required=False,
    type=str,
    default="",
    help="google service account email")
@click.option(
    '--gcs_service_account_private_key_id',
    required=False,
    type=str,
    default="",
    help="google service account private key id")
@click.option(
    '--gcs_service_account_private_key',
    required=False,
    type=str,
    default="",
    help="google service account private key")
@click.option(
    '--azure_storage_type',
    required=False,
    type=str,
    default="",
    help="azure storage kind, whether azure blob storage or azure data lake gen2")
@click.option(
    '--azure_storage_account',
    required=False,
    type=str,
    default="",
    help="azure storage account")
@click.option(
    '--azure_container',
    required=False,
    type=str,
    default="",
    help="azure storage container")
@click.option(
    '--azure_account_key',
    required=False,
    type=str,
    default="",
    help="azure storage account access key")
@click.argument("script_args", nargs=-1)
def configure(head, provider, head_address, aws_s3_bucket, aws_s3_access_key_id, aws_s3_secret_access_key, project_id, gcs_bucket,
              gcs_service_account_client_email, gcs_service_account_private_key_id,
              gcs_service_account_private_key, azure_storage_type, azure_storage_account, azure_container,
              azure_account_key, script_args):
    shell_path = os.path.join(RUNTIME_SCRIPTS_PATH, "configure.sh")
    cmds = [
        "bash",
        shell_path,
    ]

    if head:
        cmds += ["--head"]
    if provider:
        cmds += ["--provider={}".format(provider)]
    if head_address:
        cmds += ["--head_address={}".format(head_address)]

    if aws_s3_bucket:
        cmds += ["--aws_s3_bucket={}".format(aws_s3_bucket)]
    if aws_s3_access_key_id:
        cmds += ["--aws_s3_access_key_id={}".format(aws_s3_access_key_id)]
    if aws_s3_secret_access_key:
        cmds += ["--aws_s3_secret_access_key={}".format(aws_s3_secret_access_key)]

    if project_id:
        cmds += ["--project_id={}".format(project_id)]
    if gcs_bucket:
        cmds += ["--gcs_bucket={}".format(gcs_bucket)]

    if gcs_service_account_client_email:
        cmds += ["--gcs_service_account_client_email={}".format(gcs_service_account_client_email)]
    if gcs_service_account_private_key_id:
        cmds += ["--gcs_service_account_private_key_id={}".format(gcs_service_account_private_key_id)]
    if gcs_service_account_private_key:
        cmds += ["--gcs_service_account_private_key={}".format(quote(gcs_service_account_private_key))]

    if azure_storage_type:
        cmds += ["--azure_storage_type={}".format(azure_storage_type)]
    if azure_storage_account:
        cmds += ["--azure_storage_account={}".format(azure_storage_account)]
    if azure_container:
        cmds += ["--azure_container={}".format(azure_container)]
    if azure_account_key:
        cmds += ["--azure_account_key={}".format(azure_account_key)]

    if script_args:
        cmds += list(script_args)

    final_cmd = " ".join(cmds)
    run_system_command(final_cmd)

    # Update spark configuration from cluster config file
    update_spark_configurations()


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
