import click
import os
import logging
from cloudtik.core._private import constants

from cloudtik.core._private import logging_utils

from cloudtik.core._private.cli_logger import (cli_logger)

from shlex import quote

CLOUDTIK_RUNTIME_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

CLOUDTIK_RUNTIME_SCRIPTS_PATH = os.path.join(
    CLOUDTIK_RUNTIME_PATH, "spark/scripts/")

SPARK_SERVICES_SCRIPT_PATH = os.path.join(CLOUDTIK_RUNTIME_SCRIPTS_PATH, "services.sh")

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
def install():
    install_script_path = os.path.join(CLOUDTIK_RUNTIME_SCRIPTS_PATH, "install.sh")
    os.system("bash {}".format(install_script_path))


@click.command()
@click.option(
    '--provider',
    required=True,
    type=str,
    help="the provider of cluster ")
@click.option(
    '--master',
    required=False,
    type=str,
    default="",
    help="the master name or ip ")
@click.option(
    '--aws_s3a_bucket',
    required=False,
    type=str,
    default="",
    help="the bucket name of s3a")
@click.option(
    '--s3a_access_key',
    required=False,
    type=str,
    default="",
    help="the access key of s3a")
@click.option(
    '--s3a_secret_key',
    required=False,
    type=str,
    default="",
    help="the secret key of s3a")
@click.option(
    '--project_id',
    required=False,
    type=str,
    default="",
    help="gcp project id")
@click.option(
    '--gcp_gcs_bucket',
    required=False,
    type=str,
    default="",
    help="gcp cloud storage bucket name")
@click.option(
    '--fs_gs_auth_service_account_email',
    required=False,
    type=str,
    default="",
    help="google service account email")
@click.option(
    '--fs_gs_auth_service_account_private_key_id',
    required=False,
    type=str,
    default="",
    help="google service account private key id")
@click.option(
    '--fs_gs_auth_service_account_private_key',
    required=False,
    type=str,
    default="",
    help="google service account private key")
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
    '--fs_azure_account_key_blob_core_windows_net',
    required=False,
    type=str,
    default="",
    help="azure blob storage access key")
def configure(provider, master, aws_s3a_bucket, s3a_access_key, s3a_secret_key, project_id, gcp_gcs_bucket,
              fs_gs_auth_service_account_email, fs_gs_auth_service_account_private_key_id,
              fs_gs_auth_service_account_private_key, azure_storage_account, azure_container,
              fs_azure_account_key_blob_core_windows_net):
    shell_path = os.path.join(CLOUDTIK_RUNTIME_SCRIPTS_PATH, "configure.sh")
    os.system("bash {} -p {} --head_address={} --aws_s3a_bucket={} --s3a_access_key={} --s3a_secret_key={} "
              "--project_id={} --gcp_gcs_bucket={} --fs_gs_auth_service_account_email={} "
              "--fs_gs_auth_service_account_private_key_id={} --fs_gs_auth_service_account_private_key={} "
              "--azure_storage_account={} --azure_container={} --fs_azure_account_key_blob_core_windows_net={}".format(
                shell_path, provider, master, aws_s3a_bucket, s3a_access_key, s3a_secret_key, project_id, gcp_gcs_bucket,
                fs_gs_auth_service_account_email, fs_gs_auth_service_account_private_key_id,
                quote(fs_gs_auth_service_account_private_key), azure_storage_account, azure_container,
                fs_azure_account_key_blob_core_windows_net))


@click.command()
def start_head():
    os.system("bash {} start-head".format(SPARK_SERVICES_SCRIPT_PATH))


@click.command()
def start_worker():
    os.system("bash {} start-worker".format(SPARK_SERVICES_SCRIPT_PATH))


@click.command()
def stop_head():
    os.system("bash {} stop-head".format(SPARK_SERVICES_SCRIPT_PATH))


@click.command()
def stop_worker():
    os.system("bash {} stop-worker".format(SPARK_SERVICES_SCRIPT_PATH))


@click.command()
def start_jupyter():
    os.system("nohup jupyter lab  --no-browser --ip=* --allow-root >/home/cloudtik/jupyter/jupyter.log 2>&1 &")
    print("\tSuccessfully started JupyterLab on master ...... "
          "\n\tPlease open `/home/jupyter/jupyter.log` to search the token,"
          " add it to the Link: http://external_ip:8888/lab?<token> , then copy this link to a browser to use JupyterLab. "
          + "\n\tKernels like Spark are ready to be used, you can choose kernels like "
          + "Python 3 (for PySpark), Spark-Scala or spylon-kernel (for Scala Spark) to run Spark.")


cli.add_command(install)
cli.add_command(configure)
cli.add_command(start_head)
cli.add_command(start_worker)
cli.add_command(stop_head)
cli.add_command(stop_worker)
cli.add_command(start_jupyter)


def main():
    return cli()


if __name__ == "__main__":
    main()
