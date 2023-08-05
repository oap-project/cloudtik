import os
import sys
from shlex import quote

from cloudtik.core._private import constants
from cloudtik.core._private.cli_logger import cli_logger
from cloudtik.core._private.core_utils import get_cloudtik_temp_dir, check_process_exists, stop_process_tree, \
    get_named_log_file_handles
from cloudtik.core._private.node.node_services import SESSION_LATEST
from cloudtik.core._private.services import start_cloudtik_process
from cloudtik.core._private.utils import save_server_process, get_server_process

UTIL_PATH = os.path.abspath(os.path.dirname(__file__))
PROCESS_PULL_SERVER = "cloudtik_pull_server"


def pull_server(
        identifier, command,
        pull_class, pull_script,
        pull_args, interval,
        logs_dir=None, redirect_output=True):
    if not identifier:
        raise ValueError(
            "Pulling identifier cannot be empty.")
    if command == "start":
        if not pull_class and not pull_script:
            raise ValueError(
                "You need either specify a pull class or a pull script.")
        start_pull_server(
            identifier, pull_class, pull_script, pull_args,
            interval=interval, logs_dir=logs_dir,
            redirect_output=redirect_output)
    elif command == "stop":
        _stop_pull_server(identifier)
    else:
        raise ValueError(
            "Invalid command parameter: {}".format(command))


def get_pull_server_process_file(identifier: str):
    return os.path.join(
        get_cloudtik_temp_dir(), "cloudtik-pull-{}".format(identifier))


def get_pull_server_pid(proxy_process_file: str):
    server_process = get_server_process(proxy_process_file)
    if server_process is None:
        return None
    pid = server_process.get("pid")
    if pid is None:
        return None
    if not check_process_exists(pid):
        return None
    return pid


def _start_pull_server(
        identifier,
        pull_class,
        pull_script,
        pull_args,
        interval,
        logs_dir,
        stdout_file=None,
        stderr_file=None,
        logging_level=None,
        max_bytes=0,
        backup_count=0):
    """Run a process to controller the other processes.

    Args:
        identifier (str): The identifier of the pull server.
        pull_class (str): The puller module and class.
        pull_script (str): The puller script file.
        pull_args(List[str]): The list of arguments pass to the puller
        interval (int): The interval in seconds for each pull.
        logs_dir(str): The path to the log directory.
        stdout_file: A file handle opened for writing to redirect stdout to. If
            no redirection should happen, then this should be None.
        stderr_file: A file handle opened for writing to redirect stderr to. If
            no redirection should happen, then this should be None.
        logging_level (str): The logging level to use for the process.
        max_bytes (int): Log rotation parameter. Corresponding to
            RotatingFileHandler's maxBytes.
        backup_count (int): Log rotation parameter. Corresponding to
            RotatingFileHandler's backupCount.
    Returns:
        ProcessInfo for the process that was started.
    """
    pull_server_path = os.path.join(UTIL_PATH, "pull", "cloudtik_pull_server.py")
    command = [
        sys.executable,
        "-u",
        pull_server_path,
        f"--logs-dir={logs_dir}",
        f"--logging-rotate-bytes={max_bytes}",
        f"--logging-rotate-backup-count={backup_count}",
    ]
    if logging_level:
        command.append("--logging-level=" + logging_level)

    command.append("--identifier=" + identifier)
    if pull_class:
        command.append("--pull-class=" + quote(pull_class))
    if pull_script:
        command.append("--pull-script=" + quote(pull_script))
    if interval:
        command.append("--interval=" + str(interval))
    if pull_args:
        command.append(pull_args)

    process_info = start_cloudtik_process(
        command,
        PROCESS_PULL_SERVER,
        stdout_file=stdout_file,
        stderr_file=stderr_file,
        fate_share=False)
    return process_info


def start_pull_server(
        identifier,
        pull_class, pull_script,
        pull_args, interval,
        logs_dir=None, redirect_output=True):
    if not logs_dir:
        temp_dir = get_cloudtik_temp_dir()
        logs_dir = os.path.join(temp_dir, SESSION_LATEST, "logs")

    stdout_file, stderr_file = get_named_log_file_handles(
        logs_dir, "cloudtik_pull_server",
        redirect_output=redirect_output)

    # Configure log parameters.
    logging_level = os.getenv(constants.CLOUDTIK_LOGGING_LEVEL_ENV,
                              constants.LOGGER_LEVEL_INFO)
    max_bytes = int(
        os.getenv(constants.CLOUDTIK_LOGGING_ROTATE_MAX_BYTES_ENV,
                  constants.LOGGING_ROTATE_MAX_BYTES))
    backup_count = int(
        os.getenv(constants.CLOUDTIK_LOGGING_ROTATE_BACKUP_COUNT_ENV,
                  constants.LOGGING_ROTATE_BACKUP_COUNT))

    process_info = _start_pull_server(
        identifier,
        pull_class,
        pull_script,
        pull_args,
        interval,
        logs_dir,
        stdout_file=stdout_file,
        stderr_file=stderr_file,
        logging_level=logging_level,
        max_bytes=max_bytes,
        backup_count=backup_count)

    pid = process_info.process.pid
    pull_server_process_file = get_pull_server_process_file(identifier)
    pull_server_process = {"pid": pid}
    save_server_process(pull_server_process_file, pull_server_process)

    cli_logger.print(
        "Successfully started pull server: {}".format(identifier))


def _stop_pull_server(identifier):
    # find the pid file and stop it
    pull_server_process_file = get_pull_server_process_file(identifier)
    pid = get_pull_server_pid(pull_server_process_file)
    if pid is None:
        cli_logger.print("The pull server for {} was not started.", identifier)
        return

    stop_process_tree(pid)
    save_server_process(pull_server_process_file, {})
    cli_logger.print("Successfully stopped pull server of {}.", identifier)
