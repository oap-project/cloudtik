import json
import logging
from shlex import quote
from typing import Dict

from cloudtik.core._private.constants import \
    PRIVACY_REPLACEMENT, PRIVACY_REPLACEMENT_TEMPLATE

logger = logging.getLogger(__name__)

PRIVACY_KEYWORDS = ["PASSWORD", "ACCOUNT", "SECRET", "ACCESS_KEY", "PRIVATE_KEY"]


def is_key_with_privacy(key: str):
    for keyword in PRIVACY_KEYWORDS:
        if keyword in key:
            return True
    return False


def _with_environment_variables(cmd: str,
                                environment_variables: Dict[str, object],
                                cmd_to_print: str = None):
    """Prepend environment variables to a shell command.

    Args:
        cmd (str): The base command.
        environment_variables (Dict[str, object]): The set of environment
            variables. If an environment variable value is a dict, it will
            automatically be converted to a one line yaml string.
        cmd_to_print (str): The command to print for base command if there is one
    """

    as_strings = []
    as_strings_to_print = []
    with_privacy = False
    for key, val in environment_variables.items():
        # json.dumps will add an extra quote to string value
        # since we use quote to make sure value is safe for shell, we don't need the quote for string
        escaped_val = json.dumps(val, separators=(",", ":"))
        if isinstance(val, str):
            escaped_val = escaped_val.strip("\"\'")

        s = "export {}={};".format(key, quote(escaped_val))
        as_strings.append(s)

        if is_key_with_privacy(key):
            with_privacy = True
            val_len = len(escaped_val)
            replacement_len = len(PRIVACY_REPLACEMENT)
            if val_len > replacement_len:
                escaped_val = PRIVACY_REPLACEMENT_TEMPLATE.format("-" * (val_len - replacement_len))
            else:
                escaped_val = PRIVACY_REPLACEMENT
            s = "export {}={};".format(key, quote(escaped_val))

        as_strings_to_print.append(s)

    all_vars = "".join(as_strings)
    cmd_with_vars = all_vars + cmd

    cmd_with_vars_to_print = None
    if cmd_to_print or with_privacy:
        all_vars_to_print = "".join(as_strings_to_print)
        cmd_with_vars_to_print = all_vars_to_print + (cmd if cmd_to_print is None else cmd_to_print)
    return cmd_with_vars, cmd_with_vars_to_print


def _with_shutdown(cmd, cmd_to_print=None):
    cmd += "; sudo shutdown -h now"
    if cmd_to_print:
        cmd_to_print += "; sudo shutdown -h now"
    return cmd, cmd_to_print


def _with_interactive(cmd):
    force_interactive = (
        f"true && source ~/.bashrc && "
        f"export PYTHONWARNINGS=ignore && ({cmd})")
    return ["bash", "--login", "-c", "-i", quote(force_interactive)]


def _with_login_shell(cmd, interactive=True):
    force_interactive = (
        f"true && source ~/.bashrc && "
        f"export PYTHONWARNINGS=ignore && ({cmd})")
    shell_cmd = ["bash", "--login", "-c"]
    if interactive:
        shell_cmd += ["-i"]
    shell_cmd += [quote(force_interactive)]
    return shell_cmd
