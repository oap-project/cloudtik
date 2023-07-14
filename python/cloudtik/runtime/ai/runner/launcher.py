import logging
import os
import sys

from cloudtik.runtime.ai.runner.util.utils import is_python_program

logger = logging.getLogger(__name__)


class Launcher:
    r"""
     Base class for launcher
    """
    def __init__(self, args, distributor):
        self.args = args
        self.distributor = distributor
        self.environ_set = {}

    def launch(self):
        pass

    def set_env(self, env_name, env_value):
        value = os.getenv(env_name, "")
        if value != "" and value != env_value:
            logger.warning("{} in environment variable is {} while the value you set is {}".format(
                env_name, os.environ[env_name], env_value))
            self.environ_set[env_name] = os.environ[env_name]
        else:
            self.environ_set[env_name] = env_value

    def with_python_command(self, cmd):
        args = self.args
        with_python = not args.no_python
        if with_python and (
                args.module or is_python_program(args.command)):
            # check whether the program in the command is end with py
            cmd.append(sys.executable)
            cmd.append("-u")
            if args.module:
                cmd.append("-m")
