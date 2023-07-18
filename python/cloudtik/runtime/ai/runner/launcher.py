import copy
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
        self.resolve()
        self.setup()
        result = self.run()
        self.finalize()
        return result

    def resolve(self):
        # By default to run single proc per node if not specified
        nproc_per_node = self.get_nproc_per_node()
        self.distributor.resolve(nproc_per_node=nproc_per_node)

    def setup(self):
        pass

    def run(self):
        pass

    def finalize(self):
        pass

    def get_nproc_per_node(self):
        return self.args.nproc_per_node if self.args.nproc_per_node else 1

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

    def wrap_func(self):
        args = self.args
        func = args.func
        func_args = args.func_args
        if func_args is None:
            func_args = ()
        func_kwargs = args.func_kwargs
        if func_kwargs is None:
            func_kwargs = {}

        def wrapped_func():
            return func(*func_args, **func_kwargs)

        return wrapped_func

    def _get_env(self, args):
        env = None
        if hasattr(args, "env") and args.env:
            # make a copy
            env = copy.copy(args.env)
        if self.environ_set:
            if env is None:
                env = copy.copy(self.environ_set)
            else:
                # update
                env.update(self.environ_set)
        return env

    def _init_launcher_args(self, launcher_args, excludes=None):
        args = self.args
        attrs = vars(launcher_args)
        for attr_name in attrs.keys():
            if hasattr(args, attr_name) and (not excludes or attr_name in excludes):
                attr_value = getattr(args, attr_name)
                setattr(args, attr_name, attr_value)

        # set extra arguments passing from run API
        self._set_args(launcher_args, args.launcher_kwargs)

    @staticmethod
    def _set_args(args, kwargs, excludes=None):
        if not kwargs:
            return
        for key, value in kwargs.items():
            if hasattr(args, key) and (not excludes or key in excludes):
                setattr(args, key, value)

    @staticmethod
    def _get_kwargs(args, attrs):
        kwargs = {}
        for attr in attrs:
            attr_value = getattr(args, attr)
            if attr_value is not None:
                kwargs[attr] = attr_value
        return kwargs
