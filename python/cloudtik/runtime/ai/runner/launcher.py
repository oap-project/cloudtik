import glob
import logging
import os
from os.path import expanduser

logger = logging.getLogger(__name__)


class Launcher:
    r"""
     Base class for launcher
    """
    def __init__(self, args, distributor):
        self.args = args
        self.distributor = distributor

    def launch(self):
        pass

    def add_lib_preload(self, lib_type=None):
        library_paths = []
        if "CONDA_PREFIX" in os.environ:
            library_paths.append(os.environ["CONDA_PREFIX"] + "/lib/")
        if "VIRTUAL_ENV" in os.environ:
            library_paths.append(os.environ["VIRTUAL_ENV"] + "/lib/")

        library_paths += ["{}/.local/lib/".format(expanduser("~")), "/usr/local/lib/",
                          "/usr/local/lib64/", "/usr/lib/", "/usr/lib64/"]

        lib_find = False
        lib_set = False
        for item in os.getenv("LD_PRELOAD", "").split(":"):
            if item.endswith('lib{}.so'.format(lib_type)):
                lib_set = True
                break
        if not lib_set:
            for lib_path in library_paths:
                library_file = lib_path + "lib" + lib_type + ".so"
                matches = glob.glob(library_file)
                if len(matches) > 0:
                    if "LD_PRELOAD" in os.environ:
                        os.environ["LD_PRELOAD"] = matches[0] + ":" + os.environ["LD_PRELOAD"]
                    else:
                        os.environ["LD_PRELOAD"] = matches[0]
                    lib_find = True
                    break
        return lib_set or lib_find

    def log_env(self, env_name=""):
        if env_name in os.environ:
            logger.info("{}={}".format(env_name, os.environ[env_name]))

    def set_env(self, env_name, env_value=None):
        if not env_value:
            logger.warning("{} is None".format(env_name))
        if env_name not in os.environ:
            os.environ[env_name] = env_value
        elif os.environ[env_name] != env_value:
            logger.warning("{} in environment variable is {} while the value you set is {}".format(env_name, os.environ[env_name], env_value))
        self.log_env(env_name)
