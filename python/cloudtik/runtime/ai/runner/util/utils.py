import os
from shlex import quote


CLOUDTIK_COMMAND_PREFIX = 'cloudtik head exec'


def get_cloudtik_exec(local_command, host):
    final_command = quote(local_command)
    return f'{CLOUDTIK_COMMAND_PREFIX} {final_command} --node-ip={host}'


def get_cloudtik_rsh():
    runtime_home = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(runtime_home, "scripts", "cloudtik-rsh.sh")


def _cache(f):
    cache = dict()

    def wrapper(*args, **kwargs):
        key = (args, frozenset(kwargs.items()))

        if key in cache:
            return cache[key]
        else:
            retval = f(*args, **kwargs)
            cache[key] = retval
            return retval

    return wrapper


def is_python_program(command):
    if not command:
        return False
    return command[0].endswith(".py")
