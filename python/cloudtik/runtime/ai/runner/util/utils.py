import os
import sys
from shlex import quote


from cloudtik.runtime.ai.runner.util import network
from cloudtik.runtime.ai.runner.util.http.http_client import read_data_from_kvstore, put_data_into_kvstore
from cloudtik.runtime.ai.runner.util.http.http_server import KVStoreServer

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


def _run_func(
        func, num_proc, run_command,
        executable, verbose):
    # get the driver IPv4 address
    driver_ip = network.get_default_ip_address()
    run_func_server = KVStoreServer(verbose=verbose)
    run_func_server_port = run_func_server.start_server()
    put_data_into_kvstore(driver_ip, run_func_server_port,
                          'runfunc', 'func', func)

    executable = executable or sys.executable
    command = [executable, '-m', 'cloudtik.runtime.ai.runner.util.run_func',
               str(driver_ip), str(run_func_server_port)]

    try:
        run_command(command)
        results = [None] * num_proc
        # TODO: make it parallel to improve performance
        for i in range(num_proc):
            results[i] = read_data_from_kvstore(
                driver_ip, run_func_server_port,
                'runfunc_result', str(i))
        return results
    finally:
        run_func_server.shutdown_server()
