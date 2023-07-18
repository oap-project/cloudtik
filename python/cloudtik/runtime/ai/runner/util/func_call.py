import sys

from cloudtik.runtime.ai.runner.util import network
from cloudtik.runtime.ai.runner.util.http.http_client import read_data_from_kvstore, put_data_into_kvstore
from cloudtik.runtime.ai.runner.util.http.http_server import KVStoreServer


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
