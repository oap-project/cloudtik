# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import copy
import errno
import io
import math
import os
import signal
import sys
import threading
import time
from shlex import quote

from cloudtik.runtime.ai.runner import get_cloudtik_exec
from cloudtik.runtime.ai.runner.util import env as env_util, network, safe_shell_exec, threads, hosts
from cloudtik.runtime.ai.runner.util.hosts import get_host_assignments, parse_hosts

SSH_COMMAND_PREFIX = 'ssh -o PasswordAuthentication=no -o StrictHostKeyChecking=no'

# Number of attempts for sshing into the hosts
SSH_ATTEMPTS = 5

SSH_CONNECT_TIMEOUT_S = 10


def get_ssh_command(local_command, host, port=None, identity_file=None, timeout_s=None):
    final_command = quote(local_command)
    port_arg = f'-p {port}' if port is not None else ''
    identity_file_arg = f'-i {identity_file}' if identity_file is not None else ''
    timeout_arg = f'-o ConnectTimeout={timeout_s}' if timeout_s is not None else ''
    return f'{SSH_COMMAND_PREFIX} {host} {port_arg} {identity_file_arg} {timeout_arg} {final_command}'


def get_remote_command(
        local_command, host, use_ssh=False,
        port=None, identity_file=None, timeout_s=None):
    return get_ssh_command(local_command, host, port, identity_file, timeout_s) if use_ssh \
        else get_cloudtik_exec(local_command, host)


def _check_connectivity(hosts_slots_str, settings):
    all_host_names, _ = hosts.parse_hosts_and_slots(hosts_slots_str)
    if settings.verbose >= 2:
        print('Filtering local host names.')
    remote_host_names = network.filter_local_addresses(all_host_names)
    if settings.verbose >= 2:
        print('Remote host found: ' + ' '.join(remote_host_names))

    if len(remote_host_names) > 0:
        if settings.verbose >= 2:
            print('Checking ssh on all remote hosts.')
        # Check if we can ssh into all remote hosts successfully.
        if not _check_hosts_connectivity(
                remote_host_names, settings):
            raise RuntimeError('could not connect to some hosts via ssh')
        if settings.verbose >= 2:
            print('SSH was successful into all the remote hosts.')


def _check_hosts_connectivity(
        host_addresses, settings):
    """
    checks if ssh can successfully be performed to all the hosts.
    :param host_addresses: list of addresses to ssh into. for example,
        ['worker-0','worker-1']
        ['10.11.11.11', '10.11.11.12']
    :type host_addresses: list(strings)
    :return: Returns True if all ssh was successful into all the addresses.
    """

    def exec_command(command):
        exit_code = 1
        output_msg = ''

        # Try ssh 5 times
        for i in range(SSH_ATTEMPTS):
            output = io.StringIO()
            try:
                exit_code = safe_shell_exec.execute(command,
                                                    stdout=output,
                                                    stderr=output)
                if exit_code == 0:
                    break
                output_msg = output.getvalue()
            finally:
                output.close()
        return exit_code, output_msg

    args_list = [[get_remote_command(local_command='true',
                                     host=host_address,
                                     use_ssh=settings.use_ssh,
                                     port=settings.ssh_port,
                                     identity_file=settings.ssh_identity_file,
                                     timeout_s=SSH_CONNECT_TIMEOUT_S)]
                 for host_address in host_addresses]
    ssh_exit_codes = \
        threads.execute_function_multithreaded(exec_command,
                                               args_list)

    ssh_successful_to_all_hosts = True
    for index, ssh_status in ssh_exit_codes.items():
        exit_code, output_msg = ssh_status[0], ssh_status[1]
        if exit_code != 0:
            print('ssh not successful for host {host}:\n{msg_output}'
                  .format(host=host_addresses[index],
                          msg_output=output_msg))

            ssh_successful_to_all_hosts = False
    if not ssh_successful_to_all_hosts:
        return None  # we could return False here but do not want it to be cached
    return True


def _pad_rank(rank, size):
    width = int(math.log10(size - 1)) + 1
    return str(rank).zfill(width)


def _mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class MultiFile(object):
    def __init__(self, files):
        self._files = files

    def write(self, text):
        for f in self._files:
            f.write(text)

    def flush(self):
        for f in self._files:
            f.flush()


def create_slot_env_vars(slot_info):
    host_name = slot_info.hostname
    slot_env = {
        "HOSTNAME": str(host_name),
        "RANK": str(slot_info.rank),
        "WORLD_SIZE": str(slot_info.size),
        "LOCAL_RANK": str(slot_info.local_rank),
        "LOCAL_WORLD_SIZE": str(slot_info.local_size),
    }
    return slot_env


def _slot_info_to_command_fn(run_command, env):
    # TODO: Workaround for over-buffered outputs. Investigate how mpirun avoids this problem.
    env = copy.copy(env)  # copy env so we do not leak env modifications
    env['PYTHONUNBUFFERED'] = '1'

    def slot_info_to_command(slot_info):
        """
        Given a slot_info, creates a command to launch a single job.

        :param slot_info: host and slot to execute the run command on
        :return:
        """
        env_vars = create_slot_env_vars(slot_info)
        slot_env = " ".join(
            [f"{k}={str(v)}" for k, v in env_vars.items()])

        return '{shell} {slot_env} {env} {run_command}' .format(
            shell='env',
            slot_env=slot_env,
            env=' '.join(['%s=%s' % (key, quote(value)) for key, value in env.items()
                          if env_util.is_exportable(key)]),
            run_command=run_command)

    return slot_info_to_command


def _exec_command_fn(num_proc, settings):
    """
    Return a function to execute a command on remote node for specific slot.
    """

    # Non-elastic gloo runs should terminate all workers when any fail.
    terminate_all_event = threading.Event()

    def _exec_command(command, slot_info, events):
        index = slot_info.rank
        host_name = slot_info.hostname

        host_address = network.resolve_host_address(host_name)
        local_addresses = network.get_local_host_addresses()
        if host_address not in local_addresses:
            local_command = 'cd {pwd} > /dev/null 2>&1 ; {command}'.format(
                pwd=os.getcwd(), command=command)
            command = get_remote_command(local_command,
                                         host=host_name,
                                         use_ssh=settings.use_ssh,
                                         port=settings.ssh_port,
                                         identity_file=settings.ssh_identity_file)

        if settings.verbose:
            print(command)

        # Redirect output if requested
        stdout = stderr = None
        stdout_file = stderr_file = None
        if settings.output_filename:
            padded_rank = _pad_rank(index, num_proc)
            output_dir_rank = os.path.join(
                settings.output_filename, 'rank.{rank}'.format(rank=padded_rank))
            if not os.path.exists(output_dir_rank):
                os.mkdir(output_dir_rank)

            stdout_file = open(os.path.join(output_dir_rank, 'stdout'), 'w')
            stderr_file = open(os.path.join(output_dir_rank, 'stderr'), 'w')

            stdout = MultiFile([sys.stdout, stdout_file])
            stderr = MultiFile([sys.stderr, stderr_file])

        all_events = []
        if events:
            all_events += events
        if terminate_all_event:
            all_events += [terminate_all_event]

        try:
            exit_code = safe_shell_exec.execute(command,
                                                index=index,
                                                stdout=stdout,
                                                stderr=stderr,
                                                events=all_events,
                                                prefix_output_with_timestamp=settings.prefix_output_with_timestamp)
            if exit_code != 0:
                print('Process {idx} exit with status code {ec}.'.format(idx=index, ec=exit_code))
        except Exception as e:
            print('Exception happened during safe_shell_exec, exception '
                  'message: {message}'.format(message=e))
            exit_code = 1
        finally:
            if stdout_file:
                stdout_file.close()
            if stderr_file:
                stderr_file.close()
        if exit_code != 0 and terminate_all_event:
            if not any(ev.is_set() for ev in all_events):
                print('Terminating remaining workers after failure of Process {idx}.'.format(idx=index))
            terminate_all_event.set()
        return exit_code, time.time()

    return _exec_command


def create_run_env_vars(nics):
    run_envs = {}
    if nics:
        run_envs['NCCL_SOCKET_IFNAME'] = ','.join(nics)
    return run_envs


def get_run_command(command, nics):
    env_vars = create_run_env_vars(nics)
    env_string = " ".join(
        [f"{k}={str(v)}" for k, v in env_vars.items()])
    run_command = (
        '{env_string} '
        '{command}'  # expect a lot of environment variables
        .format(env_string=env_string,
                command=' '.join(quote(par) for par in command)))
    return run_command


def register_shutdown_event():
    # Create a event for communication between threads
    event = threading.Event()

    def set_event_on_signal(signum, frame):
        event.set()

    signal.signal(signal.SIGINT, set_event_on_signal)
    signal.signal(signal.SIGTERM, set_event_on_signal)
    return event


def _launch_job(
        num_proc, hosts_slots_str,
        command, exec_command,
        settings, env):
    """
    Launches the given command multiple times using rsh.
    Each command is launched via exec_command.

    :param command: command to launch
    :param exec_command: means to execute a single command
    :param settings: settings for the distribution
    :param env: environment to use
    """
    # Make the output directory if it does not exist
    if settings.output_filename:
        _mkdir_p(settings.output_filename)

    # allocate processes into slots
    hosts = parse_hosts(hosts_slots_str)
    host_alloc_plan = get_host_assignments(hosts, num_proc)

    run_command = get_run_command(command, settings.nics)

    slot_info_to_command = _slot_info_to_command_fn(run_command, env)
    event = register_shutdown_event()
    args_list = [[slot_info_to_command(slot_info), slot_info, [event]]
                 for slot_info in host_alloc_plan]

    # If an error occurs in one thread, entire process will be terminated.
    # Otherwise, threads will keep running.
    res = threads.execute_function_multithreaded(exec_command,
                                                 args_list,
                                                 block_until_all_done=True)

    for name, value in sorted(res.items(), key=lambda item: item[1][1]):
        exit_code, timestamp = value
        if exit_code != 0:
            raise RuntimeError('One or more processes exited with non-zero '
                               'status, thus causing the job to be terminated. The first process '
                               'to do so was:\nProcess name: {name}\nExit code: {code}\n'
                               .format(name=name, code=exit_code))
