import logging

from cloudtik.runtime.ai.runner.distributed_launcher import DistributedLauncher
from cloudtik.runtime.ai.runner.rsh.rsh_exec import _exec_command_fn, _launch_job, _check_connectivity
from cloudtik.runtime.ai.runner.util.func_call import _run_func

logger = logging.getLogger(__name__)


class RSHLauncherArgs(object):
    def __init__(self):
        self.verbose = None
        self.nics = None

        self.check_connectivity = False
        self.use_ssh = False
        self.ssh_port = None
        self.ssh_identity_file = None

        self.output_filename = None
        self.prefix_output_with_timestamp = None

        self.env = None


class RSHLauncher(DistributedLauncher):
    r"""
     Launcher for distributed training with remote shell
     """

    def __init__(self, args, distributor):
        super().__init__(args, distributor)
        self.rargs = RSHLauncherArgs()
        self._init_launcher_args(self.rargs)

    def setup(self):
        super().setup()

        rargs = self.rargs

        if rargs.check_connectivity:
            hosts_slots_str = self.distributor.hosts_slots_str
            _check_connectivity(hosts_slots_str, rargs)

    def run(self):
        args = self.args

        def run_command(command):
            self._run_command(command)

        if args.func:
            func = self.wrap_func()
            num_proc = self.distributor.num_proc
            return _run_func(
                func, num_proc, run_command,
                executable=args.executable,
                verbose=args.verbose)
        else:
            command = self.get_command_to_run()
            run_command(command)
            return None

    def _run_command(self, command):
        rargs = self.rargs

        env = self._get_env(rargs)

        num_proc = self.distributor.num_proc
        hosts_slots_str = self.distributor.hosts_slots_str

        # Each thread will use rsh command to launch the job on each remote host. If an
        # error occurs in one thread, entire process will be terminated. Otherwise,
        # threads will keep running and ssh session.
        exec_command = _exec_command_fn(num_proc, rargs)
        _launch_job(
            num_proc, hosts_slots_str,
            command, exec_command,
            rargs, env)
