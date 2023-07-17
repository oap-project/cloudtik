import logging

from cloudtik.runtime.ai.runner.horovod.horovod_launcher import HorovodLauncher

logger = logging.getLogger(__name__)


class _HorovodSparkArgs(object):
    def __init__(self):
        self.verbose = None
        self.mpi_args = None

        self.use_gloo = None
        self.use_mpi = None
        self.nics = None

        self.start_timeout = None
        self.env = None
        self.stdout = None
        self.stderr = None
        self.prefix_output_with_timestamp = None


class HorovodSparkLauncher(HorovodLauncher):
    """
    Launcher for distributed training with Horovod on Spark
    """

    def __init__(self, args, distributor):
        super().__init__(args, distributor)

    def setup(self):
        # TODO: how about the master addr and port for such cases
        pass

    def run(self):
        # Run with Horovod on Spark
        from horovod.spark import run

        args = self.args
        if not args.func:
            raise ValueError("Horovod on Spark launcher support running function only.")

        hargs = _HorovodSparkArgs()

        num_proc = self.distributor.num_proc
        func = args.run_func
        func_args = args.func_args
        if func_args is None:
            func_args = ()
        func_kwargs = args.func_kwargs
        if func_kwargs is None:
            func_kwargs = {}

        # set the launcher arguments (run CLI or run API)
        hargs.verbose = args.verbose
        hargs.mpi_args = args.mpi_args
        hargs.use_mpi = args.use_mpi
        hargs.use_gloo = args.use_gloo
        hargs.nics = args.nics

        # set extra arguments passing from run API
        self._set_args(hargs, args.launcher_kwargs)

        env = self._get_env(hargs)

        run_kwargs = self._get_kwargs(
            hargs, ["verbose", "use_mpi", "use_gloo", "nics",
                    "start_timeout", "stdout", "stderr",
                    "prefix_output_with_timestamp"])
        return run(
            func, args=func_args, kwargs=func_kwargs,
            num_proc=num_proc, executable=args.executable,
            extra_mpi_args=hargs.mpi_args, env=env,
            **run_kwargs)
