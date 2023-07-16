import copy
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
        self.prefix_output_with_timestamp = False


class HorovodSparkLauncher(HorovodLauncher):
    """
    Launcher for distributed training with Horovod Spark
    """

    def __init__(self, args, distributor):
        super().__init__(args, distributor)

    def run(self):
        # Run with Horovod Spark
        from horovod.spark import run

        args = self.args
        if not args.func:
            raise ValueError("Horovod Spark launcher support running function only.")

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
        for key, value in args.launcher_kwargs.items():
            if hasattr(hargs, key):
                setattr(hargs, key, value)

        env = None
        if args.env:
            # make a copy
            env = copy.copy(args.env)
        if self.environ_set:
            if env is None:
                env = copy.copy(self.environ_set)
            else:
                # update
                env.update(self.environ_set)

        return run(
            func, args=func_args, kwargs=func_kwargs,
            num_proc=num_proc, executable=args.executable,
            use_mpi=hargs.use_mpi, use_gloo=hargs.use_gloo,
            extra_mpi_args=hargs.mpi_args, verbose=hargs.verbose,
            nics=hargs.nics, start_timeout=hargs.start_timeout,
            env=env, stdout=hargs.stdout, stderr=hargs.stderr,
            prefix_output_with_timestamp=hargs.prefix_output_with_timestamp)
