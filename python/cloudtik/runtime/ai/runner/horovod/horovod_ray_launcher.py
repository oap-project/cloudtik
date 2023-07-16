import logging

from cloudtik.runtime.ai.runner.horovod.horovod_launcher import HorovodLauncher

logger = logging.getLogger(__name__)


class _HorovodRayArgs(object):
    def __init__(self):
        self.verbose = None
        self.nics = None

        self.timeout_s = None
        self.placement_group_timeout_s = None

        self.cpus_per_worker = None
        self.use_gpu = None
        self.gpus_per_worker = None
        self.use_current_placement_group = None

        self.env = None


class HorovodRayLauncher(HorovodLauncher):
    """
    Launcher for distributed training with Horovod on Ray
    """

    def __init__(self, args, distributor):
        super().__init__(args, distributor)

    def run(self):
        # Run with Horovod on Ray
        from horovod.ray import RayExecutor

        args = self.args
        if not args.func:
            raise ValueError("Horovod on Ray launcher support running function only.")

        hargs = _HorovodRayArgs()

        func = args.run_func
        func_args = args.func_args
        if func_args is None:
            func_args = ()
        func_kwargs = args.func_kwargs
        if func_kwargs is None:
            func_kwargs = {}

        # set the launcher arguments (run CLI or run API)
        hargs.verbose = args.verbose
        hargs.nics = args.nics

        # set extra arguments passing from run API
        self._set_args(hargs, args.launcher_kwargs)

        env = self._get_env(hargs)

        settings_kwargs = self._get_kwargs(
            hargs, ["verbose", "nics", "timeout_s",
                    "placement_group_timeout_s"])
        settings = RayExecutor.create_settings(**settings_kwargs)

        executor_kwargs = self._get_kwargs(
            hargs, ["cpus_per_worker", "use_gpu", "gpus_per_worker",
                    "use_current_placement_group"])
        # The executor uses two different strategies for the two cases:
        # 1. if num_workers specified -> PGStrategy
        # 2. if num_hosts and num_workers_per_host specified -> ColocatedStrategy
        num_proc = self.distributor.num_proc
        nnodes = self.distributor.nnodes
        nproc_per_node = self.distributor.nproc_per_node
        if self.distributor.origin_num_proc:
            # User specify num_proc, consider to be case #1
            nnodes = None
        else:
            # User don't specify num_proc, consider to be case #2
            num_proc = None
        executor = RayExecutor(
            settings,
            num_workers=num_proc,
            num_hosts=nnodes,
            num_workers_per_host=nproc_per_node,
            **executor_kwargs)

        executor.start(extra_env_vars=env)
        result = executor.run(func, args=func_args, kwargs=func_kwargs)
        executor.shutdown()

        return result
