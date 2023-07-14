import argparse
import logging
import os

from cloudtik.runtime.ai.runner.distributed_training_launcher import DistributedTrainingLauncher

logger = logging.getLogger(__name__)


def make_nic_action():
    # This is an append Action that splits the values on ','
    class NicAction(argparse.Action):
        def __init__(self,
                     option_strings,
                     dest,
                     default=None,
                     type=None,
                     choices=None,
                     required=False,
                     help=None):
            super(NicAction, self).__init__(
                option_strings=option_strings,
                dest=dest,
                nargs=1,
                default=default,
                type=type,
                choices=choices,
                required=required,
                help=help)

        def __call__(self, parser, args, values, option_string=None):
            if ',' in values[0]:
                values = values[0].split(',')

            # union the existing dest nics with the new ones
            items = getattr(args, self.dest, None)
            items = set() if items is None else items
            items = items.union(values)

            setattr(args, self.dest, items)

    return NicAction


def add_horovod_params(parser):
    group = parser.add_argument_group("Horovod Parameters")
    group.add_argument(
        '--gloo',
        action='store_true', dest='use_gloo',
        help='Run Horovod using the Gloo controller. This will '
             'be the default if Horovod was not built with MPI support.')
    group.add_argument(
        '--mpi',
        action='store_true', dest='use_mpi',
        help='Run Horovod using the MPI controller. This will '
             'be the default if Horovod was built with MPI support.')
    group.add_argument(
        '--network-interfaces', '--network_interfaces',
        action=make_nic_action(), dest='nics',
        help='Network interfaces that can be used for communication separated by '
             'comma. If not specified, will find the common NICs among all '
             'the workers. Example: --network-interfaces "eth0,eth1".')
    group.add_argument(
        '--output-filename', '--output_filename',
        action='store',
        help='For Gloo, writes stdout / stderr of all processes to a filename of the form '
             '<output_filename>/rank.<rank>/<stdout | stderr>. The <rank> will be padded with 0 '
             'characters to ensure lexicographical order. For MPI, delegates its behavior to mpirun.')


class HorovodTrainingLauncher(DistributedTrainingLauncher):
    r"""
     Launcher for distributed training with Horovod
     """

    def __init__(self, args, distributor):
        super().__init__(args, distributor)

    def get_command_to_run(self):
        args = self.args
        cmd = []
        self.with_python_command(cmd)
        cmd.extend(args.command)
        return cmd

    def run(self):
        args = self.args

        # Run with Horovod
        from horovod.runner import _HorovodArgs
        from horovod.runner.launch import _run

        hargs = _HorovodArgs()

        hargs.num_proc = self.distributor.num_proc
        hargs.hosts = self.distributor.hosts_slots_str

        hargs.verbose = args.verbose
        hargs.mpi_args = args.mpi_args
        hargs.use_mpi = args.use_mpi
        hargs.use_gloo = args.use_gloo
        hargs.nics = args.nics
        hargs.output_filename = args.output_filename

        if args.run_func:
            hargs.run_func = args.run_func
            hargs.executable = args.executable
        else:
            command = self.get_command_to_run()
            hargs.command = command

        if self.environ_set:
            # Horovod use os.environ
            for k, v in self.environ_set.items():
                os.environ[k] = v

        return _run(hargs)
