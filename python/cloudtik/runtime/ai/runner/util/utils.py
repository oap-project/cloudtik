import logging
import re
import sys

from cloudtik.runtime.ai.runner.util import tiny_shell_exec

logger = logging.getLogger(__name__)

# List of regular expressions to ignore environment variables by.
IGNORE_REGEXES = {'BASH_FUNC_.*', 'OLDPWD', '.*_SECRET_KEY'}

# MPI implementations
_OMPI_IMPL = 'OpenMPI'
_SMPI_IMPL = 'SpectrumMPI'
_MPICH_IMPL = 'MPICH'
_IMPI_IMPL = "IntelMPI"
_UNKNOWN_IMPL = 'Unknown'
_MISSING_IMPL = 'Missing'

# Open MPI Flags
_OMPI_FLAGS = ['-mca pml ob1', '-mca btl ^openib']
# Spectrum MPI Flags
_SMPI_FLAGS = []
_SMPI_FLAGS_TCP = ['-tcp']
# MPICH Flags
_MPICH_FLAGS = []
# Intel MPI Flags
_IMPI_FLAGS = []

# Threshold for large cluster MPI issues:
_LARGE_CLUSTER_THRESHOLD = 64
# No process binding args
_NO_BINDING_ARGS = ['-bind-to none', '-map-by slot']
# Process socket binding args
_SOCKET_BINDING_ARGS = ['-bind-to socket', '-map-by socket', '-rank-by core']

# MPI not found error message
_MPI_NOT_FOUND_ERROR_MSG = ('We does not find an installed MPI.\n\n'
                            'Choose one of:\n'
                            '1. Install Open MPI 4.0.0+ or IBM Spectrum MPI or MPICH '
                            '(use --no-cache-dir pip option).\n'
                            '2. Use built-in gloo option provided Horovod launcher.')


def is_exportable(v):
    return not any(re.match(r, v) for r in IGNORE_REGEXES)


def mpi_available(env=None):
    return _get_mpi_implementation(env) not in {_UNKNOWN_IMPL, _MISSING_IMPL}


def is_open_mpi(env=None):
    return _get_mpi_implementation(env) == _OMPI_IMPL


def is_spectrum_mpi(env=None):
    return _get_mpi_implementation(env) == _SMPI_IMPL


def is_mpich(env=None):
    return _get_mpi_implementation(env) == _MPICH_IMPL


def is_intel_mpi(env=None):
    return _get_mpi_implementation(env) == _IMPI_IMPL


def is_impi_or_mpich(env=None):
    mpi_impl_flags, _, mpi = _get_mpi_implementation_flags(False, env=env)
    if mpi_impl_flags is None:
        raise Exception(_MPI_NOT_FOUND_ERROR_MSG)

    return mpi in (_IMPI_IMPL, _MPICH_IMPL)


def _get_mpi_implementation(env=None):
    """
    Detects the available MPI implementation by invoking `mpirun --version`.
    This command is executed by the given execute function, which takes the
    command as the only argument and returns (output, exit code). Output
    represents the stdout and stderr as a string.

    Returns one of:
    - _OMPI_IMPL, _SMPI_IMPL, _MPICH_IMPL or _IMPI_IMPL for known implementations
    - _UNKNOWN_IMPL for any unknown implementation
    - _MISSING_IMPL if `mpirun --version` could not be executed.

    :param env: environment variable to use to run mpirun
    :return: string representing identified implementation
    """
    command = 'mpirun --version'
    res = tiny_shell_exec.execute(command, env)
    if res is None:
        return _MISSING_IMPL
    (output, exit_code) = res

    if exit_code == 0:
        if 'Open MPI' in output or 'OpenRTE' in output:
            return _OMPI_IMPL
        elif 'IBM Spectrum MPI' in output:
            return _SMPI_IMPL
        elif 'MPICH' in output:
            return _MPICH_IMPL
        elif 'Intel(R) MPI' in output:
            return _IMPI_IMPL

        print('Unknown MPI implementation given in output of mpirun --version:', file=sys.stderr)
        print(output, file=sys.stderr)
        return _UNKNOWN_IMPL
    else:
        print('Was unable to run {command}:'.format(command=command), file=sys.stderr)
        print(output, file=sys.stderr)
        return _MISSING_IMPL


def _get_mpi_implementation_flags(tcp_flag, env=None):
    if is_open_mpi(env):
        return list(_OMPI_FLAGS), list(_NO_BINDING_ARGS), _OMPI_IMPL
    elif is_spectrum_mpi(env):
        return (list(_SMPI_FLAGS_TCP) if tcp_flag else list(_SMPI_FLAGS)), list(_SOCKET_BINDING_ARGS), _SMPI_IMPL
    elif is_mpich(env):
        return list(_MPICH_FLAGS), list(_NO_BINDING_ARGS), _MPICH_IMPL
    elif is_intel_mpi(env):
        return list(_IMPI_FLAGS), [], _IMPI_IMPL
    else:
        return None, None, None
