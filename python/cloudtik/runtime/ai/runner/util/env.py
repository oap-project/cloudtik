# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
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

import re
import os

# List of regular expressions to ignore environment variables by.
IGNORE_REGEXES = {'BASH_FUNC_.*', 'OLDPWD', '.*_SECRET_KEY'}

RANK_ENVS = ['HOROVOD_RANK', 'OMPI_COMM_WORLD_RANK', 'PMI_RANK', 'RANK']
SIZE_ENVS = ['HOROVOD_SIZE', 'OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'WORLD_SIZE']
LOCAL_RANK_ENVS = ["HOROVOD_LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", "LOCAL_RANK"]
LOCAL_SIZE_ENVS = ["HOROVOD_LOCAL_SIZE", "OMPI_COMM_WORLD_LOCAL_SIZE", "MPI_LOCALNRANKS", "LOCAL_WORLD_SIZE"]


def is_exportable(v):
    return not any(re.match(r, v) for r in IGNORE_REGEXES)


def get_env_rank_and_size():
    for rank_var, size_var in zip(RANK_ENVS, SIZE_ENVS):
        rank = os.environ.get(rank_var)
        size = os.environ.get(size_var)
        if rank is not None and size is not None:
            return int(rank), int(size)
        elif rank is not None or size is not None:
            raise RuntimeError(
                'Could not determine process rank and size: only one of {} and {} '
                'found in environment'.format(rank_var, size_var))

    # Default to rank zero and size one if there are no environment variables
    return 0, 1


def get_env_any(env_names):
    for var in env_names:
        value = os.environ.get(var)
        if value is not None:
            return int(value)

    # None if there are no environment variables
    return None


def get_world_size():
    return get_env_any(SIZE_ENVS)


def get_rank():
    return get_env_any(RANK_ENVS)


def get_local_rank():
    return get_env_any(LOCAL_RANK_ENVS)


def get_local_world_size():
    # not always available
    return get_env_any(LOCAL_SIZE_ENVS)
