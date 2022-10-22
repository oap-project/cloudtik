from collections import namedtuple
import logging
import os
import re
import subprocess
import sys

import cloudtik.core._private.constants as constants
from cloudtik.core._private.core_utils import get_num_cpus, get_cuda_visible_devices, get_system_memory, \
    estimate_available_memory

logger = logging.getLogger(__name__)


class ResourceSpec(
        namedtuple("ResourceSpec", [
            "num_cpus", "num_gpus", "memory",
            "resources", "redis_max_memory"
        ])):
    """Represents the resource configuration passed to a node.

    All fields can be None. Before starting services, resolve() should be
    called to return a ResourceSpec with unknown values filled in with
    defaults based on the local machine specifications.

    num_cpus: The CPUs allocated for this node.
    num_gpus: The GPUs allocated for this node.
    memory: The memory allocated for this node.
    resources: The custom resources allocated for this node.
    redis_max_memory: The max amount of memory (in bytes) to allow each
        redis shard to use. Once the limit is exceeded, redis will start
        LRU eviction of entries. This only applies to the sharded redis
        tables (task, object, and profile tables). By default, this is
        capped at 10GB but can be set higher.
    """

    def __new__(cls,
                num_cpus=None,
                num_gpus=None,
                memory=None,
                resources=None,
                redis_max_memory=None):
        return super(ResourceSpec, cls).__new__(cls, num_cpus, num_gpus,
                                                memory, resources, redis_max_memory)

    def resolved(self):
        """Returns if this ResourceSpec has default values filled out."""
        for v in self._asdict().values():
            if v is None:
                return False
        return True

    def to_resource_dict(self):
        """Returns a dict suitable to pass to node initialization.

        This renames num_cpus / num_gpus to "CPU" / "GPU", translates memory
        from bytes into 100MB memory units, and checks types.
        """
        assert self.resolved()

        memory_units = constants.to_memory_units(
            self.memory, round_up=False)

        resources = dict(
            self.resources,
            CPU=self.num_cpus,
            GPU=self.num_gpus,
            memory=memory_units)

        resources = {
            resource_label: resource_quantity
            for resource_label, resource_quantity in resources.items()
            if resource_quantity != 0
        }

        # Check types.
        for resource_label, resource_quantity in resources.items():
            assert (isinstance(resource_quantity, int)
                    or isinstance(resource_quantity, float)), (
                        f"{resource_label} ({type(resource_quantity)}): "
                        f"{resource_quantity}")
            if (isinstance(resource_quantity, float)
                    and not resource_quantity.is_integer()):
                raise ValueError(
                    "Resource quantities must all be whole numbers. "
                    "Violated by resource '{}' in {}.".format(
                        resource_label, resources))
            if resource_quantity < 0:
                raise ValueError("Resource quantities must be nonnegative. "
                                 "Violated by resource '{}' in {}.".format(
                                     resource_label, resources))
            if resource_quantity > constants.CLOUDTIK_MAX_RESOURCE_QUANTITY:
                raise ValueError("Resource quantities must be at most {}. "
                                 "Violated by resource '{}' in {}.".format(
                                     constants.CLOUDTIK_MAX_RESOURCE_QUANTITY,
                                     resource_label, resources))

        return resources

    def resolve(self, is_head: bool = False, available_memory: bool = True):
        """Returns a copy with values filled out with system defaults.
        """

        resources = (self.resources or {}).copy()
        assert "CPU" not in resources, resources
        assert "GPU" not in resources, resources
        assert "memory" not in resources, resources

        num_cpus = self.num_cpus
        if num_cpus is None:
            num_cpus = get_num_cpus()

        num_gpus = self.num_gpus
        gpu_ids = get_cuda_visible_devices()
        # Check that the number of GPUs that the node wants doesn't
        # exceed the amount allowed by CUDA_VISIBLE_DEVICES.
        if (num_gpus is not None and gpu_ids is not None
                and num_gpus > len(gpu_ids)):
            raise ValueError("Attempting to start node with {} GPUs, "
                             "but CUDA_VISIBLE_DEVICES contains {}.".format(
                                 num_gpus, gpu_ids))
        if num_gpus is None:
            # Try to automatically detect the number of GPUs.
            num_gpus = _autodetect_num_gpus()
            # Don't use more GPUs than allowed by CUDA_VISIBLE_DEVICES.
            if gpu_ids is not None:
                num_gpus = min(num_gpus, len(gpu_ids))

        try:
            info_string = _get_gpu_info_string()
            gpu_types = _constraints_from_gpu_info(info_string)
            resources.update(gpu_types)
        except Exception:
            logger.exception("Could not parse gpu information.")

        # Choose a default object store size.
        system_memory = get_system_memory()
        avail_memory = estimate_available_memory()

        redis_max_memory = self.redis_max_memory
        if redis_max_memory is None:
            redis_max_memory = min(
                constants.CLOUDTIK_DEFAULT_REDIS_MEMORY_MAX_BYTES,
                max(
                    int(avail_memory * constants.CLOUDTIK_DEFAULT_REDIS_MEMORY_PROPORTION),
                    constants.CLOUDTIK_DEFAULT_REDIS_MEMORY_MIN_BYTES))
        if redis_max_memory < constants.CLOUDTIK_DEFAULT_REDIS_MEMORY_MIN_BYTES:
            raise ValueError(
                "Attempting to cap Redis memory usage at {} bytes, "
                "but the minimum allowed is {} bytes.".format(
                    redis_max_memory,
                    constants.CLOUDTIK_DEFAULT_REDIS_MEMORY_MIN_BYTES))

        memory = self.memory
        if memory is None:
            if not available_memory:
                memory = system_memory
            else:
                memory = (avail_memory - (redis_max_memory if is_head else 0))
                if memory < 100e6 and memory < 0.05 * system_memory:
                    raise ValueError(
                        "After taking into account redis memory "
                        "usage, the amount of memory on this node available for "
                        "tasks ({} GB) is less than {}% of total. ".format(
                            round(memory / 1e9, 2),
                            int(100 * (memory / system_memory))))

        spec = ResourceSpec(num_cpus, num_gpus, memory,
                            resources, redis_max_memory)
        assert spec.resolved()
        return spec


def _autodetect_num_gpus():
    """Attempt to detect the number of GPUs on this machine.

    TODO: This currently assumes NVIDIA GPUs on Linux.
    TODO: This currently does not work on macOS.
    TODO: Use a better mechanism for Windows.

    Possibly useful: tensorflow.config.list_physical_devices()

    Returns:
        The number of GPUs if any were detected, otherwise 0.
    """
    result = 0
    if sys.platform.startswith("linux"):
        proc_gpus_path = "/proc/driver/nvidia/gpus"
        if os.path.isdir(proc_gpus_path):
            result = len(os.listdir(proc_gpus_path))
    elif sys.platform == "win32":
        props = "AdapterCompatibility"
        cmdargs = ["WMIC", "PATH", "Win32_VideoController", "GET", props]
        lines = subprocess.check_output(cmdargs).splitlines()[1:]
        result = len([x.rstrip() for x in lines if x.startswith(b"NVIDIA")])
    return result


def _constraints_from_gpu_info(info_str):
    """Parse the contents of a /proc/driver/nvidia/gpus/*/information to get the
gpu model type.

    Args:
        info_str (str): The contents of the file.

    Returns:
        (str) The full model name.
    """
    if info_str is None:
        return {}
    lines = info_str.split("\n")
    full_model_name = None
    for line in lines:
        split = line.split(":")
        if len(split) != 2:
            continue
        k, v = split
        if k.strip() == "Model":
            full_model_name = v.strip()
            break
    pretty_name = _pretty_gpu_name(full_model_name)
    if pretty_name:
        constraint_name = (f"{constants.CLOUDTIK_RESOURCE_CONSTRAINT_PREFIX}"
                           f"{pretty_name}")
        return {constraint_name: 1}
    return {}


def _get_gpu_info_string():
    """Get the gpu type for this machine.

    TODO: All the caveats of _autodetect_num_gpus and we assume only one
    gpu type.

    Returns:
        (str) The gpu's model name.
    """
    if sys.platform.startswith("linux"):
        proc_gpus_path = "/proc/driver/nvidia/gpus"
        if os.path.isdir(proc_gpus_path):
            gpu_dirs = os.listdir(proc_gpus_path)
            if len(gpu_dirs) > 0:
                gpu_info_path = f"{proc_gpus_path}/{gpu_dirs[0]}/information"
                info_str = open(gpu_info_path).read()
                return info_str
    return None


# TODO: This pattern may not work for non NVIDIA Tesla GPUs (which have
# the form "Tesla V100-SXM2-16GB" or "Tesla K80").
GPU_NAME_PATTERN = re.compile(r"\w+\s+([A-Z0-9]+)")


def _pretty_gpu_name(name):
    if name is None:
        return None
    match = GPU_NAME_PATTERN.match(name)
    return match.group(1) if match else None
