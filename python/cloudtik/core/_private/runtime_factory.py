import copy
import importlib
import logging
import json
import os
from typing import Any, Dict

import yaml

from cloudtik.core._private.providers import _load_class

logger = logging.getLogger(__name__)

# For caching runtime instantiations across API calls of one python session
_runtime_instances = {}

RUNTIME_MINIMAL_EXTERNAL_CONFIG = {}

def _import_spark(runtime_config):
    from cloudtik.runtime.spark.runtime import SparkRuntime
    return SparkRuntime


def _import_runtime_external(runtime_config):
    runtime_cls = _load_class(path=runtime_config["module"])
    return runtime_cls


def _load_spark_runtime_home():
    import cloudtik.runtime.spark as spark
    return os.path.dirname(spark.__file__)


def _load_spark_defaults_config():
    return os.path.join(_load_spark_runtime_home(), "defaults.yaml")


_RUNTIMES = {
    "spark": _import_spark,
    "external": _import_runtime_external  # Import an external module
}

_RUNTIME_PRETTY_NAMES = {
    "spark": "Spark",
    "external": "External"
}

_RUNTIME_HOMES = {
    "spark": _load_spark_runtime_home,
}

_RUNTIME_DEFAULT_CONFIGS = {
    "spark": _load_spark_defaults_config,
}


def _get_runtime_cls(runtime_type: str, runtime_config: Dict[str, Any]):
    """Get the runtime class for a given runtime config.

    Note that this may be used by private runtimes that proxy methods to
    built-in runtimes, so we should maintain backwards compatibility.

    Args:
        runtime_config: runtime section of the cluster config.

    Returns:
        Runtime class
    """
    importer = _RUNTIMES.get(runtime_type)
    if importer is None:
        raise NotImplementedError("Unsupported runtime: {}".format(
            runtime_type))
    return importer(runtime_config)


def _get_runtime(runtime_type: str, runtime_config: Dict[str, Any],
                 use_cache: bool = True) -> Any:
    """Get the instantiated runtime for a given runtime config.

    Note that this may be used by private runtimes that proxy methods to
    built-in runtimes, so we should maintain backwards compatibility.

    Args:
        runtime_type: the runtime type from the runtime config.
        runtime_config: runtime section of the cluster config.
        use_cache: whether to use a cached definition if available. If
            False, the returned object will also not be stored in the cache.

    Returns:
        Runtime
    """
    runtime_key = (runtime_type, json.dumps(runtime_config, sort_keys=True))
    if use_cache and runtime_key in _runtime_instances:
        return _runtime_instances[runtime_key]

    runtime_cls = _get_runtime_cls(runtime_type, runtime_config)
    new_runtime = runtime_cls(runtime_config)

    if use_cache:
        _runtime_instances[runtime_key] = new_runtime

    return new_runtime


def _clear_runtime_cache():
    global _runtime_instances
    _runtime_instances = {}


def _get_default_config(runtime_type: str, runtime_config):
    """Retrieve a runtime.

    This is an INTERNAL API. It is not allowed to call this from outside.
    """
    if runtime_config["type"] == "external":
        return copy.deepcopy(RUNTIME_MINIMAL_EXTERNAL_CONFIG)
    load_config = _RUNTIME_DEFAULT_CONFIGS.get(runtime_type)
    if load_config is None:
        raise NotImplementedError("Unsupported runtime: {}".format(
            runtime_type))
    path_to_default = load_config()
    with open(path_to_default) as f:
        defaults = yaml.safe_load(f)

    return defaults


def _get_runtime_config_object(runtime_type: str, runtime_config, object_name: str):
    # For external runtime, from the shared config object it there is one
    if runtime_type == "external":
        return {"from": object_name}

    if not object_name.endswith(".yaml"):
        object_name += ".yaml"

    load_config_home = _RUNTIME_HOMES.get(runtime_type)
    if load_config_home is None:
        raise NotImplementedError("Unsupported runtime: {}".format(
            runtime_type))
    path_to_home = load_config_home()
    path_to_config_file = os.path.join(path_to_home, object_name)
    with open(path_to_config_file) as f:
        config_object = yaml.safe_load(f)

    return config_object
