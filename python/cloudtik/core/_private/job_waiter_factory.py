import logging
from typing import Any, Dict, Optional

from cloudtik.core._private.core_utils import _load_class
from cloudtik.core._private.runtime_factory import _get_runtime
from cloudtik.core._private.utils import RUNTIME_CONFIG_KEY, RUNTIME_TYPES_CONFIG_KEY
from cloudtik.core.job_waiter import JobWaiter

logger = logging.getLogger(__name__)

BUILT_IN_JOB_WAITER_CHAIN = "chain"


def _import_chain():
    from cloudtik.core._private.job_waiter_chain import JobWaiterChain
    return JobWaiterChain


_BUILT_IN_JOB_WAITERS = {
    "chain": _import_chain,
}


def _get_built_in_job_waiter_cls(job_waiter_name: str):
    """Get the JobWaiter class for the built-in job waiter
    Returns:
        JobWaiter class
    """
    importer = _BUILT_IN_JOB_WAITERS.get(job_waiter_name)
    if importer is None:
        raise NotImplementedError("Unsupported built-in job waiter: {}".format(
            job_waiter_name))
    return importer()


def _parse_built_in_chain(job_waiter_name: str) -> Optional[list[str]]:
    start = job_waiter_name.find("[")
    if start <= 0:
        return None

    end = job_waiter_name.rfind("]", start)
    if end <= 0:
        return None

    name = job_waiter_name[:start]
    if BUILT_IN_JOB_WAITER_CHAIN != name:
        return None

    names_in_chain = []
    chain_str = job_waiter_name[start+1:end]
    if len(chain_str) == 0:
        return names_in_chain

    items = chain_str.split(",")
    for item in items:
        striped_item = item.strip()
        if len(striped_item) > 0:
            names_in_chain += [striped_item]

    return names_in_chain


def _create_built_in_job_waiter_chain(config: Dict[str, Any], job_waiter_name):
    names_in_chain = _parse_built_in_chain(job_waiter_name)
    if names_in_chain is None:
        return None

    if len(names_in_chain) <= 0:
        raise RuntimeError("Job waiter chain is invalid.")

    job_waiter_chain_cls = _get_built_in_job_waiter_cls(BUILT_IN_JOB_WAITER_CHAIN)
    job_waiter_chain = job_waiter_chain_cls(config)

    for job_waiter_name_in_chain in names_in_chain:
        job_waiter_in_chain = create_job_waiter(config, job_waiter_name_in_chain)
        job_waiter_chain.append_job_waiter(job_waiter_in_chain)

    return job_waiter_chain


def _create_built_in_job_waiter(config: Dict[str, Any], job_waiter_name):
    # Check for built-in chain, job_waiter_name may be in the format of name(a,b,c)
    job_waiter = _create_built_in_job_waiter_chain(config, job_waiter_name)
    if job_waiter is not None:
        return job_waiter

    try:
        job_waiter_cls = _get_built_in_job_waiter_cls(job_waiter_name)
        return job_waiter_cls(config)
    except NotImplementedError:
        return None


def create_job_waiter(
        config: Dict[str, Any],
        job_waiter_name: Optional[str] = None) -> Optional[JobWaiter]:
    if job_waiter_name is None:
        return None

    # First try build in
    job_waiter = _create_built_in_job_waiter(config, job_waiter_name)
    if job_waiter is not None:
        # Built-in found
        return job_waiter

    # Then try runtime job waiters
    job_waiter = _create_runtime_job_waiter(config, job_waiter_name)
    if job_waiter is not None:
        # runtime job waiter found
        return job_waiter

    # Then try user customized job waiter
    return _create_user_job_waiter(config, job_waiter_name)


def _create_runtime_job_waiter(config: Dict[str, Any], job_waiter_name):
    runtime_config = config.get(RUNTIME_CONFIG_KEY)
    if runtime_config is None:
        return None

    try:
        runtime = _get_runtime(job_waiter_name, runtime_config)
        return runtime.get_job_waiter(config)
    except NotImplementedError:
        return None


def _get_job_waiter_cls(class_path):
    """Get the JobWaiter class from user specified module and class name.
    Returns:
        JobWaiter class
    """
    job_waiter_class = _load_class(path=class_path)
    if job_waiter_class is None:
        raise NotImplementedError("Cannot load external job waiter class: {}".format(class_path))

    return job_waiter_class


def _create_user_job_waiter(config: Dict[str, Any], job_waiter_name):
    # job_waiter_name is the job waiter class to loader
    job_waiter_cls = _get_job_waiter_cls(job_waiter_name)
    return job_waiter_cls(config)
