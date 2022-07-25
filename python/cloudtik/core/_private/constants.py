import os
import sys
import math


def env_integer(key, default):
    if key in os.environ:
        val = os.environ[key]
        if val == "inf":
            return sys.maxsize
        else:
            return int(val)
    return default


def env_bool(key, default):
    if key in os.environ:
        return True if os.environ[key].lower() == "true" else False
    return default


# The size of random ID
ID_SIZE = 28

# If a user does not specify a port for the primary Redis service,
# we attempt to start the service running at this port.
CLOUDTIK_DEFAULT_PORT = 6789

# The environment variable name for specify the address
CLOUDTIK_ADDRESS_ENV = "CLOUDTIK_ADDRESS"

# The default password to prevent redis port scanning attack.
CLOUDTIK_REDIS_DEFAULT_PASSWORD = "434C4F554454494B"

# Directory name where runtime resources will be created & cached.
CLOUDTIK_DEFAULT_RUNTIME_DIR_NAME = "runtime_resources"

CLOUDTIK_LOGGING_ROTATE_MAX_BYTES_ENV = "CLOUDTIK_LOGGING_ROTATE_MAX_BYTES"
CLOUDTIK_LOGGING_ROTATE_BACKUP_COUNT_ENV = "CLOUDTIK_LOGGING_ROTATE_BACKUP_COUNT"

# Env name for override resource spec
CLOUDTIK_RESOURCES_ENV = "CLOUDTIK_OVERRIDE_RESOURCES"

# Env name for user templates directory
CLOUDTIK_USER_TEMPLATES = "CLOUDTIK_USER_TEMPLATES"

LOGGER_FORMAT = (
    "%(asctime)s\t%(levelname)s %(filename)s:%(lineno)s -- %(message)s")
LOGGER_FORMAT_HELP = f"The logging format. default='{LOGGER_FORMAT}'"
LOGGER_LEVEL = "info"
LOGGER_LEVEL_CHOICES = ["debug", "info", "warning", "error", "critical"]
LOGGER_LEVEL_HELP = ("The logging level threshold, choices=['debug', 'info',"
                     " 'warning', 'error', 'critical'], default='info'")

LOGGING_ROTATE_MAX_BYTES = 512 * 1024 * 1024  # 512MB.
LOGGING_ROTATE_BACKUP_COUNT = 5  # 5 Backup files at max.

HEALTHCHECK_EXPIRATION_S = os.environ.get("CLOUDTIK_HEALTHCHECK_EXPIRATION_S", 10)

# The KV namespace of health check
CLOUDTIK_KV_NAMESPACE_HEALTHCHECK = "healthcheck"

# The default maximum number of bytes to reserved for object storage 
CLOUDTIK_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES = 200 * 10**9

# The default proportion of available memory allocated to system and runtime overhead
CLOUDTIK_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION = 0.0

# The default maximum number of bytes that the non-primary Redis shards are
# allowed to use unless overridden by the user.
CLOUDTIK_DEFAULT_REDIS_MAX_MEMORY_BYTES = 10**10
# The smallest cap on the memory used by Redis that we allow.
CLOUDTIK_REDIS_MINIMUM_MEMORY_BYTES = 10**7

# The maximum resource quantity that is allowed. This could be relaxed
CLOUDTIK_MAX_RESOURCE_QUANTITY = 100e12

# Used in gpu detection
CLOUDTIK_RESOURCE_CONSTRAINT_PREFIX = "accelerator_type:"

# Each memory "resource" counts as this many bytes of memory.
CLOUDTIK_MEMORY_RESOURCE_UNIT_BYTES = 1


def to_memory_units(memory_bytes, round_up):
    """Convert from bytes -> memory units."""
    value = memory_bytes / CLOUDTIK_MEMORY_RESOURCE_UNIT_BYTES
    if value < 1:
        raise ValueError(
            "The minimum amount of memory that can be requested is {} bytes, "
            "however {} bytes was asked.".format(CLOUDTIK_MEMORY_RESOURCE_UNIT_BYTES,
                                                 memory_bytes))
    if isinstance(value, float) and not value.is_integer():
        # TODO We currently does not support fractional resources when
        # the quantity is greater than one. We should fix memory resources to
        # be allocated in units of bytes and not 100MB.
        if round_up:
            value = int(math.ceil(value))
        else:
            value = int(math.floor(value))
    return int(value)


# Whether to avoid launching GPU nodes for CPU only tasks.
CLOUDTIK_CONSERVE_GPU_NODES = env_integer("CLOUDTIK_CONSERVE_GPU_NODES", 1)

# How long to wait for a node to start, in seconds.
CLOUDTIK_NODE_START_WAIT_S = env_integer("CLOUDTIK_NODE_START_WAIT_S", 900)

# Interval at which to check if node SSH became available.
CLOUDTIK_NODE_SSH_INTERVAL_S = env_integer("CLOUDTIK_NODE_SSH_INTERVAL_S", 5)

# Abort autoscaling if more than this number of errors are encountered. This
# is a safety feature to prevent e.g. runaway node launches.
CLOUDTIK_MAX_NUM_FAILURES = env_integer("CLOUDTIK_MAX_NUM_FAILURES", 5)

# The maximum number of nodes to launch in a single request.
# Multiple requests may be made for this batch size, up to
# the limit of CLOUDTIK_MAX_CONCURRENT_LAUNCHES.
CLOUDTIK_MAX_LAUNCH_BATCH = env_integer("CLOUDTIK_MAX_LAUNCH_BATCH", 5)

# Max number of nodes to launch at a time.
CLOUDTIK_MAX_CONCURRENT_LAUNCHES = env_integer(
    "CLOUDTIK_MAX_CONCURRENT_LAUNCHES", 10)

# Interval at which to perform autoscaling updates.
CLOUDTIK_UPDATE_INTERVAL_S = env_integer("CLOUDTIK_UPDATE_INTERVAL_S", 5)

# We will attempt to restart on nodes it hasn't heard from
# in more than this interval.
CLOUDTIK_HEARTBEAT_TIMEOUT_S = env_integer("CLOUDTIK_HEARTBEAT_TIMEOUT_S", 30)

CLOUDTIK_HEARTBEAT_PERIOD_SECONDS = env_integer("CLOUDTIK_HEARTBEAT_PERIOD_SECONDS", 1)

# The maximum number of nodes (including failed nodes) that the cluster scaler will
# track for logging purposes.
CLOUDTIK_MAX_NODES_TRACKED = 1500

CLOUDTIK_MAX_FAILURES_DISPLAYED = 20

# The maximum allowed resource demand vector size to guarantee the resource
# demand scheduler bin packing algorithm takes a reasonable amount of time
# to run.
CLOUDTIK_MAX_RESOURCE_DEMAND_VECTOR_SIZE = 1000

# Port that controller prometheus metrics will be exported to
CLOUDTIK_METRIC_PORT = env_integer("CLOUDTIK_METRIC_PORT", 44217)
CLOUDTIK_METRIC_ADDRESS_KEY = "ControllerMetricsAddress"

CLOUDTIK_RESOURCE_REQUEST_CHANNEL = b"cloudtik_resource_request"

# Number of attempts to ping the Redis server. See
# `services.py:wait_for_redis_to_start`.
CLOUDTIK_START_REDIS_WAIT_RETRIES = env_integer("CLOUDTIK_START_REDIS_WAIT_RETRIES", 16)

CLOUDTIK_PROCESSES = [
    # The first element is the substring to filter.
    # The second element, if True, is to filter ps results by command name
    # (only the first 15 charactors of the executable name on Linux);
    # if False, is to filter ps results by command with all its arguments.
    # See STANDARD FORMAT SPECIFIERS section of
    # http://man7.org/linux/man-pages/man1/ps.1.html
    # about comm and args. This can help avoid killing non-cloudtik processes.
    # Format:
    # Keyword to filter, filter by command (True)/filter by args (False)
    # The third element is the process name.
    # The forth element, if node, the process should on all nodes,if head, the process should on head node.
    ["cloudtik_cluster_controller.py", False, "ClusterController", "head"],
    ["cloudtik_node_controller.py", False, "NodeController", "node"],
    ["cloudtik_log_monitor.py", False, "LogMonitor", "node"],
    ["cloudtik_process_reaper.py", False, "ProcessReaper", "node"],
    ["redis-server", False, "RedisServer", "head"],
]


# Max Concurrent SSH Calls to stop Docker
MAX_PARALLEL_SHUTDOWN_WORKERS = env_integer("MAX_PARALLEL_SHUTDOWN_WORKERS",
                                         50)
# Max Concurrent SSH Calls to run on nodes
MAX_PARALLEL_EXEC_NODES = env_integer("MAX_PARALLEL_EXEC_NODES", 50)

# Constants used to define the different process types.
PROCESS_TYPE_CLUSTER_CONTROLLER = "cloudtik_cluster_controller"
PROCESS_TYPE_NODE_CONTROLLER = "cloudtik_node_controller"
PROCESS_TYPE_LOG_MONITOR = "cloudtik_log_monitor"
PROCESS_TYPE_REAPER = "cloudtik_process_reaper"
PROCESS_TYPE_REDIS_SERVER = "redis_server"

# Log file names
LOG_FILE_NAME_CLUSTER_CONTROLLER = f"{PROCESS_TYPE_CLUSTER_CONTROLLER}.log"
LOG_FILE_NAME_NODE_CONTROLLER = f"{PROCESS_TYPE_NODE_CONTROLLER}.log"
LOG_FILE_NAME_LOG_MONITOR = f"{PROCESS_TYPE_LOG_MONITOR}.log"

# Cluster Scaler events are denoted by the ":event_summary:" magic token.
LOG_PREFIX_EVENT_SUMMARY = ":event_summary:"

ERROR_CLUSTER_CONTROLLER_DIED = "cluster_controller_died"
ERROR_NODE_CONTROLLER_DIED = "node_controller_died"
ERROR_LOG_MONITOR_DIED = "log_monitor_died"

# kv namespaces
KV_NAMESPACE_SESSION = "session"

LOG_MONITOR_MAX_OPEN_FILES = 200

LOG_FILE_CHANNEL = "CLOUDKIT_LOG_CHANNEL"

DEFAULT_PROXY_PORT = 6000

# The mount point path of the data disks on both host and container
CLOUDTIK_DATA_DISK_MOUNT_POINT = "/mnt/cloudtik"

# The default location of downloading cloudtik wheels
CLOUDTIK_WHEELS = "https://d30257nes7d4fq.cloudfront.net/downloads/cloudtik"

# The installed python version installed for head and workers
CLOUDTIK_CLUSTER_PYTHON_VERSION = "3.7"

# The default global max workers if not set
CLOUDTIK_DEFAULT_MAX_WORKERS = env_integer("CLOUDTIK_DEFAULT_MAX_WORKERS", 32)

# Cluster status strings
CLOUDTIK_CLUSTER_STATUS_STOPPED = "STOPPED"
CLOUDTIK_CLUSTER_STATUS_RUNNING = "RUNNING"

# The maximum time to wait for the cluster ready.
CLOUDTIK_WAIT_FOR_CLUSTER_READY_TIMEOUT_S = env_integer("CLOUDTIK_WAIT_FOR_CLUSTER_READY_TIMEOUT_S", 600)
CLOUDTIK_WAIT_FOR_CLUSTER_READY_INTERVAL_S = env_integer("CLOUDTIK_WAIT_FOR_CLUSTER_READY_INTERVAL_S", 5)

# Cloudtik env exported for running commands
CLOUDTIK_RUNTIME_ENV_RUNTIMES = "CLOUDTIK_RUNTIMES"
CLOUDTIK_RUNTIME_ENV_HEAD_IP = "CLOUDTIK_HEAD_IP"
CLOUDTIK_RUNTIME_ENV_NODE_IP = "CLOUDTIK_NODE_IP"
CLOUDTIK_RUNTIME_ENV_SECRETS = "CLOUDTIK_SECRETS"
CLOUDTIK_RUNTIME_ENV_NODE_NUMBER = "CLOUDTIK_NODE_NUMBER"
CLOUDTIK_RUNTIME_ENV_NODE_TYPE = "CLOUDTIK_NODE_TYPE"
CLOUDTIK_RUNTIME_ENV_PROVIDER_TYPE = "CLOUDTIK_PROVIDER_TYPE"

# Template for cluster uri
CLOUDTIK_CLUSTER_URI_TEMPLATE = "{}:{}"

# The CloudTik runtime name
CLOUDTIK_RUNTIME_NAME = "cloudtik"

CLOUDTIK_CONFIG_SECRET = "h3EMR4cRSLCswkHTlHi+1kkeisQw/DQf2lbn9jV+/Og="
CLOUDTIK_ENCRYPTION_PREFIX = "[AES]:"

PRIVACY_REPLACEMENT = "VALUE-PROTECTED"
PRIVACY_REPLACEMENT_TEMPLATE = "VALUE-{}PROTECTED"
