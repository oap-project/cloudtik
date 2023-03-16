import logging
import os
import psutil
import datetime
import sys

from cloudtik.core._private.core_utils import get_num_cpus, get_system_memory, get_used_memory
from cloudtik.core._private.metrics import k8s_utils

logger = logging.getLogger(__name__)

# Are we in a K8s pod
IN_KUBERNETES_POD = "KUBERNETES_SERVICE_HOST" in os.environ

# Try to determine if we're in a container.
# Using existence of /sys/fs/cgroup as the criterion is consistent with existing logic
IN_CONTAINER = os.path.exists("/sys/fs/cgroup")


def to_posix_time(dt):
    return (dt - datetime.datetime(1970, 1, 1)).total_seconds()


class MetricsCollector:
    def __init__(self):
        """Initialize the collector object."""
        if IN_KUBERNETES_POD or IN_CONTAINER:
            # psutil does not give a meaningful logical cpu count when in a K8s pod, or
            # in a container in general.
            logical_cpu_count = get_num_cpus()

            # The dashboard expects a physical CPU count as well.
            # This is not always meaningful in a container, but we will go ahead
            # and give the dashboard what it wants using psutil.
            physical_cpu_count = psutil.cpu_count(logical=False)
        else:
            logical_cpu_count = psutil.cpu_count()
            physical_cpu_count = psutil.cpu_count(logical=False)

        self._cpu_counts = (logical_cpu_count, physical_cpu_count)

        self._network_stats_hist = [(0, (0.0, 0.0))]  # time, (sent, recv)
        self._disk_io_stats_hist = [
            (0, (0.0, 0.0, 0, 0))
        ]  # time, (bytes read, bytes written, read ops, write ops)

    def get_all_metrics(self):
        now = to_posix_time(datetime.datetime.utcnow())
        network_stats = self._get_network_stats()
        self._network_stats_hist.append((now, network_stats))
        network_speed_stats = self._compute_speed_from_hist(self._network_stats_hist)

        disk_stats = self._get_disk_io_stats()
        self._disk_io_stats_hist.append((now, disk_stats))
        disk_speed_stats = self._compute_speed_from_hist(self._disk_io_stats_hist)

        return {
            "now": now,
            "cpu": self._get_cpu_percent(IN_KUBERNETES_POD),
            "cpus": self._cpu_counts,
            "mem": self._get_mem_usage(),
            "boot_time": self._get_boot_time(),
            "load_avg": self._get_load_avg(),
            "disk_io": disk_stats,
            "disk_io_speed": disk_speed_stats,
            "network": network_stats,
            "network_speed": network_speed_stats,
        }

    @staticmethod
    def _get_cpu_percent(in_k8s: bool):
        if in_k8s:
            return k8s_utils.cpu_percent()
        else:
            return psutil.cpu_percent()

    @staticmethod
    def _get_mem_usage():
        total = get_system_memory()
        used = get_used_memory()
        available = total - used
        percent = round(used / total, 3)
        return total, available, percent, used

    @staticmethod
    def _get_boot_time():
        if IN_KUBERNETES_POD:
            # Return start time of container entrypoint
            return psutil.Process(pid=1).create_time()
        else:
            return psutil.boot_time()

    def _get_load_avg(self):
        if sys.platform == "win32":
            cpu_percent = psutil.cpu_percent()
            load = (cpu_percent, cpu_percent, cpu_percent)
        else:
            load = os.getloadavg()
        per_cpu_load = tuple((round(x / self._cpu_counts[0], 2) for x in load))
        return load, per_cpu_load

    @staticmethod
    def _get_disk_io_stats():
        stats = psutil.disk_io_counters()
        # stats can be None or {} if the machine is diskless.
        # https://psutil.readthedocs.io/en/latest/#psutil.disk_io_counters
        if not stats:
            return (0, 0, 0, 0)
        else:
            return (
                stats.read_bytes,
                stats.write_bytes,
                stats.read_count,
                stats.write_count,
            )

    @staticmethod
    def _get_network_stats():
        ifaces = [
            v for k, v in psutil.net_io_counters(pernic=True).items() if k[0] == "e"
        ]

        sent = sum((iface.bytes_sent for iface in ifaces))
        recv = sum((iface.bytes_recv for iface in ifaces))
        return sent, recv

    @staticmethod
    def _compute_speed_from_hist(hist):
        while len(hist) > 7:
            hist.pop(0)
        then, prev_stats = hist[0]
        now, now_stats = hist[-1]
        time_delta = now - then
        return tuple((y - x) / time_delta for x, y in zip(prev_stats, now_stats))
