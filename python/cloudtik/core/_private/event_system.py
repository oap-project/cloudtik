import threading
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Union

from cloudtik.core._private.cli_logger import cli_logger


class CreateClusterEvent(Enum):
    """Events to track in cloudtik.core.api.create_or_update_cluster.

    Attributes:
        up_started : Invoked at the beginning of create_or_update_cluster.
        ssh_keypair_downloaded : Invoked when the ssh keypair is downloaded.
        cluster_booting_started : Invoked when when the cluster booting starts.
        acquiring_new_head_node : Invoked before the head node is acquired.
        head_node_acquired : Invoked after the head node is acquired.
        ssh_control_acquired : Invoked when the node is being updated.
        run_initialization_cmd : Invoked before all initialization
            commands are called and again before each initialization command.
        run_setup_cmd : Invoked before all setup commands are
            called and again before each setup command.
        start_cloudtik_runtime : Invoked before start commands are run.
        start_cloudtik_runtime_completed : Invoked after start commands
            are run.
        cluster_booting_completed : Invoked after cluster booting
            is completed.
    """

    up_started = auto()
    ssh_keypair_downloaded = auto()
    cluster_booting_started = auto()
    acquiring_new_head_node = auto()
    head_node_acquired = auto()
    ssh_control_acquired = auto()
    run_initialization_cmd = auto()
    run_setup_cmd = auto()
    start_cloudtik_runtime = auto()
    start_cloudtik_runtime_completed = auto()
    cluster_booting_completed = auto()


class _EventSystem:
    """Event system that handles storing and calling callbacks for events.

    Attributes:
        callback_map (Dict[str, Dict[str, List[Callable]]]) : Stores list of callbacks
            for events when registered for unique cluster.
    """

    def __init__(self):
        self.callback_map = {}
        self._lock = threading.Lock()

    def add_callback_handler(
            self,
            cluster_uri: str,
            event: str,
            callback: Union[Callable[[Dict], None], List[Callable[[Dict],
                                                                  None]]],
    ):
        """Stores callback handler for event of the uri

        Args:
            cluster_uri (str): The unique identifier of a cluster which is
                a combination of provider:cluster_name
            event (str): Event that callback should be called on. See
                CreateClusterEvent for details on the events available to be
                registered against.
            callback (Callable[[Dict], None]): Callable object that is invoked
                when specified event occurs.
        """
        if event not in CreateClusterEvent.__members__.values():
            cli_logger.warning(f"{event} is not currently tracked, and this"
                               " callback will not be invoked.")
        with self._lock:
            self.callback_map.setdefault(
                cluster_uri,
                {}).setdefault(
                event,
                []).extend([callback] if type(callback) is not list else callback)

    def execute_callback(self,
                         cluster_uri: str,
                         event: CreateClusterEvent,
                         event_data: Optional[Dict[str, Any]] = None):
        """Executes all callbacks for event of the uri

        Args:
            cluster_uri (str): The unique identifier of a cluster which is
                a combination of provider:cluster_name
            event (str): Event that is invoked. See CreateClusterEvent
                for details on the available events.
            event_data (Dict[str, Any]): Argument that is passed to each
                callable object stored for this particular event.
        """
        if event_data is None:
            event_data = {}

        event_data["event_name"] = event

        # Return a copy and call in the caller's thread context
        callbacks = self._get_callbacks_to_call(cluster_uri, event)
        for callback in callbacks:
            callback(event_data)

    def clear_callbacks_for_event(self,
                                  cluster_uri: str,
                                  event: str):
        """Clears stored callable objects for event.

        Args:
            cluster_uri (str): The unique identifier of a cluster which is
                a combination of provider:cluster_name
            event (str): Event that has callable objects stored in map.
                See CreateClusterEvent for details on the available events.
        """
        with self._lock:
            if cluster_uri in self.callback_map:
                event_callback_map = self.callback_map[cluster_uri]
                if event in event_callback_map:
                    del event_callback_map[event]

    def clear_callbacks_for_cluster(self,
                                    cluster_uri: str):
        """Clears stored callable objects for cluster.

        Args:
            cluster_uri (str): The unique identifier of a cluster which is
                a combination of provider:cluster_name
        """
        with self._lock:
            if cluster_uri in self.callback_map:
                del self.callback_map[cluster_uri]

    def _get_callbacks_to_call(self,
                               cluster_uri: str,
                               event: CreateClusterEvent) -> List[Callable[[Dict], None]]:
        with self._lock:
            if cluster_uri in self.callback_map:
                event_callback_map = self.callback_map[cluster_uri]
                if event in event_callback_map:
                    return [callback for callback in event_callback_map[event]]
        return []


global_event_system = _EventSystem()
