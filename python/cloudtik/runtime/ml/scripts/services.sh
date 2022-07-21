#!/bin/bash

case "$1" in
start-head)
    # Start MLflow service
    ;;
stop-head)
    # Stop MLflow service
    ;;
start-worker)
    # No need to run anything for worker node
    ;;
stop-worker)
    # No need to run anything for worker node
    ;;
-h|--help)
    echo "Usage: $0 start-head|stop-head|start-worker|stop-worker" >&2
    ;;
*)
    echo "Usage: $0 start-head|stop-head|start-worker|stop-worker" >&2
    ;;
esac

exit 0
