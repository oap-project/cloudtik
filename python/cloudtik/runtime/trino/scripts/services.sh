#!/bin/bash

if [ ! -n "${TRINO_HOME}" ]; then
    echo "Trino is not installed for TRINO_HOME environment variable is not set."
    exit 1
fi

case "$1" in
start-head)
    $TRINO_HOME/bin/launcher start
    ;;
stop-head)
    $TRINO_HOME/bin/launcher stop
    ;;
start-worker)
    $TRINO_HOME/bin/launcher start
    ;;
stop-worker)
    $TRINO_HOME/bin/launcher stop
    ;;
-h|--help)
    echo "Usage: $0 start-head|stop-head|start-worker|stop-worker" >&2
    ;;
*)
    echo "Usage: $0 start-head|stop-head|start-worker|stop-worker" >&2
    ;;
esac

exit 0
