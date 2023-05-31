#!/bin/bash

if [ ! -n "${PRESTO_HOME}" ]; then
    echo "Presto is not installed for PRESTO_HOME environment variable is not set."
    exit 1
fi

case "$1" in
start-head)
    $PRESTO_HOME/bin/launcher start
    ;;
stop-head)
    $PRESTO_HOME/bin/launcher stop
    ;;
start-worker)
    $PRESTO_HOME/bin/launcher start
    ;;
stop-worker)
    $PRESTO_HOME/bin/launcher stop
    ;;
-h|--help)
    echo "Usage: $0 start|stop --head" >&2
    ;;
*)
    echo "Usage: $0 start|stop --head" >&2
    ;;
esac

exit 0
