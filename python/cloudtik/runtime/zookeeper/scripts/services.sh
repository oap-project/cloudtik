#!/bin/bash

if [ ! -n "${ZOOKEEPER_HOME}" ]; then
    echo "ZOOKEEPER_HOME environment variable is not set."
    exit 1
fi

case "$1" in
start-head)
    # Do nothing for head. Zookeeper doesn't participate on head node
    ;;
stop-head)
    # Do nothing for head. Zookeeper doesn't participate on head node
    ;;
start-worker)
    ${ZOOKEEPER_HOME}/bin/zkServer.sh start
    ;;
stop-worker)
    ${ZOOKEEPER_HOME}/bin/zkServer.sh stop
    ;;
-h|--help)
    echo "Usage: $0 start-head|stop-head|start-worker|stop-worker" >&2
    ;;
*)
    echo "Usage: $0 start-head|stop-head|start-worker|stop-worker" >&2
    ;;
esac

exit 0
