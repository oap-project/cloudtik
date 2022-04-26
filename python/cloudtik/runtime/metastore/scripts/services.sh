#!/bin/bash

if [ ! -n "${METASTORE_HOME}" ]; then
    echo "Hive Metastore is not installed."
    exit 1
fi

case "$1" in
start-head)
    sudo service mysql start
    nohup $METASTORE_HOME/bin/start-metastore >${METASTORE_HOME}/logs/start-metastore.log 2>&1 &
    ;;
stop-head)
    sudo service mysql stop
    pkill -f 'HiveMetaStore'
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
