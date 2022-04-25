#!/bin/bash

if [ ! -n "${METASTORE_HOME}" ]; then
    echo "Hive Metastore is not installed."
    exit 1
fi

case "$1" in
start-head)
    sudo service mysql start
    $METASTORE_HOME/bin/start-metastore &
    ;;
stop-head)
    sudo service mysql stop
    #TODO: design a way to stop hive metastore
    ps -aux|grep metastore|awk '{print $2}'|xargs kill
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
