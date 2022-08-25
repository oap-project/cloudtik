#!/bin/bash

if [ ! -n "${HADOOP_HOME}" ]; then
    echo "HADOOP_HOME environment variable is not set."
    exit 1
fi

case "$1" in
start-head)
    echo "Starting Resource Manager..."
    $HADOOP_HOME/bin/yarn --daemon start resourcemanager
    echo "Starting Flink History Server..."
    $FLINK_HOME/bin/historyserver.sh start > /dev/null
    nohup jupyter lab --no-browser > /tmp/logs/jupyterlab.log 2>&1 &
    ;;
stop-head)
    $HADOOP_HOME/bin/yarn --daemon stop resourcemanager
    $FLINK_HOME/bin/historyserver.sh stop > /dev/null
    jupyter lab stop
    ;;
start-worker)
    $HADOOP_HOME/bin/yarn --daemon start nodemanager
    ;;
stop-worker)
    $HADOOP_HOME/bin/yarn --daemon stop nodemanager
    ;;
-h|--help)
    echo "Usage: $0 start-head|stop-head|start-worker|stop-worker" >&2
    ;;
*)
    echo "Usage: $0 start-head|stop-head|start-worker|stop-worker" >&2
    ;;
esac

exit 0
