#!/bin/bash

if [ ! -n "${HADOOP_HOME}" ]; then
    echo "HADOOP_HOME environment variable is not set."
    exit 1
fi

if [ ! -n "${SPARK_HOME}" ]; then
    echo "SPARK_HOME environment variable is not set."
    exit 1
fi

case "$1" in
start-head)
    echo "Starting Resource Manager..."
    $HADOOP_HOME/bin/yarn --daemon start resourcemanager
    echo "Starting Spark History Server..."
    export SPARK_LOCAL_IP=${CLOUDTIK_NODE_IP}; $SPARK_HOME/sbin/start-history-server.sh > /dev/null
    nohup jupyter lab --no-browser > /tmp/logs/jupyterlab.log 2>&1 &
    ;;
stop-head)
    $HADOOP_HOME/bin/yarn --daemon stop resourcemanager
    $SPARK_HOME/sbin/stop-history-server.sh
    # workaround for stopping jupyter when password being set
    kill $(pgrep jupyter)
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
