#!/bin/bash

if [ ! -n "${HADOOP_HOME}" ]; then
    echo "HADOOP_HOME environment variable is not set."
    exit 1
fi

if [ ! -n "${FLINK_HOME}" ]; then
    echo "FLINK_HOME environment variable is not set."
    exit 1
fi

case "$1" in
start-head)
    echo "Starting Resource Manager..."
    $HADOOP_HOME/bin/yarn --daemon start resourcemanager
    echo "Starting Flink History Server..."
    # Make sure HADOOP_CLASSPATH is set
    export HADOOP_CLASSPATH=`$HADOOP_HOME/bin/hadoop classpath`
    $FLINK_HOME/bin/historyserver.sh start > /dev/null
    echo "Starting Jupyter..."
    nohup jupyter lab --no-browser > /tmp/logs/jupyterlab.log 2>&1 &
    ;;
stop-head)
    $HADOOP_HOME/bin/yarn --daemon stop resourcemanager
    # Make sure HADOOP_CLASSPATH is set
    export HADOOP_CLASSPATH=`$HADOOP_HOME/bin/hadoop classpath`
    $FLINK_HOME/bin/historyserver.sh stop > /dev/null
    # workaround for stopping jupyter when password being set
    # workaround for stopping jupyter when password being set
    JUPYTER_PID=$(pgrep jupyter)
    if [ -n "$JUPYTER_PID" ]; then
      echo "Stopping Jupyter..."
      kill $JUPYTER_PID >/dev/null 2>&1
    fi
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
