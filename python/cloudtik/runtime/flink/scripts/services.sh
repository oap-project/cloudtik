#!/bin/bash

if [ ! -n "${HADOOP_HOME}" ]; then
    echo "HADOOP_HOME environment variable is not set."
    exit 1
fi

if [ ! -n "${FLINK_HOME}" ]; then
    echo "FLINK_HOME environment variable is not set."
    exit 1
fi

USER_HOME=/home/$(whoami)
RUNTIME_PATH=$USER_HOME/runtime

command=$1
shift

# Parsing arguments
IS_HEAD_NODE=false

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    -h|--head)
        IS_HEAD_NODE=true
        ;;
    *)
        echo "Unknown argument passed."
        exit 1
    esac
    shift
done

case "$command" in
start)
    if [ $IS_HEAD_NODE == "true" ]; then
        echo "Starting Resource Manager..."
        $HADOOP_HOME/bin/yarn --daemon start resourcemanager
        echo "Starting Flink History Server..."
        # Make sure HADOOP_CLASSPATH is set
        export HADOOP_CLASSPATH=`$HADOOP_HOME/bin/hadoop classpath`
        $FLINK_HOME/bin/historyserver.sh start > /dev/null
        echo "Starting Jupyter..."
        nohup jupyter lab --no-browser > $RUNTIME_PATH/jupyter/logs/jupyterlab.log 2>&1 &
    else
        $HADOOP_HOME/bin/yarn --daemon start nodemanager
    fi
    ;;
stop)
    if [ $IS_HEAD_NODE == "true" ]; then
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
    else
        $HADOOP_HOME/bin/yarn --daemon stop nodemanager
    fi
    ;;
-h|--help)
    echo "Usage: $0 start|stop --head" >&2
    ;;
*)
    echo "Usage: $0 start|stop --head" >&2
    ;;
esac

exit 0
