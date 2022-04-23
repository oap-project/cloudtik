#!/bin/bash

if [ ! -n "${HADOOP_HOME}" ]; then
    echo "HADOOP_HOME environment variable is not set."
    exit 1
fi

case "$1" in
start-head)
    if [ "$ENABLE_HDFS" == "true" ]; then
        $HADOOP_HOME/bin/hdfs --daemon start namenode
    fi
    echo "Starting Resource Manager..."
    $HADOOP_HOME/bin/yarn --daemon start resourcemanager
    echo "Starting Spark History Server..."
    export SPARK_LOCAL_IP=${CLOUDTIK_HEAD_IP}; $SPARK_HOME/sbin/start-history-server.sh > /dev/null
    nohup jupyter lab --no-browser > /tmp/logs/jupyterlab.log 2>&1 &
    ;;
stop-head)
    $HADOOP_HOME/bin/yarn --daemon stop resourcemanager
    $SPARK_HOME/sbin/stop-history-server.sh
    if [ "$ENABLE_HDFS" == "true" ]; then
        $HADOOP_HOME/bin/hdfs --daemon stop namenode
    fi
    jupyter lab stop
    ;;
start-worker)
    if [ "$ENABLE_HDFS" == "true" ]; then
        $HADOOP_HOME/sbin/hadoop-daemon.sh start datanode
    fi
    $HADOOP_HOME/bin/yarn --daemon start nodemanager
    ;;
stop-worker)
    $HADOOP_HOME/bin/yarn --daemon stop nodemanager
    if [ "$ENABLE_HDFS" == "true" ]; then
        $HADOOP_HOME/sbin/hadoop-daemon.sh stop datanode
    fi
    ;;
-h|--help)
    echo "Usage: $0 start-head|stop-head|start-worker|stop-worker" >&2
    ;;
*)
    echo "Usage: $0 start-head|stop-head|start-worker|stop-worker" >&2
    ;;
esac
