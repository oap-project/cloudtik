#!/bin/bash

if [ ! -n "${HADOOP_HOME}" ]; then
    echo "HADOOP_HOME environment variable is not set."
    exit 1
fi

case "$1" in
start-head)
    $HADOOP_HOME/bin/hdfs --daemon start namenode
    ;;
stop-head)
    $HADOOP_HOME/bin/hdfs --daemon stop namenode
    ;;
start-worker)
    $HADOOP_HOME/sbin/hadoop-daemon.sh start datanode
    ;;
stop-worker)
    $HADOOP_HOME/sbin/hadoop-daemon.sh stop datanode
    ;;
-h|--help)
    echo "Usage: $0 start-head|stop-head|start-worker|stop-worker" >&2
    ;;
*)
    echo "Usage: $0 start-head|stop-head|start-worker|stop-worker" >&2
    ;;
esac

exit 0
