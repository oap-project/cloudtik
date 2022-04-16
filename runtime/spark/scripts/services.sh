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
    $HADOOP_HOME/bin/yarn --daemon start resourcemanager
    $SPARK_HOME/sbin/start-history-server.sh
    which jupyter && nohup jupyter lab  --no-browser --ip=* > /tmp/logs/jupyterlab.log 2>&1 &
    sudo service apache2 start
    sudo service gmetad start
    sudo service ganglia-monitor start
    ;;
stop-head)
    $HADOOP_HOME/bin/yarn --daemon stop resourcemanager
    $SPARK_HOME/sbin/stop-history-server.sh
    if [ "$ENABLE_HDFS" == "true" ]; then
        $HADOOP_HOME/bin/hdfs --daemon stop namenode
    fi
    which jupyter && jupyter lab stop
    sudo service ganglia-monitor stop
    sudo service gmetad stop
    sudo service apache2 stop
    ;;
start-worker)
    if [ "$ENABLE_HDFS" == "true" ]; then
        $HADOOP_HOME/sbin/hadoop-daemon.sh start datanode
    fi
    $HADOOP_HOME/bin/yarn --daemon start nodemanager
    sudo service ganglia-monitor start
    ;;
stop-worker)
    $HADOOP_HOME/bin/yarn --daemon stop nodemanager
    if [ "$ENABLE_HDFS" == "true" ]; then
        $HADOOP_HOME/sbin/hadoop-daemon.sh stop datanode
    fi
    sudo service ganglia-monitor stop
    ;;
-h|--help)
    echo "Usage: $0 start-head|stop-head|start-worker|stop-worker" >&2
    ;;
*)
    echo "Usage: $0 start-head|stop-head|start-worker|stop-worker" >&2
    ;;
esac
