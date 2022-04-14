#!/bin/bash

args=$(getopt -a -o node_action:, enable_hdfs: -- "$@")
eval set -- "${args}"

if [ ! -n "${HADOOP_HOME}" ]; then
  echo "HADOOP_HOME environment variable is not set."
  exit 1
fi

while true
do
    case "$1" in
    --node_action)
        NODE_ACTION=$2
        shift
        ;;
    --enable_hdfs)
        ENABLE_HDFS=$2
        shift
        ;;
    esac
    shift
done

case "$NODE_ACTION" in
  start-head)
    $HADOOP_HOME/bin/yarn --daemon start resourcemanager
    if [ $ENABLE_HDFS == "true" ]; then
        $HADOOP_HOME/sbin/hadoop-daemon.sh start namenode
    fi
    $SPARK_HOME/sbin/start-history-server.sh
    which jupyter && nohup jupyter lab  --no-browser --ip=* > jupyterlab.log 2>&1 &
    sudo service apache2 start
    sudo service gmetad start
    sudo service ganglia-monitor start
    ;;
  stop-head)
    $HADOOP_HOME/bin/yarn --daemon stop resourcemanager
    if [ $ENABLE_HDFS == "true" ]; then
        $HADOOP_HOME/sbin/hadoop-daemon.sh stop namenode
    fi
    $SPARK_HOME/sbin/stop-history-server.sh
    which jupyter && jupyter lab stop
    sudo service ganglia-monitor stop
    sudo service gmetad stop
    sudo service apache2 stop
    ;;
  start-worker)
    $HADOOP_HOME/bin/yarn --daemon start nodemanager
    if [ $ENABLE_HDFS == "true" ]; then
        $HADOOP_HOME/sbin/hadoop-daemon.sh start datanode
    fi
    sudo service ganglia-monitor start
    ;;
  stop-worker)
    $HADOOP_HOME/bin/yarn --daemon stop nodemanager
    if [ $ENABLE_HDFS == "true" ]; then
        $HADOOP_HOME/sbin/hadoop-daemon.sh stop datanode
    fi
    sudo service ganglia-monitor stop
    ;;
  -h|--help)
    echo "Usage: $0 --node_action=start-head|stop-head|start-worker|stop-worker --enable_hdfs=\$ENABLE_HDFS" >&2
    ;;
  *)
    echo "Usage: $0 --node_action=start-head|stop-head|start-worker|stop-worker --enable_hdfs=\$ENABLE_HDFS" >&2
    ;;
esac
