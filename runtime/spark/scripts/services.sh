#!/bin/bash

if [ ! -n "${HADOOP_HOME}" ]; then
  echo "HADOOP_HOME environment variable is not set."
  exit 1
fi

case "$1" in
  start-head)
    $HADOOP_HOME/bin/yarn --daemon start resourcemanager
    which jupyter && nohup jupyter lab > jupyterlab.log 2>&1 &
    ;;
  stop-head)
    $HADOOP_HOME/bin/yarn --daemon stop resourcemanager
    which jupyter && jupyter lab stop
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
