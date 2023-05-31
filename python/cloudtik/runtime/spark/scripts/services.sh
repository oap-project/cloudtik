#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

if [ ! -n "${HADOOP_HOME}" ]; then
    echo "HADOOP_HOME environment variable is not set."
    exit 1
fi

if [ ! -n "${SPARK_HOME}" ]; then
    echo "SPARK_HOME environment variable is not set."
    exit 1
fi

USER_HOME=/home/$(whoami)
RUNTIME_PATH=$USER_HOME/runtime
export CLOUD_FS_MOUNT_PATH=/cloudtik/fs

# import util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

# Cloud storage fuse functions
. "$ROOT_DIR"/common/scripts/cloud-storage-fuse.sh

case "$1" in
start-head)
    IS_HEAD_NODE=true
    set_head_address

    # Mount cloud filesystem or hdfs
    mount_cloud_fs

    echo "Starting Resource Manager..."
    $HADOOP_HOME/bin/yarn --daemon start resourcemanager
    echo "Starting Spark History Server..."
    export SPARK_LOCAL_IP=${CLOUDTIK_NODE_IP}; $SPARK_HOME/sbin/start-history-server.sh > /dev/null
    echo "Starting Jupyter..."
    nohup jupyter lab --no-browser > $RUNTIME_PATH/jupyter/logs/jupyterlab.log 2>&1 &
    ;;
stop-head)
    IS_HEAD_NODE=true
    set_head_address

    $HADOOP_HOME/bin/yarn --daemon stop resourcemanager
    $SPARK_HOME/sbin/stop-history-server.sh
    # workaround for stopping jupyter when password being set
    JUPYTER_PID=$(pgrep jupyter)
    if [ -n "$JUPYTER_PID" ]; then
      echo "Stopping Jupyter..."
      kill $JUPYTER_PID >/dev/null 2>&1
    fi

    unmount_cloud_fs
    ;;
start-worker)
    IS_HEAD_NODE=false
    set_head_address

    mount_cloud_fs
    $HADOOP_HOME/bin/yarn --daemon start nodemanager
    ;;
stop-worker)
    IS_HEAD_NODE=false
    set_head_address

    $HADOOP_HOME/bin/yarn --daemon stop nodemanager
    unmount_cloud_fs
    ;;
-h|--help)
    echo "Usage: $0 start|stop --head" >&2
    ;;
*)
    echo "Usage: $0 start|stop --head" >&2
    ;;
esac

exit 0
