#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

# import util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

if [ ! -n "${HADOOP_HOME}" ]; then
    echo "HADOOP_HOME environment variable is not set."
    exit 1
fi

set_head_option "$@"
set_service_command "$@"

# HDFS use its own conf dir
export HADOOP_CONF_DIR= ${HADOOP_HOME}/etc/hdfs

case "$SERVICE_COMMAND" in
start)
    if [ $IS_HEAD_NODE == "true" ]; then
        $HADOOP_HOME/bin/hdfs --daemon start namenode
    else
        $HADOOP_HOME/bin/hdfs --daemon start datanode
    fi
    ;;
stop)
    if [ $IS_HEAD_NODE == "true" ]; then
        $HADOOP_HOME/bin/hdfs --daemon stop namenode
    else
        $HADOOP_HOME/bin/hdfs --daemon stop datanode
    fi
    ;;
-h|--help)
    echo "Usage: $0 start|stop [--head]" >&2
    ;;
*)
    echo "Usage: $0 start|stop [--head]" >&2
    ;;
esac

exit 0
