#!/bin/bash

if [ ! -n "${HADOOP_HOME}" ]; then
    echo "HADOOP_HOME environment variable is not set."
    exit 1
fi

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
