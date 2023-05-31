#!/bin/bash

if [ ! -n "${ZOOKEEPER_HOME}" ]; then
    echo "ZOOKEEPER_HOME environment variable is not set."
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
    # Do nothing for head. Zookeeper doesn't participate on head node
    if [ $IS_HEAD_NODE == "false" ]; then
        ${ZOOKEEPER_HOME}/bin/zkServer.sh start
    fi
    ;;
stop)
    # Do nothing for head. Zookeeper doesn't participate on head node
    if [ $IS_HEAD_NODE == "false" ]; then
        ${ZOOKEEPER_HOME}/bin/zkServer.sh stop
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
