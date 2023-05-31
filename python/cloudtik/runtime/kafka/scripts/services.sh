#!/bin/bash

if [ ! -n "${KAFKA_HOME}" ]; then
    echo "KAFKA_HOME environment variable is not set."
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
    # Do nothing for head. Kafka doesn't participate on head node
    if [ $IS_HEAD_NODE == "false" ]; then
        nohup ${KAFKA_HOME}/bin/kafka-server-start.sh ${KAFKA_HOME}/config/server.properties >${KAFKA_HOME}/logs/kafka-server-start.log 2>&1 &
    fi
    ;;
stop)
    # Do nothing for head. Kafka doesn't participate on head node
    if [ $IS_HEAD_NODE == "false" ]; then
        ${KAFKA_HOME}/bin/kafka-server-stop.sh stop
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
