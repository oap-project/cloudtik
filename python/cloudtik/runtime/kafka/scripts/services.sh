#!/bin/bash

if [ ! -n "${KAFKA_HOME}" ]; then
    echo "KAFKA_HOME environment variable is not set."
    exit 1
fi

case "$1" in
start-head)
    # Do nothing for head. Kafka doesn't participate on head node
    # nohup ${KAFKA_HOME}/bin/kafka-server-start.sh ${KAFKA_HOME}/config/server.properties >${KAFKA_HOME}/logs/kafka-server-start.log 2>&1 &
    ;;
stop-head)
    # Do nothing for head. Kafka doesn't participate on head node
    # ${KAFKA_HOME}/bin/kafka-server-stop.sh stop
    ;;
start-worker)
    nohup ${KAFKA_HOME}/bin/kafka-server-start.sh ${KAFKA_HOME}/config/server.properties >${KAFKA_HOME}/logs/kafka-server-start.log 2>&1 &
    ;;
stop-worker)
    ${KAFKA_HOME}/bin/kafka-server-stop.sh stop
    ;;
-h|--help)
    echo "Usage: $0 start|stop --head" >&2
    ;;
*)
    echo "Usage: $0 start|stop --head" >&2
    ;;
esac

exit 0
