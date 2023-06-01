#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

# import util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

if [ ! -n "${KAFKA_HOME}" ]; then
    echo "KAFKA_HOME environment variable is not set."
    exit 1
fi

set_head_option "$@"
set_service_command "$@"

case "$SERVICE_COMMAND" in
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
