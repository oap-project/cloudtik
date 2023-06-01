#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

# import util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

if [ ! -n "${ZOOKEEPER_HOME}" ]; then
    echo "ZOOKEEPER_HOME environment variable is not set."
    exit 1
fi

set_head_option "$@"
set_service_command "$@"

case "$SERVICE_COMMAND" in
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
