#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

# import util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

set_head_option "$@"
set_service_command "$@"
set_node_ip_address

USER_HOME=/home/$(whoami)
RUNTIME_PATH=$USER_HOME/runtime
POSTGRES_HOME=$RUNTIME_PATH/postgres

case "$SERVICE_COMMAND" in
start)
    POSTGRES_CONFIG_FILE=${POSTGRES_HOME}/conf/postgresql.conf
    nohup postgres \
        -c config_file=${POSTGRES_CONFIG_FILE} \
        >${POSTGRES_HOME}/logs/postgres.log 2>&1 &
    ;;
stop)
    stop_process_by_name "postgres"
    ;;
-h|--help)
    echo "Usage: $0 start|stop --head" >&2
    ;;
*)
    echo "Usage: $0 start|stop --head" >&2
    ;;
esac

exit 0
