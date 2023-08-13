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
MYSQL_HOME=$RUNTIME_PATH/mysql

case "$SERVICE_COMMAND" in
start)
    MYSQL_CONFIG_FILE=${MYSQL_HOME}/conf/my.cnf
    nohup mysqld --defaults-file=${MYSQL_CONFIG_FILE} >${MYSQL_HOME}/logs/mysqld.log 2>&1 &
    ;;
stop)
    stop_process_by_name "mysqld"
    ;;
-h|--help)
    echo "Usage: $0 start|stop --head" >&2
    ;;
*)
    echo "Usage: $0 start|stop --head" >&2
    ;;
esac

exit 0
