#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

USER_HOME=/home/$(whoami)
RUNTIME_PATH=$USER_HOME/runtime
BIND_HOME=$RUNTIME_PATH/bind
BIND_BACKUP_RESOLV_CONF=${BIND_HOME}/conf/resolv.conf.backup

# import util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

set_head_option "$@"
set_service_command "$@"

case "$SERVICE_COMMAND" in
start)
    sudo service named start

    if [ "${BIND_DEFAULT_RESOLVER}" == "true" ]; then
        # update the /etc/resolv.conf
        update_resolv_conf ${BIND_BACKUP_RESOLV_CONF} "127.0.0.1"
    fi
    ;;
stop)
    if [ "${BIND_DEFAULT_RESOLVER}" == "true" ]; then
        # restore the /etc/resolv.conf
        restore_resolv_conf ${BIND_BACKUP_RESOLV_CONF}
    fi

    sudo service named stop
    ;;
-h|--help)
    echo "Usage: $0 start|stop --head" >&2
    ;;
*)
    echo "Usage: $0 start|stop --head" >&2
    ;;
esac

exit 0
