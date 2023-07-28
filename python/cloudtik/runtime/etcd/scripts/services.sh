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

USER_HOME=/home/$(whoami)
RUNTIME_PATH=$USER_HOME/runtime
ETCD_HOME=$RUNTIME_PATH/etcd

case "$SERVICE_COMMAND" in
start)
    ETCD_CONFIG_FILE=${ETCD_HOME}/conf/etcd.yaml
    nohup etcd --config-file=${ETCD_CONFIG_FILE} >/dev/null 2>&1 &
    ;;
stop)
    # We don't use the service
    # sudo service etcd start
    # Stop etcd
    ETCD_PID=$(pgrep etcd)
    if [ -n "${ETCD_PID}" ]; then
      echo "Stopping etcd..."
      # SIGTERM = 15
      kill -15 ${ETCD_PID} >/dev/null 2>&1
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
