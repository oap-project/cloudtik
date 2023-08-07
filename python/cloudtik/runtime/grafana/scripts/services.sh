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
GRAFANA_HOME=$RUNTIME_PATH/grafana

case "$SERVICE_COMMAND" in
start)
    if [ "${GRAFANA_HIGH_AVAILABILITY}" == "true" ] \
        || [ "${IS_HEAD_NODE}" == "true" ]; then
        GRAFANA_CONFIG_FILE=${GRAFANA_HOME}/conf/grafana.ini
        GRAFANA_PID_FILE=${GRAFANA_HOME}/grafana.pid
        nohup ${GRAFANA_HOME}/bin/grafana server \
          --config ${GRAFANA_CONFIG_FILE} \
          --homepath ${GRAFANA_HOME} \
          --pidfile ${GRAFANA_PID_FILE} >${GRAFANA_HOME}/logs/grafana.log 2>&1 &
    fi
    ;;
stop)
    if [ "${GRAFANA_HIGH_AVAILABILITY}" == "true" ] \
        || [ "${IS_HEAD_NODE}" == "true" ]; then
        stop_process_by_name "grafana"
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
