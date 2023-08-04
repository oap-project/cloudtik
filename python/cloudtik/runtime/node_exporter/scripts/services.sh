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
NODE_EXPORTER_HOME=$RUNTIME_PATH/node_exporter

function get_node_exporter_port() {
    local service_port=9100
    if [ ! -z "${NODE_EXPORTER_SERVICE_PORT}" ]; then
        service_port=${NODE_EXPORTER_SERVICE_PORT}
    fi
    echo "${service_port}"
}

case "$SERVICE_COMMAND" in
start)
    NODE_EXPORTER_PORT=$(get_node_exporter_port)
    NODE_EXPORTER_ADDRESS="${NODE_IP_ADDRESS}:${NODE_EXPORTER_PORT}"
    nohup ${NODE_EXPORTER_HOME}/node_exporter \
          --web.listen-address=${NODE_EXPORTER_ADDRESS} >${NODE_EXPORTER_HOME}/logs/node_exporter.log 2>&1 &
    ;;
stop)
    stop_process_by_name "node_exporter"
    ;;
-h|--help)
    echo "Usage: $0 start|stop --head" >&2
    ;;
*)
    echo "Usage: $0 start|stop --head" >&2
    ;;
esac

exit 0
