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
PROMETHEUS_HOME=$RUNTIME_PATH/prometheus

function get_data_dir() {
    data_disk_dir=$(get_any_data_disk_dir)
    if [ -z "$data_disk_dir" ]; then
        data_dir="${PROMETHEUS_HOME}/data"
    else
        data_dir="$data_disk_dir/prometheus/data"
    fi
    echo "${data_dir}"
}

case "$SERVICE_COMMAND" in
start)
    if [ "${PROMETHEUS_HIGH_AVAILABILITY}" == "true" ] || [ "${IS_HEAD_NODE}" == "true" ]; then
        PROMETHEUS_CONFIG_FILE=${PROMETHEUS_HOME}/conf/prometheus.yaml
        PROMETHEUS_DATA_PATH=$(get_data_dir)
        SERVICE_PORT=9090
        if [ ! -z "${PROMETHEUS_SERVICE_PORT}" ]; then
            SERVICE_PORT=${PROMETHEUS_SERVICE_PORT}
        fi
        PROMETHEUS_LISTEN_ADDRESS="${NODE_IP_ADDRESS}:${SERVICE_PORT}"
        nohup ${PROMETHEUS_HOME}/prometheus \
          --config.file=${PROMETHEUS_CONFIG_FILE} \
          --storage.tsdb.path=${PROMETHEUS_DATA_PATH} \
          --web.listen-address=${PROMETHEUS_LISTEN_ADDRESS} \
          --web.enable-lifecycle >${PROMETHEUS_HOME}/logs/prometheus.log 2>&1 &
    fi

    NODE_EXPORTER_ADDRESS="${NODE_IP_ADDRESS}:9100"
    nohup ${PROMETHEUS_HOME}/node_exporter/node_exporter \
          --web.listen-address=${NODE_EXPORTER_ADDRESS} >${PROMETHEUS_HOME}/logs/node_exporter.log 2>&1 &
    ;;
stop)
    if [ "${PROMETHEUS_HIGH_AVAILABILITY}" == "true" ] || [ "${IS_HEAD_NODE}" == "true" ]; then
        stop_process_by_name "prometheus"
    fi
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
