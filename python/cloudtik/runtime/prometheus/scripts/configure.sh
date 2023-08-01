#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

USER_HOME=/home/$(whoami)
RUNTIME_PATH=$USER_HOME/runtime
PROMETHEUS_HOME=$RUNTIME_PATH/prometheus

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function prepare_base_conf() {
    source_dir=$(dirname "${BIN_DIR}")/conf
    output_dir=/tmp/prometheus/conf
    rm -rf  $output_dir
    mkdir -p $output_dir
    cp -r $source_dir/* $output_dir
}

function check_prometheus_installed() {
    if ! command -v prometheus &> /dev/null
    then
        echo "Prometheus is not installed for prometheus command is not available."
        exit 1
    fi
}

function update_local_target() {
    # scrape myself
    SERVICE_PORT=9090
    if [ ! -z "${PROMETHEUS_SERVICE_PORT}" ]; then
        SERVICE_PORT=${PROMETHEUS_SERVICE_PORT}
    fi
    PROMETHEUS_LISTEN_ADDRESS="${NODE_IP_ADDRESS}:${SERVICE_PORT}"
    sed -i "s#{%local.target.address%}#${PROMETHEUS_LISTEN_ADDRESS}#g" ${config_template_file}
}

function configure_prometheus() {
    prepare_base_conf
    cd $output_dir
    prometheus_output_dir=$output_dir

    config_template_file=${output_dir}/prometheus.yaml

    update_local_target

    PROMETHEUS_CONFIG_DIR=${PROMETHEUS_HOME}/conf
    mkdir -p ${PROMETHEUS_CONFIG_DIR}
    cp -r ${config_template_file} ${PROMETHEUS_CONFIG_DIR}/prometheus.yaml
}

set_head_option "$@"
check_prometheus_installed
set_head_address
set_node_ip_address
configure_prometheus

exit 0
