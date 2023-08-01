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

function update_home_dir() {
    prepare_base_conf
    prometheus_output_dir=$output_dir
    config_template_file=${output_dir}/prometheus.yaml

    mkdir -p ${PROMETHEUS_HOME}/logs
    sed -i "s#{%prometheus.home%}#${PROMETHEUS_HOME}#g" `grep "{%prometheus.home%}" -rl ${output_dir}`

    PROMETHEUS_CONFIG_DIR=${PROMETHEUS_HOME}/conf
    mkdir -p ${PROMETHEUS_CONFIG_DIR}
    cp -r $output_dir/* ${PROMETHEUS_CONFIG_DIR}/
}

set_head_option "$@"
check_prometheus_installed
set_head_address
set_node_ip_address
configure_prometheus

exit 0
