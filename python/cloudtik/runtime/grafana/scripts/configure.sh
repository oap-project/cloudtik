#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

USER_HOME=/home/$(whoami)
RUNTIME_PATH=$USER_HOME/runtime
GRAFANA_HOME=$RUNTIME_PATH/grafana

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function prepare_base_conf() {
    source_dir=$(dirname "${BIN_DIR}")/conf
    output_dir=/tmp/grafana/conf
    rm -rf  $output_dir
    mkdir -p $output_dir
    cp -r $source_dir/* $output_dir
}

function check_grafana_installed() {
    if ! command -v grafana &> /dev/null
    then
        echo "Grafana is not installed for grafana command is not available."
        exit 1
    fi
}

function get_service_port() {
    local service_port=3000
    if [ ! -z "${GRAFANA_SERVICE_PORT}" ]; then
        service_port=${GRAFANA_SERVICE_PORT}
    fi
    echo "${service_port}"
}

function get_data_dir() {
    data_disk_dir=$(get_any_data_disk_dir)
    if [ -z "$data_disk_dir" ]; then
        data_dir="${GRAFANA_HOME}/data"
    else
        data_dir="$data_disk_dir/grafana/data"
    fi
    echo "${data_dir}"
}

function configure_grafana() {
    prepare_base_conf
    grafana_output_dir=$output_dir
    config_template_file=${output_dir}/grafana.ini

    mkdir -p ${GRAFANA_HOME}/logs
    mkdir -p ${GRAFANA_HOME}/plugins

    GRAFANA_CONFIG_DIR=${GRAFANA_HOME}/conf
    mkdir -p ${GRAFANA_CONFIG_DIR}

    sed -i "s#{%server.address%}#${NODE_IP_ADDRESS}#g" ${config_template_file}

    local SERVER_PORT=$(get_service_port)
    sed -i "s#{%server.port%}#${SERVER_PORT}#g" ${config_template_file}

    local DATA_DIR=$(get_data_dir)
    sed -i "s#{%data.dir%}#${DATA_DIR}#g" ${config_template_file}

    local LOG_DIR=${GRAFANA_HOME}/logs
    sed -i "s#{%logs.dir%}#${LOG_DIR}#g" ${config_template_file}

    local PLUGINS_DIR=${GRAFANA_HOME}/plugins
    sed -i "s#{%plugins.dir%}#${PLUGINS_DIR}#g" ${config_template_file}

    local PROVISIONING_DIR=${GRAFANA_HOME}/conf/provisioning
    sed -i "s#{%provisioning.dir%}#${PROVISIONING_DIR}#g" ${config_template_file}

    cp -r ${config_template_file} ${GRAFANA_CONFIG_DIR}/grafana.ini
}

set_head_option "$@"
check_grafana_installed
set_head_address
set_node_ip_address
configure_grafana

exit 0
