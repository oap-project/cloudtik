#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

USER_HOME=/home/$(whoami)
RUNTIME_PATH=$USER_HOME/runtime
CONSUL_HOME=$RUNTIME_PATH/consul

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function prepare_base_conf() {
    source_dir=$(dirname "${BIN_DIR}")/conf
    output_dir=/tmp/consul/conf
    rm -rf  $output_dir
    mkdir -p $output_dir
    cp -r $source_dir/* $output_dir
}

function check_consul_installed() {
    if ! command -v consul &> /dev/null
    then
        echo "Consul is not installed for consul command is not available."
        exit 1
    fi
}

function update_consul_data_dir() {
    data_disk_dir=$(get_any_data_disk_dir)
    if [ -z "$data_disk_dir" ]; then
        consul_data_dir="$CONSUL_HOME/data"
    else
        consul_data_dir="$data_disk_dir/consul/data"
    fi

    mkdir -p ${consul_data_dir}
    sed -i "s!{%data.dir%}!${consul_data_dir}!g" ${consul_output_dir}/consul.json
}

function update_ui_config() {
    if [ "$IS_HEAD_NODE" == "true" ]; then
        UI_ENABLED=true
    else
        UI_ENABLED=false
    fi
    sed -i "s!{%ui.enabled%}!${UI_ENABLED}!g" ${consul_output_dir}/server.json
}

function configure_consul() {
    prepare_base_conf
    consul_output_dir=$output_dir/consul

    mkdir -p ${CONSUL_HOME}/logs

    # General agent configuration. retry_join will be set in python script
    local DATA_CENTER=default
    if [ ! -z "${CONSUL_DATA_CENTER}" ]; then
        DATA_CENTER=${CONSUL_DATA_CENTER}
    fi
    sed -i "s!{%data.center%}!${DATA_CENTER}!g" ${consul_output_dir}/consul.json
    sed -i "s!{%bind.address%}!${NODE_IP_ADDRESS}!g" ${consul_output_dir}/consul.json

    if [ "${CONSUL_SERVER}" == "true" ]; then
        # client address bind to both node ip and local host
        CLIENT_ADDRESS="${NODE_IP_ADDRESS} 127.0.0.1"
    else
        # bind to local host for client
        CLIENT_ADDRESS="127.0.0.1"
    fi
    sed -i "s!{%client.address%}!${CLIENT_ADDRESS}!g" ${consul_output_dir}/consul.json

    update_consul_data_dir

    if [ "${CONSUL_SERVER}" == "true" ]; then
        # Server agent configuration
        sed -i "s!{%number.servers%}!${CONSUL_NUM_SERVERS}!g" ${consul_output_dir}/server.json
        update_ui_config
    fi

    CONSUL_CONFIG_DIR=${CONSUL_HOME}/consul.d
    mkdir -p ${CONSUL_CONFIG_DIR}
    cp -r ${consul_output_dir}/consul.json ${CONSUL_CONFIG_DIR}/consul.json

    if [ "${CONSUL_SERVER}" == "true" ]; then
        cp -r ${consul_output_dir}/server.json ${CONSUL_CONFIG_DIR}/server.json
    fi

    chmod 640 ${CONSUL_CONFIG_DIR}/consul.json

    if [ "${CONSUL_SERVER}" == "true" ]; then
        chmod 640 ${CONSUL_CONFIG_DIR}/server.json
    fi
}

set_head_option "$@"
check_consul_installed
set_head_address
set_node_ip_address
configure_consul

exit 0
