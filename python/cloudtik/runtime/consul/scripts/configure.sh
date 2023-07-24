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
    source_dir=$(cd $(dirname ${BASH_SOURCE[0]})/..;pwd)/conf
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

function update_join_list() {
    # join list for servers, we use the head
    if [ $IS_HEAD_NODE == "true" ]; then
        # for head, use its own address
        JOIN_LIST="\"${NODE_IP_ADDRESS}\""
    else
        JOIN_LIST="\"${HEAD_ADDRESS}\""
    fi
    sed -i "s!{%join.list%}!${JOIN_LIST}!g" ${consul_output_dir}/consul.hcl
}

function update_consul_data_dir() {
    data_disk_dir=$(get_any_data_disk_dir)
    if [ -z "$data_disk_dir" ]; then
        consul_data_dir="$CONSUL_HOME/data"
    else
        consul_data_dir="$data_disk_dir/consul/data"
    fi

    mkdir -p ${consul_data_dir}
    sed -i "s!{%data.dir%}!${consul_data_dir}!g" ${consul_output_dir}/consul.hcl
}

function update_ui_config() {
    if [ $IS_HEAD_NODE == "true" ]; then
        UI_ENABLED=true
    else
        UI_ENABLED=false
    fi
    sed -i "s!{%ui.config.enabled%}!${UI_ENABLED}!g" ${consul_output_dir}/consul.hcl
}

function configure_consul() {
    prepare_base_conf
    mkdir -p ${CONSUL_HOME}/logs
    cd $output_dir
    consul_output_dir=$output_dir/consul

    sed -i "s!{%bind.address%}!${NODE_IP_ADDRESS}!g" ${consul_output_dir}/consul.hcl
    sed -i "s!{%client.address%}!${NODE_IP_ADDRESS}!g" ${consul_output_dir}/consul.hcl
    sed -i "s!{%server.nodes%}!${CONSUL_SERVERS}!g" ${consul_output_dir}/consul.hcl

    update_join_list
    update_consul_data_dir
    update_ui_config

    CONSUL_CONFIG_DIR=${CONSUL_HOME}/consul.d
    CONSUL_CONFIG_DIR_INSTALLED=/etc/consul.d
    mkdir -p ${CONSUL_CONFIG_DIR}
    if [ -d "${CONSUL_CONFIG_DIR_INSTALLED}" ]; then
        sudo cp -r /etc/consul.d/* ${CONSUL_CONFIG_DIR}/
    fi
    sudo cp -r ${consul_output_dir}/consul.hcl ${CONSUL_CONFIG_DIR}/consul.hcl
    sudo chown --recursive $(whoami):users ${CONSUL_CONFIG_DIR}
    sudo chmod 640 ${CONSUL_CONFIG_DIR}/consul.hcl
}

set_head_option "$@"
check_consul_installed
set_head_address
set_node_ip_address
configure_consul

exit 0
