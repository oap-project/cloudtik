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
    if [ "${CONSUL_SERVER}" == "true" ]; then
        # join list for servers, we use the head
        if [ $IS_HEAD_NODE == "true" ]; then
            # for head, use its own address
            JOIN_LIST="\"${NODE_IP_ADDRESS}\""
        else
            JOIN_LIST="\"${HEAD_ADDRESS}\""
        fi
    else
        # for client mode cluster, CONSUL_JOIN_LIST will be set by runtime
        if [ -z  "${CONSUL_JOIN_LIST}" ]; then
            echo "WARNING: CONSUL_JOIN_LIST is empty. It should be set."
        fi

        JOIN_LIST=${CONSUL_JOIN_LIST}
    fi
    sed -i "s#{%join.list%}#${JOIN_LIST}#g" ${consul_output_dir}/consul.hcl
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
    if [ "$IS_HEAD_NODE" == "true" ]; then
        UI_ENABLED=true
    else
        UI_ENABLED=false
    fi
    sed -i "s!{%ui.config.enabled%}!${UI_ENABLED}!g" ${consul_output_dir}/server.hcl
}

function configure_consul() {
    prepare_base_conf
    mkdir -p ${CONSUL_HOME}/logs
    cd $output_dir
    consul_output_dir=$output_dir/consul

    # General agent configuration
    sed -i "s!{%bind.address%}!${NODE_IP_ADDRESS}!g" ${consul_output_dir}/consul.hcl

    # client address bind to both node ip and local host
    CLIENT_ADDRESS="${NODE_IP_ADDRESS} 127.0.0.1"
    sed -i "s!{%client.address%}!${CLIENT_ADDRESS}!g" ${consul_output_dir}/consul.hcl
    update_join_list
    update_consul_data_dir

    if [ "${CONSUL_SERVER}" == "true" ]; then
        # Server agent configuration
        sed -i "s!{%server.number%}!${CONSUL_NUM_SERVERS}!g" ${consul_output_dir}/server.hcl
        update_ui_config
    fi

    CONSUL_CONFIG_DIR=${CONSUL_HOME}/consul.d
    CONSUL_CONFIG_DIR_INSTALLED=/etc/consul.d
    mkdir -p ${CONSUL_CONFIG_DIR}
    sudo cp -r ${consul_output_dir}/consul.hcl ${CONSUL_CONFIG_DIR}/consul.hcl

    if [ "${CONSUL_SERVER}" == "true" ]; then
        sudo cp -r ${consul_output_dir}/server.hcl ${CONSUL_CONFIG_DIR}/server.hcl
    fi

    sudo chmod 640 ${CONSUL_CONFIG_DIR}/consul.hcl

    if [ "${CONSUL_SERVER}" == "true" ]; then
        sudo chmod 640 ${CONSUL_CONFIG_DIR}/server.hcl
    fi
}

set_head_option "$@"
check_consul_installed
set_head_address
set_node_ip_address
configure_consul

exit 0
