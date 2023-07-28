#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

USER_HOME=/home/$(whoami)
RUNTIME_PATH=$USER_HOME/runtime
ETCD_HOME=$RUNTIME_PATH/etcd

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function prepare_base_conf() {
    source_dir=$(dirname "${BIN_DIR}")/conf
    output_dir=/tmp/etcd/conf
    rm -rf  $output_dir
    mkdir -p $output_dir
    cp -r $source_dir/* $output_dir
}


function check_etcd_installed() {
    if ! command -v etcd &> /dev/null
    then
        echo "etcd is not installed for etcd command is not available."
        exit 1
    fi
}

function update_data_dir() {
    data_disk_dir=$(get_any_data_disk_dir)
    if [ -z "$data_disk_dir" ]; then
        data_dir="${ETCD_HOME}/data"
    else
        data_dir="$data_disk_dir/etcd/data"
    fi

    mkdir -p ${data_dir}
    sed -i "s#{%data.dir%}#${data_dir}#g" ${config_template_file}
}

function configure_etcd() {
    prepare_base_conf
    cd $output_dir

    ETC_LOG_DIR=${ETCD_HOME}/logs
    mkdir -p ${ETC_LOG_DIR}

    config_template_file=${output_dir}/etcd.yaml
    sed -i "s#{%node.ip%}#${NODE_IP_ADDRESS}#g" ${config_template_file}

    NODE_NAME="server${CLOUDTIK_NODE_NUMBER}"
    sed -i "s#{%node.name%}#${NODE_NAME}#g" ${config_template_file}

    update_data_dir

    ETC_LOG_FILE=${ETC_LOG_DIR}/etcd-server.log
    sed -i "s#{%log.file%}#${ETC_LOG_FILE}#g" ${config_template_file}

    sed -i "s#{%initial.cluster.token%}#${ETCD_CLUSTER_NAME}#g" ${config_template_file}

    ETCD_CONFIG_DIR=${ETCD_HOME}/conf
    mkdir -p ${ETCD_CONFIG_DIR}
    cp -r ${config_template_file} ${ETCD_CONFIG_DIR}/etcd.yaml
}

set_head_option "$@"

if [ "$IS_HEAD_NODE" == "false" ]; then
    check_etcd_installed
    set_head_address
    set_node_ip_address
    configure_etcd
fi

exit 0
