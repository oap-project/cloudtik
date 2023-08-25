#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

USER_HOME=/home/$(whoami)
RUNTIME_PATH=$USER_HOME/runtime
APISIX_HOME=$RUNTIME_PATH/apisix

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function prepare_base_conf() {
    local source_dir=$(dirname "${BIN_DIR}")/conf
    output_dir=/tmp/apisix/conf
    rm -rf  $output_dir
    mkdir -p $output_dir
    cp -r $source_dir/* $output_dir
}

function check_apisix_installed() {
    if ! command -v apisix &> /dev/null
    then
        echo "APISIX is not installed for apisix command is not available."
        exit 1
    fi
}

function configure_apisix() {
    prepare_base_conf

    APISIX_CONF_DIR=${APISIX_HOME}/conf
    mkdir -p ${APISIX_CONF_DIR}

    config_template_file=${output_dir}/config.yaml
    sed -i "s#{%listen.ip%}#${NODE_IP_ADDRESS}#g" ${config_template_file}
    sed -i "s#{%listen.port%}#${APISIX_SERVICE_PORT}#g" ${config_template_file}
    sed -i "s#{%cluster.name%}#${CLOUDTIK_CLUSTER}#g" ${config_template_file}

    cp ${config_template_file} ${APISIX_CONF_INCLUDE_DIR}/config.yaml
}

set_head_option "$@"
check_apisix_installed
set_node_ip_address
configure_apisix

exit 0
