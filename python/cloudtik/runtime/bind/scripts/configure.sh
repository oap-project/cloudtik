#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

USER_HOME=/home/$(whoami)
RUNTIME_PATH=$USER_HOME/runtime
BIND_HOME=$RUNTIME_PATH/bind

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function prepare_base_conf() {
    local source_dir=$(dirname "${BIN_DIR}")/conf
    output_dir=/tmp/bind/conf
    rm -rf  $output_dir
    mkdir -p $output_dir
    cp -r $source_dir/* $output_dir
}

function check_bind_installed() {
    if ! command -v named &> /dev/null
    then
        echo "Bind is not installed for named command is not available."
        exit 1
    fi
}

function configure_data_dir() {
    local data_disk_dir=$(get_first_data_disk_dir)
    if [ -z "$data_disk_dir" ]; then
        data_dir="${BIND_HOME}/data"
    else
        data_dir="$data_disk_dir/bind/data"
    fi

    mkdir -p ${data_dir}
    sed -i "s!{%data.dir%}!${data_dir}!g" ${output_dir}/named.conf.options
}

function configure_bind() {
    prepare_base_conf

    ETC_DEFAULT=/etc/default
    sudo mkdir -p ${ETC_DEFAULT}

    sed -i "s#{%bind.home%}#${BIND_HOME}#g" ${output_dir}/named
    sudo cp ${output_dir}/named ${ETC_DEFAULT}/named

    BIND_CONF_DIR=${BIND_HOME}/conf
    mkdir -p ${BIND_CONF_DIR}

    config_template_file=${output_dir}/named.conf

    sed -i "s#{%bind.home%}#${BIND_HOME}#g" ${config_template_file}

    sed -i "s#{%listen.address%}#${NODE_IP_ADDRESS}#g" ${output_dir}/named.conf.options
    sed -i "s#{%listen.port%}#${BIND_SERVICE_PORT}#g" ${output_dir}/named.conf.options

    configure_data_dir

    # generate additional name server records for specific (service discovery) domain
    if [ "${BIND_CONSUL_RESOLVE}" == "true" ]; then
        # TODO: handle consul port other than default
        cp ${output_dir}/named.conf.consul ${BIND_CONF_DIR}/named.conf.consul
        echo "include \"${BIND_HOME}/conf/named.conf.consul\";" >> ${config_template_file}
    fi

    cp ${output_dir}/named.conf.options ${BIND_CONF_DIR}/named.conf.options
    cp ${config_template_file} ${BIND_CONF_DIR}/named.conf
}

set_head_option "$@"
check_bind_installed
set_node_ip_address
configure_bind

exit 0
