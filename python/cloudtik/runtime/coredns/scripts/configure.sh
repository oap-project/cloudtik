#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

USER_HOME=/home/$(whoami)
RUNTIME_PATH=$USER_HOME/runtime
COREDNS_HOME=$RUNTIME_PATH/coredns

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function prepare_base_conf() {
    local source_dir=$(dirname "${BIN_DIR}")/conf
    output_dir=/tmp/coredns/conf
    rm -rf  $output_dir
    mkdir -p $output_dir
    cp -r $source_dir/* $output_dir
}

function check_coredns_installed() {
    if [ ! -f "${COREDNS_HOME}/coredns" ]; then
        echo "CoreDNS is not installed for coredns command is not available."
        exit 1
    fi
}

function configure_coredns() {
    prepare_base_conf

    COREDNS_CONF_DIR=${COREDNS_HOME}/conf
    mkdir -p ${COREDNS_CONF_DIR}

    config_template_file=${output_dir}/Corefile

    sed -i "s#{%bind.ip%}#${NODE_IP_ADDRESS}#g" \
      `grep "{%bind.ip%}" -rl ${output_dir}`
    sed -i "s#{%bind.port%}#${COREDNS_SERVICE_PORT}#g" \
      `grep "{%bind.port%}" -rl ${output_dir}`

    # generate additional name server records for specific (service discovery) domain
    if [ "${BIND_CONSUL_RESOLVE}" == "true" ]; then
        # TODO: handle consul port other than default
        echo "import ${COREDNS_CONF_DIR}/Corefile.consul" >> ${config_template_file}
        cp ${output_dir}/Corefile.consul ${COREDNS_CONF_DIR}/Corefile.consul
    fi

    echo "import ${COREDNS_CONF_DIR}/Corefile.upstream" >> ${config_template_file}
    cp ${output_dir}/Corefile.upstream ${COREDNS_CONF_DIR}/Corefile.upstream

    cp ${config_template_file} ${COREDNS_CONF_DIR}/Corefile
}

set_head_option "$@"
check_coredns_installed
set_node_ip_address
configure_coredns

exit 0
