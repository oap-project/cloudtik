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
    mkdir -p ${COREDNS_HOME}/logs

    COREDNS_CONF_DIR=${COREDNS_HOME}/conf
    mkdir -p ${COREDNS_CONF_DIR}

    config_template_file=${output_dir}/Corefile

    sed -i "s#{%bind.ip%}#${NODE_IP_ADDRESS}#g" \
      `grep "{%bind.ip%}" -rl ${output_dir}`
    sed -i "s#{%bind.port%}#${COREDNS_SERVICE_PORT}#g" \
      `grep "{%bind.port%}" -rl ${output_dir}`

    # generate additional name server records for specific (service discovery) domain
    if [ "${COREDNS_CONSUL_RESOLVE}" == "true" ]; then
        # TODO: handle consul port other than default
        echo "import ${COREDNS_CONF_DIR}/Corefile.consul" >> ${config_template_file}
        cp ${output_dir}/Corefile.consul ${COREDNS_CONF_DIR}/Corefile.consul
    fi

    SYSTEM_RESOLV_CONF="/etc/resolv.conf"
    ORIGIN_RESOLV_CONF="${COREDNS_HOME}/conf/resolv.conf"

    # backup the system resolv conf only once
    if [ ! -f "${ORIGIN_RESOLV_CONF}"]; then
        cp ${SYSTEM_RESOLV_CONF} ${ORIGIN_RESOLV_CONF}
    fi

    if [ "${COREDNS_DEFAULT_RESOLVER}" == "true" ]; then
        UPSTREAM_RESOLV_CONF=${ORIGIN_RESOLV_CONF}
    else
        UPSTREAM_RESOLV_CONF=${SYSTEM_RESOLV_CONF}
    fi

    sed -i "s#{%upstream.resolv.conf%}#${UPSTREAM_RESOLV_CONF}#g" \
      ${COREDNS_CONF_DIR}/Corefile.upstream

    echo "import ${COREDNS_CONF_DIR}/Corefile.upstream" >> ${config_template_file}
    cp ${output_dir}/Corefile.upstream ${COREDNS_CONF_DIR}/Corefile.upstream

    cp ${config_template_file} ${COREDNS_CONF_DIR}/Corefile
}

set_head_option "$@"
check_coredns_installed
set_node_ip_address
configure_coredns

exit 0
