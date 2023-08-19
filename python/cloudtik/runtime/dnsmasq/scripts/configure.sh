#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

USER_HOME=/home/$(whoami)
RUNTIME_PATH=$USER_HOME/runtime
DNSMASQ_HOME=$RUNTIME_PATH/dnsmasq

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function prepare_base_conf() {
    local source_dir=$(dirname "${BIN_DIR}")/conf
    output_dir=/tmp/dnsmasq/conf
    rm -rf  $output_dir
    mkdir -p $output_dir
    cp -r $source_dir/* $output_dir
}

function check_dnsmasq_installed() {
    if ! command -v dnsmasq &> /dev/null
    then
        echo "dnsmasq is not installed for dnsmasq command is not available."
        exit 1
    fi
}

function configure_dnsmasq() {
    prepare_base_conf

    ETC_DEFAULT=/etc/default
    sudo mkdir -p ${ETC_DEFAULT}

    sed -i "s#{%dnsmasq.home%}#${DNSMASQ_HOME}#g" ${output_dir}/dnsmasq
    sudo cp ${output_dir}/dnsmasq ${ETC_DEFAULT}/dnsmasq

    DNSMASQ_CONF_DIR=${DNSMASQ_HOME}/conf
    DNSMASQ_CONF_INCLUDE_DIR=${DNSMASQ_CONF_DIR}/conf.d
    mkdir -p ${DNSMASQ_CONF_INCLUDE_DIR}

    config_template_file=${output_dir}/dnsmasq.conf
    sed -i "s#{%listen.address%}#${NODE_IP_ADDRESS}#g" ${config_template_file}
    sed -i "s#{%listen.port%}#${DNSMASQ_SERVICE_PORT}#g" ${config_template_file}

    # dnsmasq will use /etc/resolv.conf for upstream.
    # TODO: if we want to use this DNS server as the system default, we need:
    # 1. copy the /etc/resolv.conf to a backup file if backup file not exists
    # 2. direct dnsmasq to use the backup copy as upstream
    # 3. modify /etc/resolve.conf to use dnsmasq as resolver

    SYSTEM_RESOLV_CONF="/etc/resolv.conf"
    ORIGIN_RESOLV_CONF="${DNSMASQ_HOME}/conf/resolv.conf"

    # backup the system resolv conf only once
    if [ ! -f "${ORIGIN_RESOLV_CONF}"]; then
        cp ${SYSTEM_RESOLV_CONF} ${ORIGIN_RESOLV_CONF}
    fi

    if [ "${DNSMASQ_DEFAULT_RESOLVER}" == "true" ]; then
        UPSTREAM_RESOLV_CONF=${ORIGIN_RESOLV_CONF}
    else
        UPSTREAM_RESOLV_CONF=${SYSTEM_RESOLV_CONF}
    fi

    sed -i "s#{%upstream.resolv.conf%}#${UPSTREAM_RESOLV_CONF}#g" \
      ${config_template_file}
    # python configure script will update the system resolv.conf

    cp ${config_template_file} ${DNSMASQ_CONF_INCLUDE_DIR}/dnsmasq.conf

    # generate additional name server records for specific (service discovery) domain
    if [ "${DNSMASQ_CONSUL_RESOLVE}" == "true" ]; then
        # TODO: handle consul port other than default
        cp ${output_dir}/consul ${DNSMASQ_CONF_INCLUDE_DIR}/consul
    fi
}

set_head_option "$@"
check_dnsmasq_installed
set_node_ip_address
configure_dnsmasq

exit 0
