#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

USER_HOME=/home/$(whoami)
RUNTIME_PATH=$USER_HOME/runtime
HAPROXY_HOME=$RUNTIME_PATH/haproxy

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function prepare_base_conf() {
    source_dir=$(dirname "${BIN_DIR}")/conf
    output_dir=/tmp/haproxy/conf
    rm -rf  $output_dir
    mkdir -p $output_dir
    cp -r $source_dir/* $output_dir
}

function check_haproxy_installed() {
    if ! command -v haproxy &> /dev/null
    then
        echo "HAProxy is not installed for haproxy command is not available."
        exit 1
    fi
}

function configure_haproxy_with_dns_consul() {
    # configure a load balancer based on Consul DNS interface
    local config_template_file=${output_dir}/haproxy-dns-consul.cfg

    # Consul DNS interface based service discovery
    sed -i "s#{%backend.service.name%}#${HAPROXY_BACKEND_SERVICE_NAME}#g" ${config_template_file}
    sed -i "s#{%backend.max.servers%}#${HAPROXY_BACKEND_MAX_SERVERS}#g" ${config_template_file}

    if [ -z "${HAPROXY_BACKEND_SERVICE_TAG}" ]; then
        # if no tag specified, use the protocol of tcp
        BACKEND_SERVICE_TAG=tcp
    else
        BACKEND_SERVICE_TAG==${HAPROXY_BACKEND_SERVICE_TAG}
    fi
    sed -i "s#{%backend.service.tag%}#${BACKEND_SERVICE_TAG}#g" ${config_template_file}
    cp ${config_template_file} ${HAPROXY_CONFIG_DIR}/haproxy.cfg
}

function configure_haproxy_with_static() {
    # configure a load balancer with static address
    local config_template_file=${output_dir}/haproxy-static.cfg
    # python configure script will write the list of static servers
    cp ${config_template_file} ${HAPROXY_CONFIG_DIR}/haproxy.cfg
}

function configure_haproxy() {
    prepare_base_conf

    ETC_DEFAULT=/etc/default
    sudo mkdir -p ${ETC_DEFAULT}

    sed -i "s#{%haproxy.home%}#${HAPROXY_HOME}#g" ${output_dir}/haproxy
    sudo cp ${output_dir}/haproxy ${ETC_DEFAULT}/haproxy

    HAPROXY_CONFIG_DIR=${HAPROXY_HOME}/conf
    mkdir -p ${HAPROXY_CONFIG_DIR}

    # TODO: to support user specified external IP address
    sed -i "s#{%frontend.ip%}#${NODE_IP_ADDRESS}#g" `grep "{%frontend.ip%}" -rl ${output_dir}`
    sed -i "s#{%frontend.port%}#${HAPROXY_FRONTEND_PORT}#g" `grep "{%frontend.port%}" -rl ${output_dir}`
    sed -i "s#{%frontend.protocol%}#${HAPROXY_FRONTEND_PROTOCOL}#g" `grep "{%frontend.protocol%}" -rl ${output_dir}`

    if [ "${HAPROXY_CONFIG_MODE}" == "dns" ]; then
        configure_haproxy_with_dns_consul
    elif [ "${HAPROXY_CONFIG_MODE}" == "static" ]; then
        configure_haproxy_with_static
    else
        echo "WARNING: Unsupported configure mode: ${HAPROXY_CONFIG_MODE}"
    fi
}

set_head_option "$@"
check_haproxy_installed
set_head_address
set_node_ip_address
configure_haproxy

exit 0
