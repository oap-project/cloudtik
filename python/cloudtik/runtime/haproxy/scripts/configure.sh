#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

USER_HOME=/home/$(whoami)

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

function configure_haproxy_with_consul_dns() {
    # configure a load balancer based on Consul DNS interface
    local config_template_file=${haproxy_output_dir}/haproxy-consul-dns.cfg
    # TODO: to support user specified external IP address
    sed -i "s#{%frontend.ip%}#${NODE_IP_ADDRESS}#g" ${config_template_file}
    sed -i "s#{%frontend.port%}#${HAPROXY_FRONTEND_PORT}#g" ${config_template_file}
    sed -i "s#{%frontend.protocol%}#${HAPROXY_FRONTEND_PROTOCOL}#g" ${config_template_file}

    # Consul DNS interface based service discovery
    sed -i "s#{%backend.service.name%}#${HAPROXY_BACKEND_SERVICE_NAME}#g" ${config_template_file}
    sed -i "s#{%backend.service.max.servers%}#${HAPROXY_BACKEND_SERVCE_MAX_SERVERS}#g" ${config_template_file}

    if [ -z "${HAPROXY_BACKEND_SERVICE_PROTOCOL}" ]; then
        BACKEND_SERVICE_PROTOCOL=${HAPROXY_FRONTEND_PROTOCOL}
    else
        BACKEND_SERVICE_PROTOCOL==${HAPROXY_BACKEND_SERVICE_PROTOCOL}
    fi
    sed -i "s#{%backend.service.protocol%}#${BACKEND_SERVICE_PROTOCOL}#g" ${config_template_file}

    HAPROXY_CONFIG_DIR=/etc/haproxy
    sudo mkdir -p ${HAPROXY_CONFIG_DIR}
    cp -r ${config_template_file} ${HAPROXY_CONFIG_DIR}/haproxy.cfg
}

function configure_haproxy() {
    prepare_base_conf
    cd $output_dir
    haproxy_output_dir=$output_dir/haproxy

    if [ "${HAPROXY_CONFIG_MODE}" == "CONSUL-DNS" ]; then
        configure_haproxy_with_consul_dns
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

}