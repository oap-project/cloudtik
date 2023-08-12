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

function configure_dns_backend() {
    # configure a load balancer based on Consul DNS interface
    local config_template_file=${output_dir}/haproxy-dns-consul.cfg

    # Consul DNS interface based service discovery
    sed -i "s#{%backend.max.servers%}#${HAPROXY_BACKEND_MAX_SERVERS}#g" ${config_template_file}
    sed -i "s#{%backend.service.dns.name%}#${HAPROXY_BACKEND_SERVICE_DNS_NAME}#g" ${config_template_file}

    cat ${config_template_file} >> ${haproxy_config_file}
}

function configure_static_backend() {
    # configure a load balancer with static address
    local config_template_file=${output_dir}/haproxy-static.cfg
    # python configure script will write the list of static servers
    cat ${config_template_file} >> ${haproxy_config_file}
}

function configure_dynamic_backend() {
    local haproxy_template_file=${output_dir}/haproxy-template.cfg
    cp ${haproxy_config_file} ${haproxy_template_file}

    # configure a load balancer with static address
    local config_template_file=${output_dir}/haproxy-dynamic.cfg

    sed -i "s#{%backend.max.servers%}#${HAPROXY_BACKEND_MAX_SERVERS}#g" ${config_template_file}

    cat ${config_template_file} >> ${haproxy_config_file}
    # This is used as the template to generate the configuration file
    # with dynamic list of servers
    cat ${output_dir}/haproxy-static.cfg >> ${haproxy_template_file}
    cp ${haproxy_template_file} ${HAPROXY_CONFIG_DIR}/haproxy-template.cfg
}

function configure_load_balancer() {
    if [ "${HAPROXY_CONFIG_MODE}" == "dns" ]; then
        configure_dns_backend
    elif [ "${HAPROXY_CONFIG_MODE}" == "static" ]; then
        configure_static_backend
    elif [ "${HAPROXY_CONFIG_MODE}" == "dynamic" ]; then
        configure_dynamic_backend
    else
        echo "WARNING: Unsupported configure mode: ${HAPROXY_CONFIG_MODE}"
    fi
}

function configure_api_gateway() {
    # python script will use this template to generate config for API gateway backends
    cp ${haproxy_config_file} ${HAPROXY_CONFIG_DIR}/haproxy-template.cfg
}

function configure_haproxy() {
    prepare_base_conf

    haproxy_config_file=${output_dir}/haproxy.cfg

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
    sed -i "s#{%backend.balance%}#${HAPROXY_BACKEND_BALANCE}#g" `grep "{%backend.balance%}" -rl ${output_dir}`

    if [ "${HAPROXY_APP_MODE}" == "load-balancer" ]; then
        configure_load_balancer
    elif [ "${NGINX_APP_MODE}" == "api-gateway" ]; then
        configure_api_gateway
    else
        echo "WARNING: Unknown application mode: ${NGINX_APP_MODE}"
    fi

    cp ${haproxy_config_file} ${HAPROXY_CONFIG_DIR}/haproxy.cfg
}

set_head_option "$@"
check_haproxy_installed
set_head_address
set_node_ip_address
configure_haproxy

exit 0
