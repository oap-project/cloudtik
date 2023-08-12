#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

USER_HOME=/home/$(whoami)
RUNTIME_PATH=$USER_HOME/runtime
NGINX_HOME=$RUNTIME_PATH/nginx

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function prepare_base_conf() {
    source_dir=$(dirname "${BIN_DIR}")/conf
    output_dir=/tmp/nginx/conf
    rm -rf  $output_dir
    mkdir -p $output_dir
    cp -r $source_dir/* $output_dir
}


function check_nginx_installed() {
    if ! command -v nginx &> /dev/null
    then
        echo "NGINX is not installed for nginx command is not available."
        exit 1
    fi
}

function configure_web() {
    cat ${output_dir}/nginx-web-base.conf >> ${nginx_config_file}
    mkdir -p ${NGINX_CONFIG_DIR}/web
    cp ${output_dir}/nginx-web.conf ${NGINX_CONFIG_DIR}/web/web.conf
}

function configure_dns_backend() {
    # configure a load balancer based on Consul DNS interface
    local config_template_file=${output_dir}/nginx-load-balancer-dns.conf

    # Consul DNS interface based service discovery
    sed -i "s#{%backend.service.dns.name%}#${NGINX_BACKEND_SERVICE_DNS_NAME}#g" ${config_template_file}
    sed -i "s#{%backend.service.port%}#${NGINX_BACKEND_SERVICE_PORT}#g" ${config_template_file}

    cat ${config_template_file} >> ${nginx_config_file}
}

function configure_static_backend() {
    # python configure script will write upstream block
    cat ${output_dir}/nginx-load-balancer-static.conf >> ${nginx_config_file}
    mkdir -p ${NGINX_CONFIG_DIR}/upstreams
}

function configure_dynamic_backend() {
    # python discovery script will write upstream block and do reload if needed
    cat ${output_dir}/nginx-load-balancer-static.conf >> ${nginx_config_file}
    mkdir -p ${NGINX_CONFIG_DIR}/upstreams
}

function configure_load_balancer() {
    if [ "${NGINX_CONFIG_MODE}" == "dns" ]; then
        configure_dns_backend
    elif [ "${NGINX_CONFIG_MODE}" == "static" ]; then
        configure_static_backend
    elif [ "${NGINX_CONFIG_MODE}" == "dynamic" ]; then
        configure_dynamic_backend
    else
        echo "WARNING: Unsupported configure mode for load balancer: ${NGINX_CONFIG_MODE}"
    fi
}

function configure_api_gateway_dns() {
    # discovery services will discovery the api backends
    :
}

function configure_api_gateway_dynamic() {
    # discovery services will discovery the api backends and upstream servers
    mkdir -p ${NGINX_CONFIG_DIR}/upstreams
}

function configure_api_gateway() {
    cat ${output_dir}/nginx-api-gateway-base.conf >> ${nginx_config_file}
    cp ${output_dir}/nginx-api-gateway.conf ${NGINX_CONFIG_DIR}/api-gateway.conf
    cp ${output_dir}/nginx-api-gateway-json-errors.conf ${NGINX_CONFIG_DIR}/api-gateway-json-errors.conf
    mkdir -p ${NGINX_CONFIG_DIR}/api-gateway

    if [ "${NGINX_CONFIG_MODE}" == "dns" ]; then
        configure_api_gateway_dns
    elif [ "${NGINX_CONFIG_MODE}" == "dynamic" ]; then
        configure_api_gateway_dynamic
    else
        echo "WARNING: Unsupported configure mode for API gateway: ${NGINX_CONFIG_MODE}"
    fi
}

function configure_nginx() {
    prepare_base_conf
    nginx_config_file=${output_dir}/nginx.conf

    ETC_DEFAULT=/etc/default
    sudo mkdir -p ${ETC_DEFAULT}

    sed -i "s#{%nginx.home%}#${NGINX_HOME}#g" ${output_dir}/nginx
    sudo cp ${output_dir}/nginx ${ETC_DEFAULT}/nginx

    NGINX_CONFIG_DIR=${NGINX_HOME}/conf
    mkdir -p ${NGINX_CONFIG_DIR}

    sed -i "s#{%server.listen.ip%}#${NODE_IP_ADDRESS}#g" `grep "{%server.listen.ip%}" -rl ${output_dir}`
    sed -i "s#{%server.listen.port%}#${NGINX_LISTEN_PORT}#g" `grep "{%server.listen.port%}" -rl ${output_dir}`

    if [ "${NGINX_APP_MODE}" == "web" ]; then
        configure_web
    elif [ "${NGINX_APP_MODE}" == "load-balancer" ]; then
        configure_load_balancer
    elif [ "${NGINX_APP_MODE}" == "api-gateway" ]; then
        configure_api_gateway
    else
        echo "WARNING: Unknown application mode: ${NGINX_APP_MODE}"
    fi

    cp ${nginx_config_file} ${NGINX_CONFIG_DIR}/nginx.conf
}

set_head_option "$@"
check_nginx_installed
set_head_address
set_node_ip_address
configure_nginx

exit 0
