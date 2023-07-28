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

function configure_haproxy() {
    prepare_base_conf

    # TODO: configure a load balancer based on service discovery policies
    # should implement the capability of dynamic list of servers
}

set_head_option "$@"
check_haproxy_installed
set_head_address
set_node_ip_address
configure_haproxy

exit 0
