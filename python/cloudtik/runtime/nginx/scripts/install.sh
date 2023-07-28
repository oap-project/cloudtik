#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function install_nginx() {
    if ! command -v nginx &> /dev/null
    then
        deb_arch=$(get_deb_arch)
        wget -O - -q http://nginx.org/keys/nginx_signing.key | sudo apt-key add - > /dev/null
        echo "deb [arch=${deb_arch}] http://nginx.org/packages/mainline/ubuntu/ $(lsb_release -cs) nginx" | sudo tee /etc/apt/sources.list.d/nginx.list > /dev/null
        sudo apt-get -qq update -y > /dev/null
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -qq nginx -y > /dev/null
        sudo rm -f /etc/apt/sources.list.d/nginx.list
    fi
}

set_head_option "$@"
install_nginx
clean_install_cache
