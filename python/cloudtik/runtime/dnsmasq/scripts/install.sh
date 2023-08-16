#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function install_dnsmasq() {
    if ! command -v dnsmasq &> /dev/null
    then
        apt-get -qq update -y > /dev/null; \
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -qq  -y dnsmasq > /dev/null
    fi
}

set_head_option "$@"
install_dnsmasq
clean_install_cache
