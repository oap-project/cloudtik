#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function install_haproxy() {
    if ! command -v haproxy &> /dev/null
    then
        sudo apt-get install -qq --no-install-recommends software-properties-common -y > /dev/null
        sudo add-apt-repository ppa:vbernat/haproxy-2.8 -y > /dev/null
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -qq haproxy=2.8.\* -y > /dev/null
    fi
}

set_head_option "$@"
install_haproxy
clean_install_cache
