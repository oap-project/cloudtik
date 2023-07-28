#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function install_etcd() {
    if ! command -v etcd &> /dev/null
    then
        # We can newer version from github. For example,
        # https://github.com/etcd-io/etcd/releases/download/v3.5.9/etcd-v3.5.9-linux-amd64.tar.gz
        sudo apt-get -qq update -y > /dev/null
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -qq etcd -y > /dev/null
    fi
}

set_head_option "$@"
install_etcd
clean_install_cache
