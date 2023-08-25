#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

APISIX_VERSION=3.4


# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function install_apisix() {
    if ! command -v apisix &> /dev/null
    then
        wget -q -O - https://openresty.org/package/pubkey.gpg | sudo apt-key add - && \
        wget -q -O - http://repos.apiseven.com/pubkey.gpg | sudo apt-key add - && \
        echo "deb http://openresty.org/package/debian bullseye openresty" \
          | sudo tee /etc/apt/sources.list.d/openresty.list >/dev/null && \
        echo "deb http://repos.apiseven.com/packages/debian bullseye main" \
          | sudo tee /etc/apt/sources.list.d/apisix.list >/dev/null

        sudo apt-get -qq update -y > /dev/null && \
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -qq -y \
          apisix=${APISIX_VERSION}.\* > /dev/null && \
        sudo rm -f /etc/apt/sources.list.d/openresty.list && \
        sudo rm -f /etc/apt/sources.list.d/apisix.list && \
    fi
}

set_head_option "$@"
install_apisix
clean_install_cache
