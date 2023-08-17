#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

export COREDNS_VERSION=1.11.1

export USER_HOME=/home/$(whoami)
export RUNTIME_PATH=$USER_HOME/runtime
export COREDNS_HOME=$RUNTIME_PATH/coredns


# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh


function install_coredns() {
    if [ ! -f "${COREDNS_HOME}/coredns" ]; then
        deb_arch=$(get_deb_arch)
        mkdir -p $RUNTIME_PATH
        (cd $RUNTIME_PATH && wget -q --show-progress \
          https://github.com/coredns/coredns/releases/download/v${COREDNS_VERSION}/coredns_${COREDNS_VERSION}_linux_${deb_arch}.tgz -O coredns.tgz && \
          mkdir -p "${COREDNS_HOME}" && \
          tar --extract --file coredns.tgz --directory "${COREDNS_HOME}" --no-same-owner && \
          rm coredns.tgz)
    fi
}

set_head_option "$@"
install_coredns
clean_install_cache
