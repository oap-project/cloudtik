#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

export NODE_EXPORTER_VERSION=1.6.1

export USER_HOME=/home/$(whoami)
export RUNTIME_PATH=$USER_HOME/runtime

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function install_node_exporter() {
    if [ ! -f "${NODE_EXPORTER_HOME}/node_exporter" ]; then
        export NODE_EXPORTER_HOME=$RUNTIME_PATH/node_exporter
        deb_arch=$(get_deb_arch)
        mkdir -p $RUNTIME_PATH
        (cd $RUNTIME_PATH && wget -q --show-progress https://github.com/prometheus/node_exporter/releases/download/v${NODE_EXPORTER_VERSION}/node_exporter-${NODE_EXPORTER_VERSION}.linux-${deb_arch}.tar.gz -O node_exporter.tar.gz && \
          mkdir -p "$NODE_EXPORTER_HOME" && \
          tar --extract --file node_exporter.tar.gz --directory "$NODE_EXPORTER_HOME" --strip-components 1 --no-same-owner && \
          rm node_exporter.tar.gz)
        echo "export NODE_EXPORTER_HOME=$NODE_EXPORTER_HOME">> ${USER_HOME}/.bashrc
    fi
}

set_head_option "$@"
install_node_exporter
clean_install_cache
