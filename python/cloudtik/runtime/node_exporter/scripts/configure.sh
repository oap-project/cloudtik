#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

USER_HOME=/home/$(whoami)
RUNTIME_PATH=$USER_HOME/runtime
NODE_EXPORTER_HOME=$RUNTIME_PATH/node_exporter

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function check_node_exporter_installed() {
    if [ ! -f "${NODE_EXPORTER_HOME}/node_exporter" ]; then
        echo "Node Exporter is not installed for node_exporter command is not available."
        exit 1
    fi
}

function configure_node_exporter() {
    mkdir -p ${NODE_EXPORTER_HOME}/logs
}

set_head_option "$@"
check_node_exporter_installed
configure_node_exporter

exit 0
