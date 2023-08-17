#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

export PROMETHEUS_VERSION=2.45.0

export USER_HOME=/home/$(whoami)
export RUNTIME_PATH=$USER_HOME/runtime
export PROMETHEUS_HOME=$RUNTIME_PATH/prometheus

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function install_prometheus() {
    if [ ! -f "${PROMETHEUS_HOME}/prometheus" ]; then
        deb_arch=$(get_deb_arch)
        mkdir -p $RUNTIME_PATH
        (cd $RUNTIME_PATH && wget -q --show-progress https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-${deb_arch}.tar.gz -O prometheus.tar.gz && \
          mkdir -p "$PROMETHEUS_HOME" && \
          tar --extract --file prometheus.tar.gz --directory "$PROMETHEUS_HOME" --strip-components 1 --no-same-owner && \
          rm prometheus.tar.gz)
        echo "export PROMETHEUS_HOME=$PROMETHEUS_HOME">> ${USER_HOME}/.bashrc
    fi
}

set_head_option "$@"
install_prometheus
clean_install_cache
