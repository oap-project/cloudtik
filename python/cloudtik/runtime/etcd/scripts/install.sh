#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

export ETCD_VERSION=3.5.9

export USER_HOME=/home/$(whoami)
export RUNTIME_PATH=$USER_HOME/runtime

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function install_etcd() {
    if ! command -v etcd &> /dev/null
    then
        export ETCD_HOME=$RUNTIME_PATH/etcd
        # We can newer version from github. For example,
        # https://github.com/etcd-io/etcd/releases/download/v3.5.9/etcd-v3.5.9-linux-amd64.tar.gz
        # sudo apt-get -qq update -y > /dev/null
        # sudo DEBIAN_FRONTEND=noninteractive apt-get install -qq etcd -y > /dev/null
        deb_arch=$(get_deb_arch)
        mkdir -p $RUNTIME_PATH
        (cd $RUNTIME_PATH && wget -q --show-progress https://github.com/etcd-io/etcd/releases/download/v${ETCD_VERSION}/etcd-v${ETCD_VERSION}-linux-${deb_arch}.tar.gz -O etcd.tar.gz && \
          mkdir -p "$ETCD_HOME" && \
          tar --extract --file etcd.tar.gz --directory "$ETCD_HOME" --strip-components 1 --no-same-owner && \
          mkdir -p "$ETCD_HOME/bin" && \
          mv "$ETCD_HOME/etcd" "$ETCD_HOME/etcdctl" "$ETCD_HOME/etcdutl" "$ETCD_HOME/bin/"
          rm etcd.tar.gz)
        echo "export ETCD_HOME=$ETCD_HOME">> ${USER_HOME}/.bashrc
        echo "export PATH=\$ETCD_HOME/bin:\$PATH" >> ${USER_HOME}/.bashrc
    fi
}

set_head_option "$@"
install_etcd
clean_install_cache
