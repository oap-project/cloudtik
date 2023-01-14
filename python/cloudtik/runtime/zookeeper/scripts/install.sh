#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

IS_HEAD_NODE=false

while true
do
    case "$1" in
    -h|--head)
        IS_HEAD_NODE=true
        ;;
    --)
        shift
        break
        ;;
    esac
    shift
done

export ZOOKEEPER_VERSION=3.7.1

export USER_HOME=/home/$(whoami)
export RUNTIME_PATH=$USER_HOME/runtime
mkdir -p $RUNTIME_PATH

# JDK install function
. "$ROOT_DIR"/common/scripts/jdk-install.sh

function install_zookeeper() {
    # install zookeeper
    export ZOOKEEPER_HOME=$RUNTIME_PATH/zookeeper

    if [ ! -d "${ZOOKEEPER_HOME}" ]; then
      (cd $RUNTIME_PATH && wget -q --show-progress https://downloads.apache.org/zookeeper/zookeeper-${ZOOKEEPER_VERSION}/apache-zookeeper-${ZOOKEEPER_VERSION}-bin.tar.gz -O zookeeper.tar.gz && \
          mkdir -p "$ZOOKEEPER_HOME" && \
          tar --extract --file zookeeper.tar.gz --directory "$ZOOKEEPER_HOME" --strip-components 1 --no-same-owner && \
          rm zookeeper.tar.gz)
        echo "export ZOOKEEPER_HOME=$ZOOKEEPER_HOME">> ${USER_HOME}/.bashrc
        echo "export PATH=\$ZOOKEEPER_HOME/bin:\$PATH" >> ${USER_HOME}/.bashrc
    fi
}

install_jdk
install_zookeeper
