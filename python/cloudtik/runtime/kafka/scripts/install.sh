#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h::p: -l head:: -- "$@")
eval set -- "${args}"

IS_HEAD_NODE=false

while true
do
    case "$1" in
    --head)
        IS_HEAD_NODE=true
        ;;
    --)
        shift
        break
        ;;
    esac
    shift
done

export KAFKA_VERSION=3.1.0
export KAFKA_SCALA_VERSION=2.13

export USER_HOME=/home/$(whoami)
export RUNTIME_PATH=$USER_HOME/runtime
mkdir -p $RUNTIME_PATH

# JDK install function
. "$ROOT_DIR"/common/scripts/jdk-install.sh

function install_kafka() {
    # install kafka
    export KAFKA_HOME=$RUNTIME_PATH/kafka

    if [ ! -d "${KAFKA_HOME}" ]; then
      (cd $RUNTIME_PATH && wget -q --show-progress https://downloads.apache.org/kafka/${KAFKA_VERSION}/kafka_${KAFKA_SCALA_VERSION}-${KAFKA_VERSION}.tgz -O kafka-${KAFKA_VERSION}.tgz && \
          tar -zxf kafka-${KAFKA_VERSION}.tgz && \
          mv kafka_${KAFKA_SCALA_VERSION}-${KAFKA_VERSION} kafka && \
          rm kafka-${KAFKA_VERSION}.tgz)
        echo "export KAFKA_HOME=$KAFKA_HOME">> ${USER_HOME}/.bashrc
        echo "export PATH=\$KAFKA_HOME/bin:\$PATH" >> ${USER_HOME}/.bashrc
    fi
}

install_jdk
install_kafka
