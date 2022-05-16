#!/bin/bash

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

export ZOOKEEPER_VERSION=3.7.1

export USER_HOME=/home/$(whoami)
export RUNTIME_PATH=$USER_HOME/runtime
mkdir -p $RUNTIME_PATH

function install_jdk() {
    # install JDK
    export JAVA_HOME=$RUNTIME_PATH/jdk

    if [ ! -d "${JAVA_HOME}" ]; then
      (cd $RUNTIME_PATH && wget -q --show-progress https://devops.egov.org.in/Downloads/jdk/jdk-8u192-linux-x64.tar.gz  && \
          gunzip jdk-8u192-linux-x64.tar.gz && \
          tar -xf jdk-8u192-linux-x64.tar && \
          rm jdk-8u192-linux-x64.tar && \
          mv jdk1.8.0_192 jdk)
        echo "export JAVA_HOME=$JAVA_HOME">> ${USER_HOME}/.bashrc
        echo "export PATH=\$JAVA_HOME/bin:\$PATH" >> ${USER_HOME}/.bashrc
    fi
}

function install_zookeeper() {
    # install zookeeper
    export ZOOKEEPER_HOME=$RUNTIME_PATH/zookeeper

    if [ ! -d "${ZOOKEEPER_HOME}" ]; then
      (cd $RUNTIME_PATH && wget -q --show-progress https://downloads.apache.org/zookeeper/zookeeper-${ZOOKEEPER_VERSION}/apache-zookeeper-${ZOOKEEPER_VERSION}-bin.tar.gz -O zookeeper-${ZOOKEEPER_VERSION}.tar.gz && \
          tar -zxf zookeeper-${ZOOKEEPER_VERSION}.tar.gz && \
          mv apache-zookeeper-${ZOOKEEPER_VERSION}-bin zookeeper && \
          rm zookeeper-${ZOOKEEPER_VERSION}.tar.gz)
        echo "export ZOOKEEPER_HOME=$ZOOKEEPER_HOME">> ${USER_HOME}/.bashrc
        echo "export PATH=\$ZOOKEEPER_HOME/bin:\$PATH" >> ${USER_HOME}/.bashrc
    fi
}

install_jdk
install_zookeeper
