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

export TRINO_VERSION=382

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

function install_tools() {
    which uuid > /dev/null || sudo apt-get -qq update -y; sudo apt-get -qq install uuid -y
}

function install_trino() {
    # install Trino
    export TRINO_HOME=$RUNTIME_PATH/trino

    if [ ! -d "${TRINO_HOME}" ]; then
        (cd $RUNTIME_PATH && wget -q --show-progress https://repo1.maven.org/maven2/io/trino/trino-server/${TRINO_VERSION}/trino-server-${TRINO_VERSION}.tar.gz && \
            tar -zxf trino-server-${TRINO_VERSION}.tar.gz && \
            mv trino-server-${TRINO_VERSION} trino && \
            rm trino-server-${TRINO_VERSION}.tar.gz)
            echo "export TRINO_HOME=$TRINO_HOME">> ${USER_HOME}/.bashrc
            echo "export PATH=\$TRINO_HOME/bin:\$PATH" >> ${USER_HOME}/.bashrc
    fi
}

install_jdk
install_tools
install_trino
