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

export HADOOP_VERSION=3.3.1

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

function install_hadoop() {
    # install Hadoop
    export HADOOP_HOME=$RUNTIME_PATH/hadoop

    if [ ! -d "${HADOOP_HOME}" ]; then
      (cd $RUNTIME_PATH && wget -q --show-progress http://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz -O hadoop-${HADOOP_VERSION}.tar.gz && \
          tar -zxf hadoop-${HADOOP_VERSION}.tar.gz && \
          mv hadoop-${HADOOP_VERSION} hadoop && \
          rm hadoop-${HADOOP_VERSION}.tar.gz)
      echo "export HADOOP_HOME=$HADOOP_HOME">> ${USER_HOME}/.bashrc
      echo "export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop">> ${USER_HOME}/.bashrc
      echo "export JAVA_HOME=$JAVA_HOME" >> ${HADOOP_HOME}/etc/hadoop/hadoop-env.sh
      echo "export PATH=\$HADOOP_HOME/bin:\$PATH" >> ${USER_HOME}/.bashrc
    fi
}

install_jdk
install_hadoop
