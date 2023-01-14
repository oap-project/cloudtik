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

if [ $IS_HEAD_NODE != "true" ]; then
    # Do nothing for workers
    exit 0
fi

export HADOOP_VERSION=3.3.1
export HIVE_VERSION=3.1.2

export USER_HOME=/home/$(whoami)
export RUNTIME_PATH=$USER_HOME/runtime
mkdir -p $RUNTIME_PATH

# JDK install function
. "$ROOT_DIR"/common/scripts/jdk-install.sh

# Hadoop install function
. "$ROOT_DIR"/common/scripts/hadoop-install.sh

function install_mariadb() {
    sudo apt-get -qq update -y > /dev/null
    sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install -y mariadb-server > /dev/null
}

function install_hive_metastore() {
    # install hive metastore
    export METASTORE_HOME=$RUNTIME_PATH/hive-metastore

    if [ ! -d "${METASTORE_HOME}" ]; then
      (cd $RUNTIME_PATH && wget -q --show-progress https://repo1.maven.org/maven2/org/apache/hive/hive-standalone-metastore/${HIVE_VERSION}/hive-standalone-metastore-${HIVE_VERSION}-bin.tar.gz -O hive-standalone-metastore.tar.gz && \
          mkdir -p "$METASTORE_HOME" && \
          tar --extract --file hive-standalone-metastore.tar.gz --directory "$METASTORE_HOME" --strip-components 1 --no-same-owner && \
          rm hive-standalone-metastore.tar.gz)
        wget -q --show-progress https://repo1.maven.org/maven2/mysql/mysql-connector-java/5.1.38/mysql-connector-java-5.1.38.jar -P $METASTORE_HOME/lib/
        echo "export METASTORE_HOME=$METASTORE_HOME">> ${USER_HOME}/.bashrc
    fi
}


install_jdk
install_hadoop
install_mariadb
install_hive_metastore
