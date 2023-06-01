#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

export HADOOP_VERSION=3.3.1

export USER_HOME=/home/$(whoami)
export RUNTIME_PATH=$USER_HOME/runtime
mkdir -p $RUNTIME_PATH

# JDK install function
. "$ROOT_DIR"/common/scripts/jdk-install.sh

# Hadoop install function
. "$ROOT_DIR"/common/scripts/hadoop-install.sh

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

set_head_option "$@"
install_jdk
install_hadoop
clean_install_cache
