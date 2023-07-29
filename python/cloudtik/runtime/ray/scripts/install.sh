#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

export USER_HOME=/home/$(whoami)
export RUNTIME_PATH=$USER_HOME/runtime

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function install_ray() {
    # install Ray
    pip --no-cache-dir -qq install ray[default,air,rllib]==2.3.0 "pydantic<2"
}

set_head_option "$@"
install_ray
clean_install_cache
