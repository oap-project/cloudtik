#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

export CONSUL_VERSION=1.16.0

export USER_HOME=/home/$(whoami)
export RUNTIME_PATH=$USER_HOME/runtime

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function install_consul() {
    # install consul
    if ! command -v consul &> /dev/null
    then
        deb_arch=$(get_deb_arch)
        wget -O - -q https://apt.releases.hashicorp.com/gpg | sudo apt-key add -
        echo "deb [arch=${deb_arch}] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list > /dev/null
        sudo apt-get -qq update -y > /dev/null
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -qq consul=${CONSUL_VERSION}-* -y > /dev/null
        sudo rm -f /etc/apt/sources.list.d/hashicorp.list
    fi
}

set_head_option "$@"
install_consul
clean_install_cache
