#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

USER_HOME=/home/$(whoami)
RUNTIME_PATH=$USER_HOME/runtime

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function prepare_base_conf() {
    source_dir=$(cd $(dirname ${BASH_SOURCE[0]})/..;pwd)/conf
    output_dir=/tmp/zookeeper/conf
    rm -rf  $output_dir
    mkdir -p $output_dir
    cp -r $source_dir/* $output_dir
}

function check_zookeeper_installed() {
    if [ ! -n "${ZOOKEEPER_HOME}" ]; then
        echo "ZOOKEEPER_HOME environment variable is not set."
        exit 1
    fi
}

function update_zookeeper_data_disks_config() {
    zookeeper_data_dir=""
    if [ -d "/mnt/cloudtik" ]; then
        for data_disk in /mnt/cloudtik/*; do
            [ -d "$data_disk" ] || continue
            if [ -z "$zookeeper_data_dir" ]; then
                zookeeper_data_dir=$data_disk/zookeeper/data
            fi
        done
    fi

    # if no disks mounted on /mnt/cloudtik
    if [ -z "$zookeeper_data_dir" ]; then
        zookeeper_data_dir="${RUNTIME_PATH}/shared/zookeeper/data"
    fi

    mkdir -p $zookeeper_data_dir
    sed -i "s!{%zookeeper.dataDir%}!${zookeeper_data_dir}!g" $output_dir/zookeeper/zoo.cfg
}

function update_myid() {
    # Configure my id file
    if [ ! -n "${CLOUDTIK_NODE_SEQ_ID}" ]; then
        echo "No node sequence id allocated for current node!"
        exit 1
    fi

    sed -i "s!{%zookeeper.myid%}!${CLOUDTIK_NODE_SEQ_ID}!g" $output_dir/zookeeper/myid
}

function configure_zookeeper() {
    prepare_base_conf
    mkdir -p ${ZOOKEEPER_HOME}/logs
    cd $output_dir

    update_zookeeper_data_disks_config
    # Zookeeper server ensemble will be updated in up-level of configure
    update_myid

    cp -r ${output_dir}/zookeeper/zoo.cfg  ${ZOOKEEPER_HOME}/conf/zoo.cfg
    cp -r ${output_dir}/zookeeper/myid  $zookeeper_data_dir/myid
}

set_head_option "$@"

if [ $IS_HEAD_NODE == "false" ]; then
    # Zookeeper doesn't run on head node
    check_zookeeper_installed
    set_head_address
    configure_zookeeper
fi

exit 0
