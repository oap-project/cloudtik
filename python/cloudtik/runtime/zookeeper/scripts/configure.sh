#!/bin/bash

args=$(getopt -a -o h::p: -l head::,node_ip_address::,head_address:: -- "$@")
eval set -- "${args}"

IS_HEAD_NODE=false
USER_HOME=/home/$(whoami)
RUNTIME_PATH=$USER_HOME/runtime

while true
do
    case "$1" in
    --head)
        IS_HEAD_NODE=true
        ;;
    --node_ip_address)
        NODE_IP_ADDRESS=$2
        shift
        ;;
    -h|--head_address)
        HEAD_ADDRESS=$2
        shift
        ;;
    --)
        shift
        break
        ;;
    esac
    shift
done

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

function set_head_address() {
    if [ $IS_HEAD_NODE == "true" ]; then
        if [ ! -n "${NODE_IP_ADDRESS}" ]; then
            HEAD_ADDRESS=$(hostname -I | awk '{print $1}')
        else
            HEAD_ADDRESS=${NODE_IP_ADDRESS}
        fi
    else
        if [ ! -n "${HEAD_ADDRESS}" ]; then
            # Error: no head address passed
            echo "Error: head ip address should be passed."
            exit 1
        fi
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
        zookeeper_data_dir="${RUNTIME_PATH}/shared/data/zookeeper"
    fi

    mkdir -p $zookeeper_data_dir
    sed -i "s!{%zookeeper.dataDir%}!${zookeeper_data_dir}!g" $output_dir/zookeeper/zoo.cfg
}

function update_myid() {
    # Configure my id file
    if [ ! -n "${CLOUDTIK_NODE_NUMBER}" ]; then
        echo "No node number allocated for current node!"
        exit 1
    fi

    sed -i "s!{%zookeeper.myid%}!${CLOUDTIK_NODE_NUMBER}!g" $output_dir/zookeeper/myid
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

if [ $IS_HEAD_NODE == "false" ];then
    # Zookeeper doesn't run on head node
    check_zookeeper_installed
    set_head_address
    configure_zookeeper
fi

exit 0
