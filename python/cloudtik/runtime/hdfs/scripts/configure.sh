#!/bin/bash

args=$(getopt -a -o h::p: -l head::,node_ip_address::,head_address:: -- "$@")
eval set -- "${args}"

IS_HEAD_NODE=false
USER_HOME=/home/$(whoami)

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
    output_dir=/tmp/hdfs/conf
    rm -rf  $output_dir
    mkdir -p $output_dir
    cp -r $source_dir/* $output_dir
}

function check_hadoop_installed() {
    if [ ! -n "${HADOOP_HOME}" ]; then
        echo "HADOOP_HOME environment variable is not set."
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

function update_hdfs_data_disks_config() {
    hdfs_nn_dirs="${HADOOP_HOME}/data/dfs/nn"
    hdfs_dn_dirs=""
    if [ -d "/mnt/cloudtik" ]; then
        for data_disk in /mnt/cloudtik/*; do
            [ -d "$data_disk" ] || continue
            if [ -z "$hdfs_dn_dirs" ]; then
                hdfs_dn_dirs=$data_disk/dfs/dn
            else
                hdfs_dn_dirs="$hdfs_dn_dirs,$data_disk/dfs/dn"
            fi
        done
    fi

    # if no disks mounted on /mnt/cloudtik
    if [ -z "$hdfs_dn_dirs" ]; then
        hdfs_dn_dirs="${HADOOP_HOME}/data/dfs/dn"
    fi
    sed -i "s!{%dfs.namenode.name.dir%}!${hdfs_nn_dirs}!g" `grep "{%dfs.namenode.name.dir%}" -rl ./`
    sed -i "s!{%dfs.datanode.data.dir%}!${hdfs_dn_dirs}!g" `grep "{%dfs.datanode.data.dir%}" -rl ./`
}

function configure_hdfs() {
    prepare_base_conf

    cd $output_dir
    sed -i "s/HEAD_ADDRESS/${HEAD_ADDRESS}/g" `grep "HEAD_ADDRESS" -rl ./`

    update_hdfs_data_disks_config
    cp -r ${output_dir}/hadoop/core-site.xml  ${HADOOP_HOME}/etc/hadoop/
    cp -r ${output_dir}/hadoop/hdfs-site.xml  ${HADOOP_HOME}/etc/hadoop/

    if [ $IS_HEAD_NODE == "true" ];then
        # Stop namenode in case it was running left from last try
        ${HADOOP_HOME}/bin/hdfs --daemon stop namenode > /dev/null 2>&1
        # Format hdfs once
        ${HADOOP_HOME}/bin/hdfs --loglevel WARN namenode -format -force
    fi
}

check_hadoop_installed
set_head_address
configure_hdfs

exit 0
