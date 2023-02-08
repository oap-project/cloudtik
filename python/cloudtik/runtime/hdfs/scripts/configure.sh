#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head::,node_ip_address:,head_address: -- "$@")
eval set -- "${args}"

IS_HEAD_NODE=false
USER_HOME=/home/$(whoami)

while true
do
    case "$1" in
    -h|--head)
        IS_HEAD_NODE=true
        ;;
    --node_ip_address)
        NODE_IP_ADDRESS=$2
        shift
        ;;
    --head_address)
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

# Hadoop cloud credential configuration functions
. "$ROOT_DIR"/common/scripts/hadoop-cloud-credential.sh

function prepare_base_conf() {
    source_dir=$(cd $(dirname ${BASH_SOURCE[0]})/..;pwd)/conf
    output_dir=/tmp/hdfs/conf
    rm -rf  $output_dir
    mkdir -p $output_dir
    cp -r $source_dir/* $output_dir
    # Include hadoop config file for cloud providers
    cp -r "$ROOT_DIR"/common/conf/hadoop $output_dir
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
            # Keep the directory not contain outdated information of cluster.(Local mode)
            sudo rm -rf $data_disk/dfs/dn
            if [ -z "$hdfs_dn_dirs" ]; then
                hdfs_dn_dirs=$data_disk/dfs/dn
            else
                hdfs_dn_dirs="$hdfs_dn_dirs,$data_disk/dfs/dn"
            fi
        done
    fi

    # if no disks mounted on /mnt/cloudtik
    if [ -z "$hdfs_dn_dirs" ]; then
        sudo rm -rf "${HADOOP_HOME}/data/dfs/dn"
        hdfs_dn_dirs="${HADOOP_HOME}/data/dfs/dn"
    fi
    sed -i "s!{%dfs.namenode.name.dir%}!${hdfs_nn_dirs}!g" `grep "{%dfs.namenode.name.dir%}" -rl ./`
    sed -i "s!{%dfs.datanode.data.dir%}!${hdfs_dn_dirs}!g" `grep "{%dfs.datanode.data.dir%}" -rl ./`
}

function update_cloud_storage_credential_config() {
    set_cloud_storage_provider

    # update hadoop credential config
    update_credential_config_for_provider

    if [ "${cloud_storage_provider}" != "none" ];then
        cp -r ${output_dir}/hadoop/${cloud_storage_provider}/core-site.xml ${HADOOP_HOME}/etc/hadoop/
    else
        cp -r ${output_dir}/hadoop/core-site.xml ${HADOOP_HOME}/etc/hadoop/
    fi
}

function configure_hdfs() {
    prepare_base_conf
    mkdir -p ${HADOOP_HOME}/logs
    cd $output_dir

    fs_default_dir="hdfs://${HEAD_ADDRESS}:9000"
    sed -i "s!{%fs.default.name%}!${fs_default_dir}!g" `grep "{%fs.default.name%}" -rl ./`

    # update hadoop credential config
    update_cloud_storage_credential_config

    update_hdfs_data_disks_config

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
