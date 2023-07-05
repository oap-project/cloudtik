#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

USER_HOME=/home/$(whoami)

# Hadoop cloud credential configuration functions
. "$ROOT_DIR"/common/scripts/hadoop-cloud-credential.sh

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

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

function update_proxy_user_for_current_user() {
    CURRENT_SYSTEM_USER=$(whoami)

    if [ "${CURRENT_SYSTEM_USER}" != "root" ]; then
        HADOOP_PROXY_USER_PROPERTIES="<property>\n\
        <name>hadoop.proxyuser.${CURRENT_SYSTEM_USER}.groups</name>\n\
        <value>*</value>\n\
    </property>\n\
    <property>\n\
        <name>hadoop.proxyuser.${CURRENT_SYSTEM_USER}.hosts</name>\n\
        <value>*</value>\n\
    </property>"
        sed -i "s#{%hadoop.proxyuser.properties%}#${HADOOP_PROXY_USER_PROPERTIES}#g" `grep "{%hadoop.proxyuser.properties%}" -rl ./`
    else
        sed -i "s#{%hadoop.proxyuser.properties%}#""#g" `grep "{%hadoop.proxyuser.properties%}" -rl ./`
    fi
}

function configure_hdfs() {
    prepare_base_conf
    mkdir -p ${HADOOP_HOME}/logs
    cd $output_dir

    fs_default_dir="hdfs://${HEAD_ADDRESS}:9000"
    sed -i "s!{%fs.default.name%}!${fs_default_dir}!g" `grep "{%fs.default.name%}" -rl ./`

    update_proxy_user_for_current_user
    update_hdfs_data_disks_config

    HDFS_CONF_DIR=${HADOOP_HOME}/etc/hdfs
    # copy the existing hadoop conf
    mkdir -p ${HDFS_CONF_DIR}
    cp -r  ${HADOOP_HOME}/etc/hadoop/* ${HDFS_CONF_DIR}/
    # override hdfs conf
    cp -r ${output_dir}/hadoop/core-site.xml ${HDFS_CONF_DIR}/
    cp -r ${output_dir}/hadoop/hdfs-site.xml  ${HDFS_CONF_DIR}/

    if [ $IS_HEAD_NODE == "true" ];then
        # TODO: format only once if there is no force format flag
        export HADOOP_CONF_DIR= ${HDFS_CONF_DIR}
        # Stop namenode in case it was running left from last try
        ${HADOOP_HOME}/bin/hdfs --daemon stop namenode > /dev/null 2>&1
        # Format hdfs once
        ${HADOOP_HOME}/bin/hdfs --loglevel WARN namenode -format -force
    fi
}

set_head_option "$@"
check_hadoop_installed
set_head_address
configure_hdfs

exit 0
