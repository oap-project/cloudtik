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
    output_dir=/tmp/presto/conf
    rm -rf  $output_dir
    mkdir -p $output_dir
    cp -r $source_dir/* $output_dir
}

function check_presto_installed() {
    if [ ! -n "${PRESTO_HOME}" ]; then
        echo "Presto is not installed for PRESTO_HOME environment variable is not set."
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

function update_presto_data_disks_config() {
    presto_data_dir=""
    if [ -d "/mnt/cloudtik" ]; then
        for data_disk in /mnt/cloudtik/*; do
            [ -d "$data_disk" ] || continue
            if [ -z "$presto_data_dir" ]; then
                presto_data_dir=$data_disk/presto/data
            fi
        done
    fi

    # if no disks mounted on /mnt/cloudtik
    if [ -z "$presto_data_dir" ]; then
        presto_data_dir="${RUNTIME_PATH}/shared/data"
    fi

    mkdir -p $presto_data_dir
    sed -i "s!{%node.data-dir%}!${presto_data_dir}!g" $output_dir/presto/node.properties
}

function update_hive_metastore_config() {
    # To be improved for external metastore cluster
    HIVE_PROPERTIES=${output_dir}/presto/catalog/hive.properties
    if [ "$METASTORE_ENABLED" == "true" ];then
        METASTORE_IP=${HEAD_ADDRESS}
        hive_metastore_uris="thrift://${METASTORE_IP}:9083"
        sed -i "s!{%HIVE_METASTORE_URI%}!${hive_metastore_uris}!g" ${HIVE_PROPERTIES}
        cp ${HIVE_PROPERTIES}  ${PRESTO_HOME}/etc/catalog/hive.properties
    fi
}

function update_metastore_config() {
    update_hive_metastore_config
}
function configure_presto() {
    prepare_base_conf
    update_metastore_config
    cd $output_dir
    node_id=$(uuid)
    presto_log_dir=${PRESTO_HOME}/logs
    mkdir -p ${presto_log_dir}
    sed -i "s/{%HEAD_ADDRESS%}/${HEAD_ADDRESS}/g" `grep "{%HEAD_ADDRESS%}" -rl ./`
    sed -i "s/{%node.environment%}/presto/g" $output_dir/presto/node.properties
    sed -i "s/{%node.id%}/${node_id}/g" $output_dir/presto/node.properties
    sed -i "s!{%node.log-dir%}!${presto_log_dir}!g" $output_dir/presto/node.properties

    update_presto_data_disks_config

    mkdir -p ${PRESTO_HOME}/etc
    if [ $IS_HEAD_NODE == "true" ]; then
        cp ${output_dir}/presto/config.properties  ${PRESTO_HOME}/etc/config.properties
    else
        cp ${output_dir}/presto/config.worker.properties  ${PRESTO_HOME}/etc/config.properties
    fi

    cp ${output_dir}/presto/jvm.config  ${PRESTO_HOME}/etc/jvm.config
    cp ${output_dir}/presto/node.properties  ${PRESTO_HOME}/etc/node.properties
}

check_presto_installed
set_head_address
configure_presto

exit 0
