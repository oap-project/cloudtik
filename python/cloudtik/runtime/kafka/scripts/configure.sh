#!/bin/bash

args=$(getopt -a -o h:: -l head::,node_ip_address:,head_address:,zookeeper_connect: -- "$@")
eval set -- "${args}"

IS_HEAD_NODE=false
USER_HOME=/home/$(whoami)
RUNTIME_PATH=$USER_HOME/runtime

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
    --zookeeper_connect)
        ZOOKEEPER_CONNECT=$2
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
    output_dir=/tmp/kafka/conf
    rm -rf  $output_dir
    mkdir -p $output_dir
    cp -r $source_dir/* $output_dir
}

function check_kafka_installed() {
    if [ ! -n "${KAFKA_HOME}" ]; then
        echo "KAFKA_HOME environment variable is not set."
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

function update_kafka_data_disks_config() {
    kafka_data_dir=""
    if [ -d "/mnt/cloudtik" ]; then
        for data_disk in /mnt/cloudtik/*; do
            [ -d "$data_disk" ] || continue
            if [ -z "$kafka_data_dir" ]; then
                kafka_data_dir=$data_disk/kafka/data
            else
                kafka_data_dir="$kafka_data_dir,$data_disk/kafka/data"
            fi
        done
    fi

    # if no disks mounted on /mnt/cloudtik
    if [ -z "$kafka_data_dir" ]; then
        kafka_data_dir="${RUNTIME_PATH}/shared/kafka/data"
        mkdir -p $kafka_data_dir
    fi

    sed -i "s!{%kafka.log.dir%}!${kafka_data_dir}!g" $output_dir/kafka/server.properties
}

function update_broker_id() {
    # Configure my id file
    if [ ! -n "${CLOUDTIK_NODE_NUMBER}" ]; then
        echo "No node number allocated for current node!"
        exit 1
    fi

    sed -i "s!{%kafka.broker.id%}!${CLOUDTIK_NODE_NUMBER}!g" $output_dir/kafka/server.properties
}

function update_zookeeper_connect() {
    # Update zookeeper connect
    if [ -z "$ZOOKEEPER_CONNECT" ]; then
        echo "No zookeeper connect parameter specified!"
        exit 1
    fi

    sed -i "s!{%kafka.zookeeper.connect%}!${ZOOKEEPER_CONNECT}!g" $output_dir/kafka/server.properties
}

function update_listeners() {
    kafka_listeners="PLAINTEXT://${NODE_IP_ADDRESS}:9092"
    sed -i "s!{%kafka.listeners%}!${kafka_listeners}!g" $output_dir/kafka/server.properties
}

function configure_kafka() {
    prepare_base_conf
    mkdir -p ${KAFKA_HOME}/logs
    cd $output_dir

    update_broker_id
    update_listeners
    update_zookeeper_connect
    update_kafka_data_disks_config

    cp -r ${output_dir}/kafka/server.properties  ${KAFKA_HOME}/config/server.properties
}

if [ $IS_HEAD_NODE == "false" ];then
    # Zookeeper doesn't run on head node
    check_kafka_installed
    set_head_address
    configure_kafka
fi

exit 0
