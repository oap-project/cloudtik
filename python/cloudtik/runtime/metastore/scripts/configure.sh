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

if [ not $IS_HEAD_NODE == "true" ]; then
    # Do nothing for workers
    exit 0
fi

function prepare_base_conf() {
    source_dir=$(cd $(dirname ${BASH_SOURCE[0]})/..;pwd)/conf
    output_dir=/tmp/metastore/conf
    rm -rf  $output_dir
    mkdir -p $output_dir
    cp -r $source_dir/* $output_dir
}

function check_hive_metastore_installed() {
    if [ ! -n "${METASTORE_HOME}" ]; then
        echo "Hive Metastore is not installed."
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

function configure_hive_metastore() {
    prepare_base_conf
    cd $output_dir
    sed -i "s/{%HEAD_ADDRESS%}/${HEAD_ADDRESS}/g" `grep "{%HEAD_ADDRESS%}" -rl ./`
    sed -i "s/{%DATABASE_NAME%}/${DATABASE_NAME}/g" `grep "{%DATABASE_NAME%}" -rl ./`
    sed -i "s/{%DATABASE_USER%}/${DATABASE_USER}/g" `grep "{%DATABASE_USER%}" -rl ./`
    sed -i "s/{%DATABASE_PASSWORD%}/${DATABASE_PASSWORD}/g" `grep "{%DATABASE_PASSWORD%}" -rl ./`

    # TODO: set metastore warehouse dir according to the storage options: HDFS, S3, GCS, Azure
    METASTORE_WAREHOUSE_DIR=$USER_HOME/shared/data/warehouse
    sed -i "s/{%METASTORE_WAREHOUSE_DIR%}/${METASTORE_WAREHOUSE_DIR}/g" `grep "{%METASTORE_WAREHOUSE_DIR%}" -rl ./`

    cp -r ${output_dir}/hive/metastore-site.xml  ${METASTORE_HOME}/conf/metastore-site.xml

    DATABASE_NAME=hive_metastore
    DATABASE_USER=hive
    DATABASE_PASSWORD=hive

    # Start mariadb
    sudo service mysql start
    # Do we need wait a few seconds for mysql to startup?

    # We may not need to create database as hive can create if it not exist
    # create user
    sudo mysql -u root -e "
        DROP DATABASE IF EXISTS ${DATABASE_NAME};
        CREATE DATABASE ${DATABASE_NAME};
        CREATE USER '${DATABASE_USER}'@localhost IDENTIFIED BY '${DATABASE_PASSWORD}';
        GRANT ALL PRIVILEGES ON *.* TO '${DATABASE_USER}'@'localhost';
        FLUSH PRIVILEGES;"

    # initialize the metastore database schema
    ${METASTORE_HOME}/bin/schematool -initSchema -dbType mysql

    # Stop mariadb after configured
    sudo service mysql stop
}

check_hive_metastore_installed
set_head_address
configure_hive_metastore
