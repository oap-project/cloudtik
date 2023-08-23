#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

USER_HOME=/home/$(whoami)

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function prepare_base_conf() {
    source_dir=$(dirname "${BIN_DIR}")/conf
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

function configure_hive_metastore() {
    prepare_base_conf
    config_template_file=${output_dir}/hive/metastore-site.xml

    mkdir -p ${METASTORE_HOME}/logs

    DATABASE_NAME=hive_metastore
    if [ "${SQL_DATABASE}" == "true" ] \
      && [ "$METASTORE_WITH_SQL_DATABASE" != "false" ]; then
        # a standalone SQL database
        DATABASE_ADDRESS=${SQL_DATABASE_HOST}:${SQL_DATABASE_PORT}
        DATABASE_USER=${SQL_DATABASE_USERNAME}
        DATABASE_PASSWORD=${SQL_DATABASE_PASSWORD}
        DATABASE_ENGINE=${SQL_DATABASE_ENGINE}
    else
        # local mariadb
        DATABASE_ADDRESS=localhost
        DATABASE_USER=hive
        DATABASE_PASSWORD=hive
        DATABASE_ENGINE="mysql"
    fi

    if [ "${DATABASE_ENGINE}" == "mysql" ]; then
        DATABASE_DRIVER="com.mysql.jdbc.Driver"
        DATABASE_CONNECTION="jdbc:mysql://${DATABASE_ADDRESS}/${DATABASE_NAME}?createDatabaseIfNotExist=true"
    else
        DATABASE_DRIVER="org.postgresql.Driver"
        DATABASE_CONNECTION="jdbc:postgresql://${DATABASE_ADDRESS}/${DATABASE_NAME}"
    fi

    sed -i "s/{%HEAD_ADDRESS%}/${HEAD_ADDRESS}/g" ${config_template_file}
    sed -i "s#{%DATABASE_CONNECTION%}#${DATABASE_CONNECTION}#g" ${config_template_file}
    sed -i "s#{%DATABASE_DRIVER%}#${DATABASE_DRIVER}#g" ${config_template_file}
    sed -i "s/{%DATABASE_USER%}/${DATABASE_USER}/g" ${config_template_file}
    sed -i "s/{%DATABASE_PASSWORD%}/${DATABASE_PASSWORD}/g" ${config_template_file}

    # set metastore warehouse dir according to the storage options: HDFS, S3, GCS, Azure
    # The full path will be decided on the default.fs of hadoop core-site.xml
    METASTORE_WAREHOUSE_DIR=/shared/warehouse
    sed -i "s|{%METASTORE_WAREHOUSE_DIR%}|${METASTORE_WAREHOUSE_DIR}|g" ${config_template_file}

    cp -r ${config_template_file}  ${METASTORE_HOME}/conf/metastore-site.xml
}

set_head_option "$@"

if [ $IS_HEAD_NODE == "true" ]; then
    # Do nothing for workers
    check_hive_metastore_installed
    set_head_address
    configure_hive_metastore
fi

exit 0
