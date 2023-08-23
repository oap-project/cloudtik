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

function init_or_upgrade_schema() {
    METASTORE_SCHEMA_OK=true
    ${METASTORE_HOME}/bin/schematool -validate -dbType ${DATABASE_ENGINE} > ${METASTORE_HOME}/logs/configure.log 2>&1
    if [ $? != 0 ]; then
        # Either we need to initSchema or we need upgradeSchema
        echo "Trying to initialize the metastore schema..."
        ${METASTORE_HOME}/bin/schematool -initSchema -dbType ${DATABASE_ENGINE} > ${METASTORE_HOME}/logs/configure.log 2>&1
        if [ $? != 0 ]; then
            # Failed to init the schema, it may already exists
            echo "Trying to upgrade the metastore schema..."
            ${METASTORE_HOME}/bin/schematool -upgradeSchema -dbType ${DATABASE_ENGINE} > ${METASTORE_HOME}/logs/configure.log 2>&1
            if [ $? != 0 ]; then
                echo "Metastore schema initialization or upgrade failed."
                METASTORE_SCHEMA_OK=false
            else
                echo "Successfully upgraded the metastore schema."
            fi
        else
            echo "Successfully initialized the metastore schema."
        fi
    fi
}

function configure_hive_metastore() {
    prepare_base_conf
    config_template_file=${output_dir}/hive/metastore-site.xml

    mkdir -p ${METASTORE_HOME}/logs

    DATABASE_NAME=hive_metastore
    if [ "${CLOUD_DATABASE}" == "true" ]; then
        DATABASE_ADDRESS=${CLOUD_DATABASE_HOSTNAME}:${CLOUD_DATABASE_PORT}
        DATABASE_USER=${CLOUD_DATABASE_USERNAME}
        DATABASE_PASSWORD=${CLOUD_DATABASE_PASSWORD}
        DATABASE_ENGINE=${CLOUD_DATABASE_ENGINE}
    else
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
        # create database for postgresql if not exists
        echo "SELECT 'CREATE DATABASE ${DATABASE_NAME}' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${DATABASE_NAME}')\gexec" | PGPASSWORD=${CLOUD_DATABASE_PASSWORD} \
            psql \
            --host=${CLOUD_DATABASE_HOSTNAME} \
            --port=${CLOUD_DATABASE_PORT} \
            --username=${CLOUD_DATABASE_USERNAME} > ${METASTORE_HOME}/logs/configure.log
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

    # How about move this step to service start step
    # so that the depended service already started when running this initialization
    if [ "${CLOUD_DATABASE}" != "true" ] || [ "$METASTORE_WITH_CLOUD_DATABASE" == "false" ]; then
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
            FLUSH PRIVILEGES;" > ${METASTORE_HOME}/logs/configure.log

        # initialize the metastore database schema
        init_or_upgrade_schema

        # Stop mariadb after configured
        sudo service mysql stop
    else
        # initialize the metastore database schema
        init_or_upgrade_schema
    fi

    if [ "${METASTORE_SCHEMA_OK}" != "true" ]; then
        exit 1
    fi
}

set_head_option "$@"

if [ $IS_HEAD_NODE == "true" ]; then
    # Do nothing for workers
    check_hive_metastore_installed
    set_head_address
    configure_hive_metastore
fi

exit 0
