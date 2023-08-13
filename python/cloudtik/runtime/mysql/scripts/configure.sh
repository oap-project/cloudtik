#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

USER_HOME=/home/$(whoami)
RUNTIME_PATH=$USER_HOME/runtime
MYSQL_HOME=$RUNTIME_PATH/mysql

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function prepare_base_conf() {
    local source_dir=$(dirname "${BIN_DIR}")/conf
    output_dir=/tmp/mysql/conf
    rm -rf  $output_dir
    mkdir -p $output_dir
    cp -r $source_dir/* $output_dir
}

function check_mysql_installed() {
    if ! command -v mysqld &> /dev/null
    then
        echo "MySQL is not installed for mysqld command is not available."
        exit 1
    fi
}

function update_data_dir() {
    local data_disk_dir=$(get_first_data_disk_dir)
    if [ -z "$data_disk_dir" ]; then
        data_dir="${MYSQL_HOME}/data"
    else
        data_dir="$data_disk_dir/mysql/data"
    fi

    mkdir -p ${data_dir}
    sed -i "s#{%data.dir%}#${data_dir}#g" ${config_template_file}
}

function configure_mysql() {
    prepare_base_conf
    config_template_file=${output_dir}/my.cnf

    mkdir -p ${MYSQL_HOME}/logs

    # ensure that /var/run/mysqld (used for socket and lock files) is writable
    # regardless of the UID our mysqld instance ends up having at runtime
    sudo mkdir -p /var/run/mysqld \
    && sudo chown -R $(whoami):$(id -gn) /var/run/mysqld \
    && sudo chmod 1777 /var/run/mysqld

    update_data_dir

    MYSQL_CONFIG_DIR=${MYSQL_HOME}/conf
    mkdir -p ${MYSQL_CONFIG_DIR}
    MYSQL_CONFIG_FILE=${MYSQL_CONFIG_DIR}/my.cnf
    cp -r ${config_template_file} ${MYSQL_CONFIG_FILE}

    # check and initialize the database if needed
    bash $BIN_DIR/mysql-init.sh mysqld --defaults-file=${MYSQL_CONFIG_FILE}
}

set_head_option "$@"
check_mysql_installed
configure_mysql

exit 0
