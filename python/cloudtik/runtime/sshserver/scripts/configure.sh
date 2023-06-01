#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

USER_HOME=/home/$(whoami)
SSH_CONFIG_HOME=${USER_HOME}/.ssh
SSH_SSHD_CONFIG_FILE=${SSH_CONFIG_HOME}/cloudtik-ssh-server-sshd_config
SSH_HOST_KEY_FILE=${SSH_CONFIG_HOME}/cloudtik-ssh-server-host_key
SSH_AUTHORIZED_KEYS_FILE=${SSH_CONFIG_HOME}/cloudtik-ssh-server-authorized_keys
SSH_DEFAULT_PRIVATE_KEY_FILE=${SSH_CONFIG_HOME}/id_rsa
BOOTSTRAP_PRIVATE_KEY_FILE=${USER_HOME}/cloudtik_bootstrap_key.pem

function prepare_base_conf() {
    source_dir=$(cd $(dirname ${BASH_SOURCE[0]})/..;pwd)/conf
    output_dir=/tmp/ssh_server/conf
    rm -rf  $output_dir
    mkdir -p $output_dir
    cp -r $source_dir/* $output_dir
}

function configure_ssh_server() {
    prepare_base_conf

    if [ ! -f "${SSH_HOST_KEY_FILE}" ]; then
        # generate the host key pair
        ssh-keygen -b 2048 -t rsa -q -N "" -f ${SSH_HOST_KEY_FILE} && chmod 600 ${SSH_HOST_KEY_FILE}
    fi

    if [ -f "${SSH_AUTHORIZED_KEYS_FILE}" ]; then
        chmod 600 ${SSH_AUTHORIZED_KEYS_FILE}
    fi

    cp ${output_dir}/sshd_config ${SSH_SSHD_CONFIG_FILE}

    if [ "$IS_HEAD_NODE" == "true" ]; then
        # for head, configure the private key as the default key to login ~/.ssh/id_rsa
        # We don't overwrite the existing id_rsa. Any situation for problems?
        if [ -f "${BOOTSTRAP_PRIVATE_KEY_FILE}" ] && [ ! -f "${SSH_DEFAULT_PRIVATE_KEY_FILE}" ]; then
            cp ${BOOTSTRAP_PRIVATE_KEY_FILE} ${SSH_DEFAULT_PRIVATE_KEY_FILE}
        fi
    fi
}

set_head_option "$@"
set_head_address
configure_ssh_server

exit 0
