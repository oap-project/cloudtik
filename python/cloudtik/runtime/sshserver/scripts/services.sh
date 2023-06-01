#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

# import util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

USER_HOME=/home/$(whoami)
SSH_CONFIG_HOME=${USER_HOME}/.ssh
SSH_SSHD_CONFIG_FILE=${SSH_CONFIG_HOME}/cloudtik-ssh-server-sshd_config
SSH_HOST_KEY_FILE=${SSH_CONFIG_HOME}/cloudtik-ssh-server-host_key
SSH_AUTHORIZED_KEYS_FILE=${SSH_CONFIG_HOME}/cloudtik-ssh-server-authorized_keys

set_head_option "$@"
set_service_command "$@"
set_node_ip_address

case "$SERVICE_COMMAND" in
start)
    # start sshd
    # we may not need sshd for head
    /usr/sbin/sshd -f ${SSH_SSHD_CONFIG_FILE} -h ${SSH_HOST_KEY_FILE} -p ${SSH_SERVER_PORT} -o ListenAddress=${NODE_IP_ADDRESS} -o AuthorizedKeysFile=${SSH_AUTHORIZED_KEYS_FILE}
    ;;
stop)
    # stop sshd by matching
    pkill -f "/usr/sbin/sshd -f ${SSH_SSHD_CONFIG_FILE}"
    ;;
-h|--help)
    echo "Usage: $0 start|stop --head" >&2
    ;;
*)
    echo "Usage: $0 start|stop --head" >&2
    ;;
esac

exit 0
