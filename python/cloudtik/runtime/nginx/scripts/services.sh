#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

# import util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

set_head_option "$@"
set_service_command "$@"

case "$SERVICE_COMMAND" in
start)
    if [ "${NGINX_HIGH_AVAILABILITY}" == "true" ] \
        || [ "${IS_HEAD_NODE}" == "true" ]; then
        sudo service nginx start
    fi
    ;;
stop)
    if [ "${NGINX_HIGH_AVAILABILITY}" == "true" ] \
        || [ "${IS_HEAD_NODE}" == "true" ]; then
        sudo service nginx stop
    fi
    ;;
-h|--help)
    echo "Usage: $0 start|stop --head" >&2
    ;;
*)
    echo "Usage: $0 start|stop --head" >&2
    ;;
esac

exit 0
