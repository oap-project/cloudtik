#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

# import util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

if ! command -v ray &> /dev/null
then
    echo "Ray is not installed for ray command is not available."
    exit 1
fi

set_head_option "$@"
set_service_command "$@"

case "$SERVICE_COMMAND" in
start)
    if [ $IS_HEAD_NODE == "true" ]; then
        ray start --head --node-ip-address=$CLOUDTIK_NODE_IP --port=6379 --no-monitor --disable-usage-stats > /dev/null
    else
        ray start --node-ip-address=$CLOUDTIK_NODE_IP --address=$CLOUDTIK_HEAD_IP:6379 > /dev/null
    fi
    ;;
stop)
    ray stop
    ;;
-h|--help)
    echo "Usage: $0 start|stop --head" >&2
    ;;
*)
    echo "Usage: $0 start|stop --head" >&2
    ;;
esac

exit 0
