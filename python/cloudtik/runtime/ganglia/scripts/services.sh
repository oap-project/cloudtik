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
    if [ $IS_HEAD_NODE == "true" ]; then
        sudo service apache2 start
        sudo service gmetad start
        sudo service ganglia-monitor start
    else
        sudo service ganglia-monitor start
    fi
    ;;
stop)
    if [ $IS_HEAD_NODE == "true" ]; then
        sudo service ganglia-monitor stop
        sudo service gmetad stop
        sudo service apache2 stop
    else
        sudo service ganglia-monitor stop
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
