#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

# import util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

if [ ! -n "${METASTORE_HOME}" ]; then
    echo "Hive Metastore is not installed."
    exit 1
fi

set_head_option "$@"
set_service_command "$@"

case "$SERVICE_COMMAND" in
start)
    if [ $IS_HEAD_NODE == "true" ]; then
        if [ "${CLOUD_DATABASE}" != "true" ] || [ "$METASTORE_WITH_CLOUD_DATABASE" == "false" ]; then
            sudo service mysql start
        fi
        nohup $METASTORE_HOME/bin/start-metastore >${METASTORE_HOME}/logs/start-metastore.log 2>&1 &
    fi
    ;;
stop)
    if [ $IS_HEAD_NODE == "true" ]; then
        if [ "${CLOUD_DATABASE}" != "true" ] || [ "$METASTORE_WITH_CLOUD_DATABASE" == "false" ]; then
            sudo service mysql stop
        fi
        pkill -f 'HiveMetaStore'
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
