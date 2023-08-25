#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

# import util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

# schema initialization functions
. "$BIN_DIR"/schema-init.sh

if [ ! -n "${METASTORE_HOME}" ]; then
    echo "Hive Metastore is not installed."
    exit 1
fi

set_head_option "$@"
set_service_command "$@"

case "$SERVICE_COMMAND" in
start)
    if [ "${METASTORE_HIGH_AVAILABILITY}" == "true" ] \
      || [ "${IS_HEAD_NODE}" == "true" ]; then
        if [ "${SQL_DATABASE}" != "true" ] \
          || [ "$METASTORE_WITH_SQL_DATABASE" == "false" ]; then
            # local database
            sudo service mysql start

            # do schema check and init
            init_schema
        else
            if [ "${IS_HEAD_NODE}" == "true" ]; then
                # do schema check and init only on head
                init_schema
            fi
        fi

        nohup $METASTORE_HOME/bin/start-metastore \
          >${METASTORE_HOME}/logs/start-metastore.log 2>&1 &
    fi
    ;;
stop)
    if [ "${METASTORE_HIGH_AVAILABILITY}" == "true" ] \
      || [ "${IS_HEAD_NODE}" == "true" ]; then
        if [ "${SQL_DATABASE}" != "true" ] \
          || [ "$METASTORE_WITH_SQL_DATABASE" == "false" ]; then
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
