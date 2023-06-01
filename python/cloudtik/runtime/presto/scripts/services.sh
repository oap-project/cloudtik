#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

# import util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

if [ ! -n "${PRESTO_HOME}" ]; then
    echo "Presto is not installed for PRESTO_HOME environment variable is not set."
    exit 1
fi

set_head_option "$@"
set_service_command "$@"

case "$SERVICE_COMMAND" in
start)
    $PRESTO_HOME/bin/launcher start
    ;;
stop)
    $PRESTO_HOME/bin/launcher stop
    ;;
-h|--help)
    echo "Usage: $0 start|stop --head" >&2
    ;;
*)
    echo "Usage: $0 start|stop --head" >&2
    ;;
esac

exit 0
