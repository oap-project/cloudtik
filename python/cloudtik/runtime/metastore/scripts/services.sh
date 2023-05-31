#!/bin/bash

if [ ! -n "${METASTORE_HOME}" ]; then
    echo "Hive Metastore is not installed."
    exit 1
fi

command=$1
shift

# Parsing arguments
IS_HEAD_NODE=false

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    -h|--head)
        IS_HEAD_NODE=true
        ;;
    *)
        echo "Unknown argument passed."
        exit 1
    esac
    shift
done

case "$command" in
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
