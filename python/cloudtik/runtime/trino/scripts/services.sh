#!/bin/bash

if [ ! -n "${TRINO_HOME}" ]; then
    echo "Trino is not installed for TRINO_HOME environment variable is not set."
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
    $TRINO_HOME/bin/launcher start
    ;;
stop)
    $TRINO_HOME/bin/launcher stop
    ;;
-h|--help)
    echo "Usage: $0 start|stop --head" >&2
    ;;
*)
    echo "Usage: $0 start|stop --head" >&2
    ;;
esac

exit 0
