#!/bin/bash

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
