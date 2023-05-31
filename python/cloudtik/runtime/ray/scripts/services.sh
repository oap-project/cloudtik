#!/bin/bash

if ! command -v ray &> /dev/null
then
    echo "Ray is not installed for ray command is not available."
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
