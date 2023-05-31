#!/bin/bash

if ! command -v ray &> /dev/null
then
    echo "Ray is not installed for ray command is not available."
    exit 1
fi

case "$1" in
start-head)
    ray start --head --node-ip-address=$CLOUDTIK_NODE_IP --port=6379 --no-monitor --disable-usage-stats > /dev/null
    ;;
stop-head)
    ray stop
    ;;
start-worker)
    ray start --node-ip-address=$CLOUDTIK_NODE_IP --address=$CLOUDTIK_HEAD_IP:6379 > /dev/null
    ;;
stop-worker)
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
