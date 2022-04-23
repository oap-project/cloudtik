#!/bin/bash

case "$1" in
start-head)
    sudo service apache2 start
    sudo service gmetad start
    sudo service ganglia-monitor start
    ;;
stop-head)
    sudo service ganglia-monitor stop
    sudo service gmetad stop
    sudo service apache2 stop
    ;;
start-worker)
    sudo service ganglia-monitor start
    ;;
stop-worker)
    sudo service ganglia-monitor stop
    ;;
-h|--help)
    echo "Usage: $0 start-head|stop-head|start-worker|stop-worker" >&2
    ;;
*)
    echo "Usage: $0 start-head|stop-head|start-worker|stop-worker" >&2
    ;;
esac
