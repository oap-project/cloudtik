#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )


case "$1" in
  update-config)
   shift 1 # past argument
    $SCRIPT_DIR/update-config "$@"
    ;;
  start-head|stop-head|start-worker|stop-worker)
     $SCRIPT_DIR/hadoop-daemon "$@"
    ;;
  -h|--help)
    echo "Usage: $0 update-config|format-cluster|start-head|stop-head|start-worker|stop-worker" >&2
    ;;
  *)
    echo "Usage: $0 update-config|format-cluster|start-head|stop-head|start-worker|stop-worker" >&2
    ;;
esac
