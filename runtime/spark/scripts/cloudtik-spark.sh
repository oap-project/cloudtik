#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )


case "$1" in
  update-config)
   shift 1 # past argument
    bash $SCRIPT_DIR/update-config.sh "$@"
    ;;
  start-head|stop-head|start-worker|stop-worker)
    bash $SCRIPT_DIR/hadoop-daemon.sh "$@"
    ;;
  -h|--help)
    echo "Usage: $0 update-config|start-head|stop-head|start-worker|stop-worker" >&2
    ;;
  *)
    echo "Usage: $0 update-config|start-head|stop-head|start-worker|stop-worker" >&2
    ;;
esac
