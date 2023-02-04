#!/bin/bash

USER_HOME=/home/$(whoami)
RUNTIME_PATH=$USER_HOME/runtime
MLFLOW_DATA=$RUNTIME_PATH/mlflow

case "$1" in
start-head)
    # Start MLflow service
    nohup mlflow server --backend-store-uri sqlite:///${MLFLOW_DATA}/mlflow.db --default-artifact-root ${MLFLOW_DATA}/mlruns --host 0.0.0.0 -p 5001 >${MLFLOW_DATA}/mlflow.log 2>&1 &
    ;;
stop-head)
    # Stop MLflow service
    ps aux | grep 'mlflow.server:app' | grep -v grep | awk '{print $2}' | xargs -r kill -9
    ;;
start-worker)
    # No need to run anything for worker node
    ;;
stop-worker)
    # No need to run anything for worker node
    ;;
-h|--help)
    echo "Usage: $0 start-head|stop-head|start-worker|stop-worker" >&2
    ;;
*)
    echo "Usage: $0 start-head|stop-head|start-worker|stop-worker" >&2
    ;;
esac

exit 0
