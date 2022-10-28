#!/bin/bash

case "$1" in
start-head)
    # Start MLflow service
    nohup mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root $RUNTIME_PATH/mlflow --host 0.0.0.0  -p 5001 >mlflow.log 2>&1 &
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
