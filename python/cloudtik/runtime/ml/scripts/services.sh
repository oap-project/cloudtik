#!/bin/bash

USER_HOME=/home/$(whoami)
RUNTIME_PATH=$USER_HOME/runtime

case "$1" in
start-head)
    # Start MLflow service
    if [ "${CLOUD_DATABASE}" == "true" ] && [ "$ML_WITH_CLOUD_DATABASE" != "false" ]; then
        BACKEND_STORE_URI=mysql://${CLOUD_DATABASE_USERNAME}:${CLOUD_DATABASE_PASSWORD}@${CLOUD_DATABASE_HOSTNAME}:${CLOUD_DATABASE_PORT}/mlflow
    else
        BACKEND_STORE_URI=sqlite:///${MLFLOW_DATA}/mlflow.db
    fi

    MLFLOW_DATA=$RUNTIME_PATH/mlflow
    nohup mlflow server --backend-store-uri ${BACKEND_STORE_URI} --default-artifact-root ${MLFLOW_DATA}/mlruns --host 0.0.0.0 -p 5001 >${MLFLOW_DATA}/logs/mlflow.log 2>&1 &
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
