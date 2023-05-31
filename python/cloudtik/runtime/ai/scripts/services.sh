#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

USER_HOME=/home/$(whoami)
RUNTIME_PATH=$USER_HOME/runtime
MLFLOW_DATA=$RUNTIME_PATH/mlflow
MLFLOW_ARTIFACT_PATH=shared/mlflow

# import util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function set_artifact_config_for_local_hdfs() {
    DEFAULT_ARTIFACT_ROOT="hdfs://${HEAD_ADDRESS}:9000/${MLFLOW_ARTIFACT_PATH}"
}

function set_artifact_config_for_hdfs() {
    DEFAULT_ARTIFACT_ROOT="${HDFS_NAMENODE_URI}/${MLFLOW_ARTIFACT_PATH}"
}

function set_artifact_config_for_s3() {
    DEFAULT_ARTIFACT_ROOT="s3://${AWS_S3_BUCKET}/${MLFLOW_ARTIFACT_PATH}"
}

function set_artifact_config_for_azure_data_lake() {
    if [ "$AZURE_STORAGE_TYPE" == "blob" ];then
        AZURE_SCHEMA="wasbs"
        AZURE_ENDPOINT="blob"
    else
        # Default to datalake
        # Must be Azure storage kind must be blob (Azure Blob Storage) or datalake (Azure Data Lake Storage Gen 2)
        # MLflow uses abfss instead of abfs
        AZURE_SCHEMA="abfss"
        AZURE_ENDPOINT="dfs"
    fi

    fs_dir="${AZURE_SCHEMA}://${AZURE_CONTAINER}@${AZURE_STORAGE_ACCOUNT}.${AZURE_ENDPOINT}.core.windows.net"
    DEFAULT_ARTIFACT_ROOT="${fs_dir}/${MLFLOW_ARTIFACT_PATH}"
}

function set_artifact_config_for_gcs() {
    DEFAULT_ARTIFACT_ROOT="gs://${GCP_GCS_BUCKET}/${MLFLOW_ARTIFACT_PATH}"
}

function set_artifact_config_for_cloud_storage() {
    if [ "$HDFS_ENABLED" == "true" ]; then
        set_artifact_config_for_local_hdfs
    elif [ ! -z "${HDFS_NAMENODE_URI}" ]; then
        set_artifact_config_for_hdfs
    elif [ "$AWS_CLOUD_STORAGE" == "true" ]; then
        set_artifact_config_for_s3
    elif [ "$AZURE_CLOUD_STORAGE" == "true" ]; then
        set_artifact_config_for_azure_data_lake
    elif [ "$GCP_CLOUD_STORAGE" == "true" ]; then
        set_artifact_config_for_gcs
    fi
}

function set_mlflow_server_config() {
    if [ "${CLOUD_DATABASE}" == "true" ] && [ "$AI_WITH_CLOUD_DATABASE" != "false" ]; then
        DATABASE_NAME=mlflow
        BACKEND_STORE_URI=mysql+pymysql://${CLOUD_DATABASE_USERNAME}:${CLOUD_DATABASE_PASSWORD}@${CLOUD_DATABASE_HOSTNAME}:${CLOUD_DATABASE_PORT}/${DATABASE_NAME}
    else
        BACKEND_STORE_URI=sqlite:///${MLFLOW_DATA}/mlflow.db
    fi

    if [ "$AI_WITH_CLOUD_STORAGE" != "false" ]; then
        set_artifact_config_for_cloud_storage
    fi

    if [ "${DEFAULT_ARTIFACT_ROOT}" == "" ]; then
        DEFAULT_ARTIFACT_ROOT=${MLFLOW_DATA}/mlruns
    fi
}

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
        set_head_address

        # Will set BACKEND_STORE_URI and DEFAULT_ARTIFACT_ROOT
        set_mlflow_server_config

        # Start MLflow service
        nohup mlflow server --backend-store-uri ${BACKEND_STORE_URI} --default-artifact-root ${DEFAULT_ARTIFACT_ROOT} --host 0.0.0.0 -p 5001 >${MLFLOW_DATA}/logs/mlflow.log 2>&1 &
    fi
    ;;
stop)
    if [ $IS_HEAD_NODE == "true" ]; then
        # Stop MLflow service
        ps aux | grep 'mlflow.server:app' | grep -v grep | awk '{print $2}' | xargs -r kill -9
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
