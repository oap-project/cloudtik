#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

USER_HOME=/home/$(whoami)
RUNTIME_PATH=$USER_HOME/runtime
MLFLOW_HOME=$RUNTIME_PATH/mlflow

MLFLOW_ARTIFACT_PATH=shared/mlflow

# API cloud credential configuration functions
. "$ROOT_DIR"/common/scripts/api-credential.sh

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function prepare_base_conf() {
    source_dir=$(dirname "${BIN_DIR}")/conf
    output_dir=/tmp/ai/conf
    rm -rf  $output_dir
    mkdir -p $output_dir
    cp -r $source_dir/* $output_dir
}

function configure_system_folders() {
    # Create dirs for data
    mkdir -p ${MLFLOW_HOME}/logs
    mkdir -p ${MLFLOW_HOME}/mlruns
}

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
    if [ "${HADOOP_DEFAULT_CLUSTER}" == "true" ]; then
        if [ ! -z "${HDFS_NAMENODE_URI}" ]; then
            set_artifact_config_for_hdfs
            return 0
        elif [ "$HDFS_ENABLED" == "true" ]; then
            set_artifact_config_for_local_hdfs
            return 0
        fi
    fi

    if [ "$AWS_CLOUD_STORAGE" == "true" ]; then
        set_artifact_config_for_s3
    elif [ "$AZURE_CLOUD_STORAGE" == "true" ]; then
        set_artifact_config_for_azure_data_lake
    elif [ "$GCP_CLOUD_STORAGE" == "true" ]; then
        set_artifact_config_for_gcs
    elif [ ! -z "${HDFS_NAMENODE_URI}" ]; then
        set_artifact_config_for_hdfs
    elif [ "$HDFS_ENABLED" == "true" ]; then
        set_artifact_config_for_local_hdfs
    fi
}

function update_mlflow_server_config() {
    if [ "${SQL_DATABASE}" == "true" ] && [ "$AI_WITH_SQL_DATABASE" != "false" ]; then
        DATABASE_NAME=mlflow
        CONNECTION_INFO=${SQL_DATABASE_USERNAME}:${SQL_DATABASE_PASSWORD}@${SQL_DATABASE_HOST}:${SQL_DATABASE_PORT}/${DATABASE_NAME}
        if [ "${SQL_DATABASE_ENGINE}" == "mysql" ]; then
            BACKEND_STORE_URI=mysql+pymysql://${CONNECTION_INFO}
        else
            BACKEND_STORE_URI=postgresql+psycopg2://${CONNECTION_INFO}
        fi
    else
        BACKEND_STORE_URI=sqlite:///${MLFLOW_HOME}/mlflow.db
    fi

    if [ "$AI_WITH_CLOUD_STORAGE" != "false" ]; then
        set_artifact_config_for_cloud_storage
    fi

    if [ -z "${DEFAULT_ARTIFACT_ROOT}" ]; then
        DEFAULT_ARTIFACT_ROOT=${MLFLOW_HOME}/mlruns
    fi

    sed -i "s#{%backend.store.uri%}#${BACKEND_STORE_URI}#g" ${output_dir}/mlflow
    sed -i "s#{%default.artifact.root%}#${DEFAULT_ARTIFACT_ROOT}#g" ${output_dir}/mlflow
}

function patch_libraries() {
    HOROVOD_PYTHON_HOME="${ROOT_DIR}/../../horovod"
    local PATCHES_DIR=$output_dir/patches

    # Fix the Horovod on Spark bug for handling network interfaces of loopback
    HOROVOD_SPARK_GLOO_RUN_FILE="${HOROVOD_PYTHON_HOME}/spark/gloo_run.py"
    if [ -f "${HOROVOD_SPARK_GLOO_RUN_FILE}" ]; then
       cp ${PATCHES_DIR}/horovod_spark_gloo_run.py.patch ${HOROVOD_SPARK_GLOO_RUN_FILE}
    fi

    # Improve Horovod on Spark for support MPICH and IMPI
    HOROVOD_SPARK_MPI_RUN_FILE="${HOROVOD_PYTHON_HOME}/spark/mpi_run.py"
    if [ -f "${HOROVOD_SPARK_MPI_RUN_FILE}" ]; then
       cp ${PATCHES_DIR}/horovod_spark_mpi_run.py.patch ${HOROVOD_SPARK_MPI_RUN_FILE}
    fi

    # Fix the Horovod driver NIC issue
    HOROVOD_SPARK_RUNNER_FILE="${HOROVOD_PYTHON_HOME}/spark/runner.py"
    if [ -f "${HOROVOD_SPARK_RUNNER_FILE}" ]; then
       cp ${PATCHES_DIR}/horovod_spark_runner.py.patch ${HOROVOD_SPARK_RUNNER_FILE}
    fi

    HOROVOD_SPARK_DRIVER_DRIVER_SERVICE_FILE="${HOROVOD_PYTHON_HOME}/spark/driver/driver_service.py"
    if [ -f "${HOROVOD_SPARK_DRIVER_DRIVER_SERVICE_FILE}" ]; then
       cp ${PATCHES_DIR}/horovod_spark_driver_driver_service.py.patch ${HOROVOD_SPARK_DRIVER_DRIVER_SERVICE_FILE}
    fi

    # CloudTik remote command execution for Gloo
    HOROVOD_RUNNER_UTIL_REMOTE_FILE="${HOROVOD_PYTHON_HOME}/runner/util/remote.py"
    if [ -f "$HOROVOD_RUNNER_UTIL_REMOTE_FILE" ]; then
       cp ${PATCHES_DIR}/horovod_runner_util_remote.py.patch ${HOROVOD_RUNNER_UTIL_REMOTE_FILE}
    fi

    # Fix the remote command quote handling
    HOROVOD_RUNNER_GLOO_RUN_FILE="${HOROVOD_PYTHON_HOME}/runner/gloo_run.py"
    if [ -f "$HOROVOD_RUNNER_GLOO_RUN_FILE" ]; then
       cp ${PATCHES_DIR}/horovod_runner_gloo_run.py.patch ${HOROVOD_RUNNER_GLOO_RUN_FILE}
    fi

    # CloudTik remote command execution for MPI
    HOROVOD_RUNNER_MPI_RUN_FILE="${HOROVOD_PYTHON_HOME}/runner/mpi_run.py"
    if [ -f "$HOROVOD_RUNNER_MPI_RUN_FILE" ]; then
       cp ${PATCHES_DIR}/horovod_runner_mpi_run.py.patch ${HOROVOD_RUNNER_MPI_RUN_FILE}
    fi

    # Fix the Horovod driver NIC issue
    HOROVOD_RUNNER_LAUNCH_FILE="${HOROVOD_PYTHON_HOME}/runner/launch.py"
    if [ -f "${HOROVOD_RUNNER_LAUNCH_FILE}" ]; then
       cp ${PATCHES_DIR}/horovod_runner_launch.py.patch ${HOROVOD_RUNNER_LAUNCH_FILE}
    fi

    HOROVOD_RUNNER_COMMON_SERVICE_DRIVER_SERVICE_FILE="${HOROVOD_PYTHON_HOME}/runner/common/service/driver_service.py"
    if [ -f "${HOROVOD_RUNNER_COMMON_SERVICE_DRIVER_SERVICE_FILE}" ]; then
       cp ${PATCHES_DIR}/horovod_runner_common_service_driver_service.py.patch ${HOROVOD_RUNNER_COMMON_SERVICE_DRIVER_SERVICE_FILE}
    fi

    # Fix the Horovod bug for handling network interfaces of loopback
    HOROVOD_RUNNER_DRIVER_SERVICE_FILE="${HOROVOD_PYTHON_HOME}/runner/driver/driver_service.py"
    if [ -f "$HOROVOD_RUNNER_DRIVER_SERVICE_FILE" ]; then
       cp ${PATCHES_DIR}/horovod_runner_driver_driver_service.py.patch ${HOROVOD_RUNNER_DRIVER_SERVICE_FILE}
    fi

    # Improve Horovod on Spark for support MPICH and IMPI
    HOROVOD_SPARK_MPIRUN_EXEC_FN_FILE="${HOROVOD_PYTHON_HOME}/spark/task/mpirun_exec_fn.py"
    if [ -f "${HOROVOD_SPARK_MPIRUN_EXEC_FN_FILE}" ]; then
       cp ${PATCHES_DIR}/horovod_spark_task_mpirun_exec_fn.py.patch ${HOROVOD_SPARK_MPIRUN_EXEC_FN_FILE}
    fi

    HOROVOD_RAY_UTILS_FILE="${HOROVOD_PYTHON_HOME}/ray/utils.py"
    if [ -f "${HOROVOD_RAY_UTILS_FILE}" ]; then
       cp ${PATCHES_DIR}/horovod_ray_utils.py.patch ${HOROVOD_RAY_UTILS_FILE}
    fi

    # Fix the Azure managed identity from adlfs
    ADLFS_PYTHON_HOME="${ROOT_DIR}/../../adlfs"
    ADLFS_SPEC_FILE="${ADLFS_PYTHON_HOME}/spec.py"
    if [ -f "$ADLFS_SPEC_FILE" ]; then
        cp ${PATCHES_DIR}/adlfs_spec.py.patch ${ADLFS_SPEC_FILE}
    fi

    # Fix the empty key for path from gcsfs
    GCSFS_PYTHON_HOME="${ROOT_DIR}/../../gcsfs"
    GCSFS_CORE_FILE="${GCSFS_PYTHON_HOME}/core.py"
    if [ -f "$GCSFS_CORE_FILE" ]; then
        cp ${PATCHES_DIR}/gcsfs_core.py.patch ${GCSFS_CORE_FILE}
    fi

    # Fix the ECS RAM role authentication for path from ossfs
    OSSFS_PYTHON_HOME="${ROOT_DIR}/../../ossfs"
    OSSFS_CORE_FILE="${OSSFS_PYTHON_HOME}/core.py"
    if [ -f "${OSSFS_CORE_FILE}" ]; then
        cp ${PATCHES_DIR}/ossfs_core.py.patch ${OSSFS_CORE_FILE}
    fi

    # MLflow patches for Azure Data Lake Gen2
    MLFLOW_PYTHON_HOME="${ROOT_DIR}/../../mlflow"

    MLFLOW_ARTIFACT_REPOSITORY_REGISTRY_FILE="${MLFLOW_PYTHON_HOME}/store/artifact/artifact_repository_registry.py"
    if [ -f "${MLFLOW_ARTIFACT_REPOSITORY_REGISTRY_FILE}" ]; then
        cp ${PATCHES_DIR}/mlflow_store_artifact_artifact_repository_registry.py.patch ${MLFLOW_ARTIFACT_REPOSITORY_REGISTRY_FILE}
    fi

    MLFLOW_AZURE_BLOB_ARTIFACT_REPO_FILE="${MLFLOW_PYTHON_HOME}/store/artifact/azure_blob_artifact_repo.py"
    if [ -f "${MLFLOW_AZURE_BLOB_ARTIFACT_REPO_FILE}" ]; then
        cp ${PATCHES_DIR}/mlflow_store_artifact_azure_blob_artifact_repo.py.patch ${MLFLOW_AZURE_BLOB_ARTIFACT_REPO_FILE}
    fi

    MLFLOW_AZURE_DATA_LAKE_ARTIFACT_REPO_FILE="${MLFLOW_PYTHON_HOME}/store/artifact/azure_data_lake_artifact_repo.py"
    if [ -f "${MLFLOW_AZURE_DATA_LAKE_ARTIFACT_REPO_FILE}" ]; then
        cp ${PATCHES_DIR}/mlflow_store_artifact_azure_data_lake_artifact_repo.py.patch ${MLFLOW_AZURE_DATA_LAKE_ARTIFACT_REPO_FILE}
    fi
}

function prepare_database_schema() {
    DATABASE_NAME=mlflow
    if [ "${SQL_DATABASE_ENGINE}" == "mysql" ]; then
        mysql --host=${SQL_DATABASE_HOST} --port=${SQL_DATABASE_PORT} --user=${SQL_DATABASE_USERNAME} --password=${SQL_DATABASE_PASSWORD}  -e "
                CREATE DATABASE IF NOT EXISTS ${DATABASE_NAME};" > ${MLFLOW_HOME}/logs/configure.log
    else
        # Use psql to create the database
        echo "SELECT 'CREATE DATABASE ${DATABASE_NAME}' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${DATABASE_NAME}')\gexec" | PGPASSWORD=${SQL_DATABASE_PASSWORD} \
            psql \
            --host=${SQL_DATABASE_HOST} \
            --port=${SQL_DATABASE_PORT} \
            --username=${SQL_DATABASE_USERNAME} > ${MLFLOW_HOME}/logs/configure.log
    fi
    # Future improvement: mlflow db upgrade [db_uri]
}

function configure_ai() {
    # Do necessary configurations for AI runtime
    prepare_base_conf
    cd $output_dir

    MLFLOW_CONF_DIR=${MLFLOW_HOME}/conf
    mkdir -p ${MLFLOW_CONF_DIR}

    update_mlflow_server_config
    update_api_credential_for_provider

    cp $output_dir/mlflow ${MLFLOW_CONF_DIR}/mlflow

    if [ "$IS_HEAD_NODE" == "true" ]; then
        # Preparing database if external database used
        if [ "${SQL_DATABASE}" == "true" ] && [ "$AI_WITH_SQL_DATABASE" != "false" ]; then
            prepare_database_schema
        fi
    fi

    patch_libraries
}

set_head_option "$@"
set_head_address
configure_system_folders
configure_ai

exit 0
