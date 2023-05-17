#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head::,node_ip_address:,head_address: -- "$@")
eval set -- "${args}"

IS_HEAD_NODE=false
USER_HOME=/home/$(whoami)

while true
do
    case "$1" in
    -h|--head)
        IS_HEAD_NODE=true
        ;;
    --node_ip_address)
        NODE_IP_ADDRESS=$2
        shift
        ;;
    --head_address)
        HEAD_ADDRESS=$2
        shift
        ;;
    --)
        shift
        break
        ;;
    esac
    shift
done


# API cloud credential configuration functions
. "$ROOT_DIR"/common/scripts/api-credential.sh

function prepare_base_conf() {
    output_dir=/tmp/ai/conf
    rm -rf  $output_dir
    mkdir -p $output_dir
    source_dir=$(cd $(dirname ${BASH_SOURCE[0]})/..;pwd)/conf
    cp -r $source_dir/* $output_dir
}

function set_head_address() {
    if [ $IS_HEAD_NODE == "true" ]; then
        if [ ! -n "${NODE_IP_ADDRESS}" ]; then
            HEAD_ADDRESS=$(hostname -I | awk '{print $1}')
        else
            HEAD_ADDRESS=${NODE_IP_ADDRESS}
        fi
    else
        if [ ! -n "${HEAD_ADDRESS}" ]; then
            # Error: no head address passed
            echo "Error: head ip address should be passed."
            exit 1
        fi
    fi
}

function configure_system_folders() {
    # Create dirs for data
    MLFLOW_DATA=$RUNTIME_PATH/mlflow
    mkdir -p ${MLFLOW_DATA}/logs
    mkdir -p ${MLFLOW_DATA}/mlruns
}

function patch_libraries() {
    HOROVOD_PYTHON_HOME="${ROOT_DIR}/../../horovod"

    # Fix the Horovod on Spark bug for handling network interfaces of loopback
    HOROVOD_SPARK_GLOO_RUN_FILE="${HOROVOD_PYTHON_HOME}/spark/gloo_run.py"
    if [ -f "${HOROVOD_SPARK_GLOO_RUN_FILE}" ]; then
       cp $output_dir/horovod_spark_gloo_run.py.patch ${HOROVOD_SPARK_GLOO_RUN_FILE}
    fi

    # Improve Horovod on Spark for support MPICH and IMPI
    HOROVOD_SPARK_MPI_RUN_FILE="${HOROVOD_PYTHON_HOME}/spark/mpi_run.py"
    if [ -f "${HOROVOD_SPARK_MPI_RUN_FILE}" ]; then
       cp $output_dir/horovod_spark_mpi_run.py.patch ${HOROVOD_SPARK_MPI_RUN_FILE}
    fi

    # Fix the Horovod driver NIC issue
    HOROVOD_SPARK_RUNNER_FILE="${HOROVOD_PYTHON_HOME}/spark/runner.py"
    if [ -f "${HOROVOD_SPARK_RUNNER_FILE}" ]; then
       cp $output_dir/horovod_spark_runner.py.patch ${HOROVOD_SPARK_RUNNER_FILE}
    fi

    HOROVOD_SPARK_DRIVER_DRIVER_SERVICE_FILE="${HOROVOD_PYTHON_HOME}/spark/driver/driver_service.py"
    if [ -f "${HOROVOD_SPARK_DRIVER_DRIVER_SERVICE_FILE}" ]; then
       cp $output_dir/horovod_spark_driver_driver_service.py.patch ${HOROVOD_SPARK_DRIVER_DRIVER_SERVICE_FILE}
    fi

    # CloudTik remote command execution for Gloo
    HOROVOD_RUNNER_UTIL_REMOTE_FILE="${HOROVOD_PYTHON_HOME}/runner/util/remote.py"
    if [ -f "$HOROVOD_RUNNER_UTIL_REMOTE_FILE" ]; then
       cp $output_dir/horovod_runner_util_remote.py.patch ${HOROVOD_RUNNER_UTIL_REMOTE_FILE}
    fi

    # Fix the remote command quote handling
    HOROVOD_RUNNER_GLOO_RUN_FILE="${HOROVOD_PYTHON_HOME}/runner/gloo_run.py"
    if [ -f "$HOROVOD_RUNNER_GLOO_RUN_FILE" ]; then
       cp $output_dir/horovod_runner_gloo_run.py.patch ${HOROVOD_RUNNER_GLOO_RUN_FILE}
    fi

    # CloudTik remote command execution for MPI
    HOROVOD_RUNNER_MPI_RUN_FILE="${HOROVOD_PYTHON_HOME}/runner/mpi_run.py"
    if [ -f "$HOROVOD_RUNNER_MPI_RUN_FILE" ]; then
       cp $output_dir/horovod_runner_mpi_run.py.patch ${HOROVOD_RUNNER_MPI_RUN_FILE}
    fi

    # Fix the Horovod driver NIC issue
    HOROVOD_RUNNER_LAUNCH_FILE="${HOROVOD_PYTHON_HOME}/runner/launch.py"
    if [ -f "${HOROVOD_RUNNER_LAUNCH_FILE}" ]; then
       cp $output_dir/horovod_runner_launch.py.patch ${HOROVOD_RUNNER_LAUNCH_FILE}
    fi

    HOROVOD_RUNNER_COMMON_SERVICE_DRIVER_SERVICE_FILE="${HOROVOD_PYTHON_HOME}/runner/common/service/driver_service.py"
    if [ -f "${HOROVOD_RUNNER_COMMON_SERVICE_DRIVER_SERVICE_FILE}" ]; then
       cp $output_dir/horovod_runner_common_service_driver_service.py.patch ${HOROVOD_RUNNER_COMMON_SERVICE_DRIVER_SERVICE_FILE}
    fi

    # Fix the Horovod bug for handling network interfaces of loopback
    HOROVOD_RUNNER_DRIVER_SERVICE_FILE="${HOROVOD_PYTHON_HOME}/runner/driver/driver_service.py"
    if [ -f "$HOROVOD_RUNNER_DRIVER_SERVICE_FILE" ]; then
       cp $output_dir/horovod_runner_driver_driver_service.py.patch ${HOROVOD_RUNNER_DRIVER_SERVICE_FILE}
    fi

    # Improve Horovod on Spark for support MPICH and IMPI
    HOROVOD_SPARK_MPIRUN_EXEC_FN_FILE="${HOROVOD_PYTHON_HOME}/spark/task/mpirun_exec_fn.py"
    if [ -f "${HOROVOD_SPARK_MPIRUN_EXEC_FN_FILE}" ]; then
       cp $output_dir/horovod_spark_task_mpirun_exec_fn.py.patch ${HOROVOD_SPARK_MPIRUN_EXEC_FN_FILE}
    fi

    HOROVOD_RAY_UTILS_FILE="${HOROVOD_PYTHON_HOME}/ray/utils.py"
    if [ -f "${HOROVOD_RAY_UTILS_FILE}" ]; then
       cp $output_dir/horovod_ray_utils.py.patch ${HOROVOD_RAY_UTILS_FILE}
    fi

    # Fix the Azure managed identity from adlfs
    ADLFS_PYTHON_HOME="${ROOT_DIR}/../../adlfs"
    ADLFS_SPEC_FILE="${ADLFS_PYTHON_HOME}/spec.py"
    if [ -f "$ADLFS_SPEC_FILE" ]; then
        cp $output_dir/adlfs_spec.py.patch ${ADLFS_SPEC_FILE}
    fi

    # Fix the empty key for path from gcsfs
    GCSFS_PYTHON_HOME="${ROOT_DIR}/../../gcsfs"
    GCSFS_CORE_FILE="${GCSFS_PYTHON_HOME}/core.py"
    if [ -f "$GCSFS_CORE_FILE" ]; then
        cp $output_dir/gcsfs_core.py.patch ${GCSFS_CORE_FILE}
    fi

    # Fix the ECS RAM role authentication for path from ossfs
    OSSFS_PYTHON_HOME="${ROOT_DIR}/../../ossfs"
    OSSFS_CORE_FILE="${OSSFS_PYTHON_HOME}/core.py"
    if [ -f "${OSSFS_CORE_FILE}" ]; then
        cp $output_dir/ossfs_core.py.patch ${OSSFS_CORE_FILE}
    fi

    # MLflow patches for Azure Data Lake Gen2
    MLFLOW_PYTHON_HOME="${ROOT_DIR}/../../mlflow"

    MLFLOW_ARTIFACT_REPOSITORY_REGISTRY_FILE="${MLFLOW_PYTHON_HOME}/store/artifact/artifact_repository_registry.py"
    if [ -f "${MLFLOW_ARTIFACT_REPOSITORY_REGISTRY_FILE}" ]; then
        cp $output_dir/mlflow_store_artifact_artifact_repository_registry.py.patch ${MLFLOW_ARTIFACT_REPOSITORY_REGISTRY_FILE}
    fi

    MLFLOW_AZURE_BLOB_ARTIFACT_REPO_FILE="${MLFLOW_PYTHON_HOME}/store/artifact/azure_blob_artifact_repo.py"
    if [ -f "${MLFLOW_AZURE_BLOB_ARTIFACT_REPO_FILE}" ]; then
        cp $output_dir/mlflow_store_artifact_azure_blob_artifact_repo.py.patch ${MLFLOW_AZURE_BLOB_ARTIFACT_REPO_FILE}
    fi

    MLFLOW_AZURE_DATA_LAKE_ARTIFACT_REPO_FILE="${MLFLOW_PYTHON_HOME}/store/artifact/azure_data_lake_artifact_repo.py"
    if [ -f "${MLFLOW_AZURE_DATA_LAKE_ARTIFACT_REPO_FILE}" ]; then
        cp $output_dir/mlflow_store_artifact_azure_data_lake_artifact_repo.py.patch ${MLFLOW_AZURE_DATA_LAKE_ARTIFACT_REPO_FILE}
    fi

    # Patch IPEX for integration
    IPEX_PYTHON_HOME="${ROOT_DIR}/../../intel_extension_for_pytorch"
    IPEX_CPU_LAUCNH_FILE="${IPEX_PYTHON_HOME}/cpu/launch.py"
    if [ -f "${IPEX_CPU_LAUCNH_FILE}" ]; then
        cp $output_dir/ipex_cpu_launch.py.patch ${IPEX_CPU_LAUCNH_FILE}
    fi
}

function prepare_database_schema() {
    DATABASE_NAME=mlflow
    mysql --host=${CLOUD_DATABASE_HOSTNAME} --port=${CLOUD_DATABASE_PORT} --user=${CLOUD_DATABASE_USERNAME} --password=${CLOUD_DATABASE_PASSWORD}  -e "
            CREATE DATABASE IF NOT EXISTS ${DATABASE_NAME};" > ${MLFLOW_DATA}/logs/configure.log

    # Future improvement: mlflow db upgrade [db_uri]
}

function configure_ai() {
    # Do necessary configurations for AI runtime
    prepare_base_conf
    cd $output_dir

    update_api_credential_for_provider

    if [ "$IS_HEAD_NODE" == "true" ]; then
        # Preparing database if external database used
        if [ "${CLOUD_DATABASE}" == "true" ] && [ "$AI_WITH_CLOUD_DATABASE" != "false" ]; then
            prepare_database_schema
        fi
    fi

    patch_libraries
}


set_head_address
configure_system_folders
configure_ai

exit 0
