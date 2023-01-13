#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h::p: -l head::,node_ip_address::,head_address:: -- "$@")
eval set -- "${args}"

IS_HEAD_NODE=false
USER_HOME=/home/$(whoami)

while true
do
    case "$1" in
    --head)
        IS_HEAD_NODE=true
        ;;
    --node_ip_address)
        NODE_IP_ADDRESS=$2
        shift
        ;;
    -h|--head_address)
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

function prepare_base_conf() {
    output_dir=/tmp/ml/conf
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

function configure_ml() {
    # Do necessary configurations for Machine Learning
    prepare_base_conf
    cd $output_dir

    HOROVOD_PYTHON_HOME="${ROOT_DIR}/../../horovod"

    if [ $IS_HEAD_NODE == "true" ];then
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

        # Fix the Horovod bug for handling network interfaces of loopback
        HOROVOD_RUNNER_DRIVER_SERVICE_FILE="${HOROVOD_PYTHON_HOME}/runner/driver/driver_service.py"
        if [ -f "$HOROVOD_RUNNER_DRIVER_SERVICE_FILE" ]; then
           cp $output_dir/horovod_runner_driver_service.py.patch ${HOROVOD_RUNNER_DRIVER_SERVICE_FILE}
        fi
    fi

    # Improve Horovod on Spark for support MPICH and IMPI
    HOROVOD_SPARK_MPIRUN_EXEC_FN_FILE="${HOROVOD_PYTHON_HOME}/spark/task/mpirun_exec_fn.py"
    if [ -f "${HOROVOD_SPARK_MPIRUN_EXEC_FN_FILE}" ]; then
       cp $output_dir/horovod_spark_mpirun_exec_fn.py.patch ${HOROVOD_SPARK_MPIRUN_EXEC_FN_FILE}
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
}

set_head_address
configure_ml

exit 0
