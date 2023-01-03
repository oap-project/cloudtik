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
    if [ $IS_HEAD_NODE == "true" ];then
        # Fix the Horovod on Spark bug for handling network interfaces of loopback
        HOROVOD_PYTHON_HOME="${ROOT_DIR}/../../horovod"
        SPARK_GLOO_RUN_FILE="${HOROVOD_PYTHON_HOME}/spark/gloo_run.py"
        if [ -f "$SPARK_GLOO_RUN_FILE" ]; then
           cp $output_dir/horovod_gloo_run.py.patch ${SPARK_GLOO_RUN_FILE}
        fi
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


export MOUNT_PATH=$USER_HOME/share

function configure_s3_fuse() {
    if [ ! -n "${AWS_S3A_BUCKET}" ]; then
        echo "AWS_S3A_BUCKET environment variable is not set."
        return
    fi

    if [ ! -n "${FS_S3A_ACCESS_KEY}" ]; then
        echo "FS_S3A_ACCESS_KEY environment variable is not set."
        return
    fi

    if [ ! -n "${FS_S3A_SECRET_KEY}" ]; then
        echo "FS_S3A_SECRET_KEY environment variable is not set."
        return
    fi

    echo "${FS_S3A_ACCESS_KEY}:${FS_S3A_SECRET_KEY}" > ${USER_HOME}/.passwd-s3fs
    chmod 600 ${USER_HOME}/.passwd-s3fs

    mkdir -p ${MOUNT_PATH}
    s3fs ${AWS_S3A_BUCKET} -o use_cache=/tmp -o mp_umask=002 -o multireq_max=5 ${MOUNT_PATH}
}


function configure_blob_fuse() {
    if [ ! -n "${AZURE_CONTAINER}" ]; then
        echo "AZURE_CONTAINER environment variable is not set."
        return
    fi

    if [ ! -n "${AZURE_MANAGED_IDENTITY_CLIENT_ID}" ]; then
        echo "AZURE_MANAGED_IDENTITY_CLIENT_ID environment variable is not set."
        return
    fi

    if [ ! -n "${AZURE_STORAGE_ACCOUNT}" ]; then
        echo "AZURE_STORAGE_ACCOUNT environment variable is not set."
        return
    fi
    #Use a ramdisk for the temporary path
    sudo mkdir /mnt/ramdisk
    sudo mount -t tmpfs -o size=16g tmpfs /mnt/ramdisk
    sudo mkdir /mnt/ramdisk/blobfusetmp
    sudo chown cloudtik /mnt/ramdisk/blobfusetmp


    echo "accountName ${AZURE_STORAGE_ACCOUNT}" > ${USER_HOME}/fuse_connection.cfg
    echo "authType MSI" >> ${USER_HOME}/fuse_connection.cfg
    echo "identityClientId ${AZURE_MANAGED_IDENTITY_CLIENT_ID}" >> ${USER_HOME}/fuse_connection.cfg
    echo "containerName ${AZURE_CONTAINER}" >> ${USER_HOME}/fuse_connection.cfg
    chmod 600 ${USER_HOME}/fuse_connection.cfg
    mkdir -p ${MOUNT_PATH}

    blobfuse ${MOUNT_PATH} --tmp-path=/mnt/ramdisk/blobfusetmp  --config-file=${USER_HOME}/fuse_connection.cfg  -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120

}


function configure_gcs_fuse() {
    if [ ! -n "${GCP_GCS_BUCKET}" ]; then
        echo "GCP_GCS_BUCKET environment variable is not set."
        return
    fi
    mkdir -p ${MOUNT_PATH}
    gcsfuse ${GCP_GCS_BUCKET} ${MOUNT_PATH}
}

function mount_storage_for_cloudtik() {
    cloud_storage_provider="none"
    if [ "$AWS_CLOUD_STORAGE" == "true" ]; then
        configure_s3_fuse
    elif [ "$AZURE_CLOUD_STORAGE" == "true" ]; then
        configure_blob_fuse
    elif [ "$GCP_CLOUD_STORAGE" == "true" ]; then
        configure_gcs_fuse
    fi
}

set_head_address
configure_ml
mount_storage_for_cloudtik

exit 0
