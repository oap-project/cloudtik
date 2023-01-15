#!/bin/bash

if [ ! -n "${HADOOP_HOME}" ]; then
    echo "HADOOP_HOME environment variable is not set."
    exit 1
fi

if [ ! -n "${SPARK_HOME}" ]; then
    echo "SPARK_HOME environment variable is not set."
    exit 1
fi

export CLOUD_FS_MOUNT_PATH=/cloudtik/fs

function mount_s3_fs() {
    if [ -z "${AWS_S3_BUCKET}" ]; then
        echo "AWS_S3A_BUCKET environment variable is not set."
        return
    fi

    IAM_FLAG=""
    if [ -z "${AWS_S3_ACCESS_KEY_ID}" ] || [ -z "${AWS_S3_SECRET_ACCESS_KEY}" ]; then
        IAM_FLAG="-o iam_role=auto"
    fi

    mkdir -p ${CLOUD_FS_MOUNT_PATH}
    echo "Mounting S3 bucket ${AWS_S3_BUCKET} to ${CLOUD_FS_MOUNT_PATH}..."
    s3fs ${AWS_S3_BUCKET} -o use_cache=/tmp -o mp_umask=002 -o multireq_max=5 ${IAM_FLAG} ${CLOUD_FS_MOUNT_PATH} > /dev/null
}

function mount_azure_blob_fs() {
    if [ -z "${AZURE_CONTAINER}" ]; then
        echo "AZURE_CONTAINER environment variable is not set."
        return
    fi

    if [ -z "${AZURE_MANAGED_IDENTITY_CLIENT_ID}" ]; then
        echo "AZURE_MANAGED_IDENTITY_CLIENT_ID environment variable is not set."
        return
    fi

    if [ -z "${AZURE_STORAGE_ACCOUNT}" ]; then
        echo "AZURE_STORAGE_ACCOUNT environment variable is not set."
        return
    fi

    #Use a ramdisk for the temporary path
    sudo mkdir /mnt/ramdisk
    sudo mount -t tmpfs -o size=16g tmpfs /mnt/ramdisk
    sudo mkdir /mnt/ramdisk/blobfusetmp
    sudo chown $(whoami) /mnt/ramdisk/blobfusetmp

    mkdir -p ${CLOUD_FS_MOUNT_PATH}
    echo "Mounting Azure blob container ${AZURE_CONTAINER}@${AZURE_STORAGE_ACCOUNT} to ${CLOUD_FS_MOUNT_PATH}..."
    blobfuse ${CLOUD_FS_MOUNT_PATH} --tmp-path=/mnt/ramdisk/blobfusetmp --config-file=${USER_HOME}/fuse_connection.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120 > /dev/null
}

function mount_gcs_fs() {
    if [ ! -n "${GCP_GCS_BUCKET}" ]; then
        echo "GCP_GCS_BUCKET environment variable is not set."
        return
    fi

    mkdir -p ${CLOUD_FS_MOUNT_PATH}
    echo "Mounting GCS bucket ${GCP_GCS_BUCKET} to ${CLOUD_FS_MOUNT_PATH}..."
    gcsfuse ${GCP_GCS_BUCKET} ${CLOUD_FS_MOUNT_PATH} > /dev/null
}

function mount_cloud_fs() {
    cloud_storage_provider="none"
    if [ "$AWS_CLOUD_STORAGE" == "true" ]; then
        mount_s3_fs
    elif [ "$AZURE_CLOUD_STORAGE" == "true" ]; then
        mount_azure_blob_fs
    elif [ "$GCP_CLOUD_STORAGE" == "true" ]; then
        mount_gcs_fs
    fi
}

function unmount_cloud_fs() {
    echo "Unmounting cloud fs at ${CLOUD_FS_MOUNT_PATH}..."
    fusermount -u ${CLOUD_FS_MOUNT_PATH}
}

case "$1" in
start-head)
    mount_cloud_fs
    echo "Starting Resource Manager..."
    $HADOOP_HOME/bin/yarn --daemon start resourcemanager
    echo "Starting Spark History Server..."
    export SPARK_LOCAL_IP=${CLOUDTIK_NODE_IP}; $SPARK_HOME/sbin/start-history-server.sh > /dev/null
    echo "Starting Jupyter..."
    nohup jupyter lab --no-browser > /tmp/logs/jupyterlab.log 2>&1 &
    ;;
stop-head)
    $HADOOP_HOME/bin/yarn --daemon stop resourcemanager
    $SPARK_HOME/sbin/stop-history-server.sh
    # workaround for stopping jupyter when password being set
    JUPYTER_PID=$(pgrep jupyter)
    if [ -n "$JUPYTER_PID" ]; then
      echo "Stopping Jupyter..."
      kill $JUPYTER_PID >/dev/null 2>&1
    fi
    unmount_cloud_fs
    ;;
start-worker)
    mount_cloud_fs
    $HADOOP_HOME/bin/yarn --daemon start nodemanager
    ;;
stop-worker)
    $HADOOP_HOME/bin/yarn --daemon stop nodemanager
    unmount_cloud_fs
    ;;
-h|--help)
    echo "Usage: $0 start-head|stop-head|start-worker|stop-worker" >&2
    ;;
*)
    echo "Usage: $0 start-head|stop-head|start-worker|stop-worker" >&2
    ;;
esac

exit 0
