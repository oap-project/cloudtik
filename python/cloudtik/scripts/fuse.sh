#!/bin/bash

args=$(getopt -a -o p: -l provider:,aws_s3a_bucket::,s3a_access_key::,s3a_secret_key::,project_id::,gcp_gcs_bucket::,fs_gs_auth_service_account_email::,fs_gs_auth_service_account_private_key_id::,fs_gs_auth_service_account_private_key::,azure_storage_kind::,azure_storage_account::,azure_container::,azure_account_key:: -- "$@")
eval set -- "${args}"

while true
do
    case "$1" in
    -p|--provider)
        provider=$2
        shift
        ;;
    --aws_s3a_bucket)
        AWS_S3A_BUCKET=$2
        shift
        ;;
    --s3a_access_key)
        FS_S3A_ACCESS_KEY=$2
        shift
        ;;
    --s3a_secret_key)
        FS_S3A_SECRET_KEY=$2
        shift
        ;;
    --project_id)
        PROJECT_ID=$2
        shift
        ;;
    --gcp_gcs_bucket)
        GCP_GCS_BUCKET=$2
        shift
        ;;
    --fs_gs_auth_service_account_email)
        FS_GS_AUTH_SERVICE_ACCOUNT_EMAIL=$2
        shift
        ;;
    --fs_gs_auth_service_account_private_key_id)
        FS_GS_AUTH_SERVICE_ACCOUNT_PRIVATE_KEY_ID=$2
        shift
        ;;
    --fs_gs_auth_service_account_private_key)
        FS_GS_AUTH_SERVICE_ACCOUNT_PRIVATE_KEY=$2
        shift
        ;;
    --azure_storage_kind)
        AZURE_STORAGE_KIND=$2
        shift
        ;;
    --azure_storage_account)
        AZURE_STORAGE_ACCOUNT=$2
        shift
        ;;
    --azure_container)
        AZURE_CONTAINER=$2
        shift
        ;;
    --azure_account_key)
        AZURE_ACCOUNT_KEY=$2
        shift
        ;;
    --)
        shift
        break
        ;;
    esac
    shift
done


export USER_HOME=/home/$(whoami)
export MOUNT_PATH=$USER_HOME/share


function s3_fuse() {
    if [ ! -n "${AWS_S3A_BUCKET}" ]; then
        echo "AWS_S3A_BUCKET environment variable is not set."
        exit 1
    fi

    if [ ! -n "${FS_S3A_ACCESS_KEY}" ]; then
        echo "FS_S3A_ACCESS_KEY environment variable is not set."
        exit 1
    fi

    if [ ! -n "${FS_S3A_SECRET_KEY}" ]; then
        echo "FS_S3A_SECRET_KEY environment variable is not set."
        exit 1
    fi

    sudo apt-get update
    sudo apt-get install automake autotools-dev fuse g++ git libcurl4-gnutls-dev libfuse-dev libssl-dev libxml2-dev make pkg-config

    cd $USER_HOME
    git clone https://github.com/s3fs-fuse/s3fs-fuse.git
    cd s3fs-fuse
    ./autogen.sh
    ./configure --prefix=/usr --with-openssl
    make
    sudo make install

    echo "${FS_S3A_ACCESS_KEY}:${FS_S3A_SECRET_KEY}" > ${$USER_HOME}/.passwd-s3fs
    chmod 600 ${$USER_HOME}/.passwd-s3fs

    mkdir -p ${MOUNT_PATH}
    s3fs ${AWS_S3A_BUCKET} -o use_cache=/tmp -o mp_umask=002 -o multireq_max=5 ${MOUNT_PATH}
}


function blob_fuse() {
    if [ ! -n "${AZURE_CONTAINER}" ]; then
        echo "AZURE_CONTAINER environment variable is not set."
        exit 1
    fi

    if [ ! -n "${AZURE_ACCOUNT_KEY}" ]; then
        echo "AZURE_ACCOUNT_KEY environment variable is not set."
        exit 1
    fi

    if [ ! -n "${AZURE_STORAGE_ACCOUNT}" ]; then
        echo "AZURE_STORAGE_ACCOUNT environment variable is not set."
        exit 1
    fi

    #Install blobfuse
    wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb
    sudo dpkg -i packages-microsoft-prod.deb
    sudo apt-get update
    sudo apt-get install blobfuse
    #Use a ramdisk for the temporary path
    sudo mkdir /mnt/ramdisk
    sudo mount -t tmpfs -o size=16g tmpfs /mnt/ramdisk
    sudo mkdir /mnt/ramdisk/blobfusetmp
    sudo chown ubuntu /mnt/ramdisk/blobfusetmp


    echo "accountName ${AZURE_STORAGE_ACCOUNT}" > ${$USER_HOME}/fuse_connection.cfg
    echo "accountKey ${AZURE_ACCOUNT_KEY}" >> ${$USER_HOME}/fuse_connection.cfg
    echo "containerName ${AZURE_CONTAINER}" >> ${$USER_HOME}/fuse_connection.
    chmod 600 ${$USER_HOME}/fuse_connection.cfg
    mkdir -p ${MOUNT_PATH}
    blobfuse ${MOUNT_PATH} --tmp-path=/mnt/ramdisk/blobfusetmp  --config-file=${$USER_HOME}/fuse_connection.cfg  -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120

}


function gcs_fuse()

    mkdir -p ${MOUNT_PATH}

}


function mount_storage_for_cloudtik() {
    if [ "$provider" == "aws" ]; then
      s3_fuse
    fi

    if [ "$provider" == "gcp" ]; then
      gcs_fuse
    fi

    if [ "$provider" == "azure" ]; then
      blob_fuse
    fi
}

mount_storage_for_cloudtik