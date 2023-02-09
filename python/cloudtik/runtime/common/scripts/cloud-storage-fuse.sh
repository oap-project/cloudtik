#!/bin/bash

# Assumptions for using the functions of this script:
# 1. $XXX_CLOUD_STORAGE variable is set if the cloud storage is in use
# 2. Credential values are exported through the environment variables through provider.with_environment_variables.
# 3. USER_HOME is set to the current user home.
# 4. For service functions, CLOUD_FS_MOUNT_PATH is set to the target path for mounting.
#    For service functions for local hdfs, HEAD_ADDRESS is set.
#    For service functions for remote hdfs, HDFS_NAMENODE_URI is set

# Configuring functions
function configure_fuse_options() {
    FUSE_CONF_FILE="/etc/fuse.conf"
    FIND_STR="^user_allow_other"
    if [ `grep -c "$FIND_STR" $FUSE_CONF_FILE` -eq '0' ];then
        sudo sed -i '$auser_allow_other' $FUSE_CONF_FILE
    fi
}

function configure_local_hdfs_fs() {
    configure_fuse_options
}

function configure_hdfs_fs() {
    configure_fuse_options
}

function configure_s3_fs() {
    if [ -z "${AWS_S3_BUCKET}" ]; then
        echo "AWS_S3A_BUCKET environment variable is not set."
        return
    fi

    if [ ! -z "${AWS_S3_ACCESS_KEY_ID}" ] && [ ! -z "${AWS_S3_SECRET_ACCESS_KEY}" ]; then
        echo "${AWS_S3_ACCESS_KEY_ID}:${AWS_S3_SECRET_ACCESS_KEY}" > ${USER_HOME}/.passwd-s3fs
        chmod 600 ${USER_HOME}/.passwd-s3fs
    fi
}

function configure_azure_blob_fs() {
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

    fuse_connection_cfg=${USER_HOME}/fuse_connection.cfg
    echo "accountName ${AZURE_STORAGE_ACCOUNT}" > ${fuse_connection_cfg}
    echo "authType MSI" >> ${fuse_connection_cfg}
    echo "identityClientId ${AZURE_MANAGED_IDENTITY_CLIENT_ID}" >> ${fuse_connection_cfg}
    echo "containerName ${AZURE_CONTAINER}" >> ${fuse_connection_cfg}
    chmod 600 ${fuse_connection_cfg}
}

function configure_gcs_fs() {
    if [ -z "${GCP_GCS_BUCKET}" ]; then
        echo "GCP_GCS_BUCKET environment variable is not set."
        return
    fi
}

function configure_aliyun_oss_fs() {
    if [ -z "${ALIYUN_OSS_BUCKET}" ]; then
        echo "ALIYUN_OSS_BUCKET environment variable is not set."
        return
    fi

    if [ ! -z "${ALIYUN_OSS_ACCESS_KEY_ID}" ] && [ ! -z "${ALIYUN_OSS_ACCESS_KEY_SECRET}" ]; then
        echo "${ALIYUN_OSS_BUCKET}:${ALIYUN_OSS_ACCESS_KEY_ID}:${ALIYUN_OSS_ACCESS_KEY_SECRET}" > ${USER_HOME}/.passwd-ossfs
        chmod 600 ${USER_HOME}/.passwd-ossfs
    fi
}

function configure_cloud_fs() {
    sudo mkdir /cloudtik
    sudo chown $(whoami) /cloudtik
    if [ "$HDFS_ENABLED" == "true" ]; then
        configure_local_hdfs_fs
    else
        if [ "$HDFS_STORAGE" == "true" ]; then
            configure_hdfs_fs
        elif [ "$AWS_CLOUD_STORAGE" == "true" ]; then
            configure_s3_fs
        elif [ "$AZURE_CLOUD_STORAGE" == "true" ]; then
            configure_azure_blob_fs
        elif [ "$GCP_CLOUD_STORAGE" == "true" ]; then
            configure_gcs_fs
        elif [ "$ALIYUN_CLOUD_STORAGE" == "true" ]; then
            configure_aliyun_oss_fs
        fi
    fi
}

# Installing functions
function install_hdfs_fuse() {
    if ! type fuse_dfs >/dev/null 2>&1;then
      arch=$(uname -m)
      sudo wget -q --show-progress https://d30257nes7d4fq.cloudfront.net/downloads/hadoop/fuse_dfs-${HADOOP_VERSION}-${arch} -O /usr/bin/fuse_dfs
      sudo wget -q --show-progress https://d30257nes7d4fq.cloudfront.net/downloads/hadoop/fuse_dfs_wrapper-${HADOOP_VERSION}.sh -O /usr/bin/fuse_dfs_wrapper.sh
      sudo chmod +x /usr/bin/fuse_dfs
      sudo chmod +x /usr/bin/fuse_dfs_wrapper.sh
    fi
}

function install_s3_fuse() {
    if ! type s3fs >/dev/null 2>&1;then
      echo "Installing S3 Fuse..."
      sudo apt-get -qq update -y > /dev/null
      sudo apt-get install -qq s3fs -y > /dev/null
    fi
}

function install_azure_blob_fuse() {
    if ! type blobfuse >/dev/null 2>&1; then
      echo "Installing Azure Blob Fuse..."
      wget -q -N https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb
      sudo dpkg -i packages-microsoft-prod.deb > /dev/null
      sudo apt-get -qq update -y > /dev/null
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -qq blobfuse -y > /dev/null
    fi
}

function install_gcs_fuse() {
    if ! type gcsfuse >/dev/null 2>&1; then
      echo "Installing GCS Fuse..."
      echo "deb http://packages.cloud.google.com/apt gcsfuse-bionic main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list > /dev/null
      wget -O - -q https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
      sudo apt-get -qq update -y > /dev/null
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -qq gcsfuse -y > /dev/null
    fi
}

function install_aliyun_oss_fuse() {
    if ! type ossfs >/dev/null 2>&1; then
      echo "Installing Aliyun OSS Fuse..."
      OSS_PACKAGE="ossfs_1.80.7_ubuntu20.04_amd64.deb"
      wget -q -N https://gosspublic.alicdn.com/ossfs/${OSS_PACKAGE}
      sudo apt-get -qq update -y > /dev/null
      sudo apt-get install -qq gdebi-core -y > /dev/null
      sudo gdebi --q --n ${OSS_PACKAGE} > /dev/null
      rm ${OSS_PACKAGE}
    fi
}

function install_cloud_fuse() {
    install_hdfs_fuse
    if [ "$AWS_CLOUD_STORAGE" == "true" ]; then
        install_s3_fuse
    elif [ "$AZURE_CLOUD_STORAGE" == "true" ]; then
        install_azure_blob_fuse
    elif [ "$GCP_CLOUD_STORAGE" == "true" ]; then
        install_gcs_fuse
    elif [ "$ALIYUN_CLOUD_STORAGE" == "true" ]; then
        install_aliyun_oss_fuse
    fi
}

# Service functions

function mount_local_hdfs_fs() {
    fs_default_dir="dfs://${HEAD_ADDRESS}:9000"
    # Mount local hdfs fuse here
    mkdir -p ${CLOUD_FS_MOUNT_PATH}
    echo "Mounting HDFS ${fs_default_dir} to ${CLOUD_FS_MOUNT_PATH}..."
    fuse_dfs_wrapper.sh -oinitchecks ${fs_default_dir} ${CLOUD_FS_MOUNT_PATH}
}

function mount_hdfs_fs() {
    fs_default_dir="${HDFS_NAMENODE_URI:1}"
    # Mount remote hdfs fuse here
    mkdir -p ${CLOUD_FS_MOUNT_PATH}
    echo "Mounting HDFS ${fs_default_dir} to ${CLOUD_FS_MOUNT_PATH}..."
    fuse_dfs_wrapper.sh -oinitchecks ${fs_default_dir} ${CLOUD_FS_MOUNT_PATH}
}

function mount_s3_fs() {
    if [ -z "${AWS_S3_BUCKET}" ]; then
        echo "AWS_S3_BUCKET environment variable is not set."
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

function mount_aliyun_oss_fs() {
    if [ -z "${ALIYUN_OSS_BUCKET}" ]; then
        echo "ALIYUN_OSS_BUCKET environment variable is not set."
        return
    fi

    PASSWD_FILE_FLAG=""
    RAM_ROLE_FLAG=""
    if [ ! -z "${ALIYUN_OSS_ACCESS_KEY_ID}" ] && [ ! -z "${ALIYUN_OSS_ACCESS_KEY_SECRET}" ]; then
        PASSWD_FILE_FLAG="-o passwd_file=${USER_HOME}/.passwd-ossfs"
    else
        RAM_ROLE_FLAG="-o ram_role=${ALIYUN_OSS_INSTANCE_ROLE_NAME}"
    fi

    mkdir -p ${CLOUD_FS_MOUNT_PATH}
    echo "Mounting Aliyun OSS bucket ${ALIYUN_OSS_BUCKET} to ${CLOUD_FS_MOUNT_PATH}..."
    # TODO: Endpoint setup for ECS for network going internally (for example, oss-cn-hangzhou-internal.aliyuncs.com)
    ossfs ${ALIYUN_OSS_BUCKET} ${CLOUD_FS_MOUNT_PATH} -o use_cache=/tmp -o mp_umask=002 ${PASSWD_FILE_FLAG} ${RAM_ROLE_FLAG} > /dev/null
}

function mount_cloud_fs() {
    if [ "$HDFS_ENABLED" == "true" ]; then
        mount_local_hdfs_fs
    else
        if [ ! -z  "${HDFS_NAMENODE_URI}" ];then
            HDFS_STORAGE="true"
        else
            HDFS_STORAGE="false"
        fi
        if [ "$HDFS_STORAGE" == "true" ]; then
            mount_hdfs_fs
        elif [ "$AWS_CLOUD_STORAGE" == "true" ]; then
            mount_s3_fs
        elif [ "$AZURE_CLOUD_STORAGE" == "true" ]; then
            mount_azure_blob_fs
        elif [ "$GCP_CLOUD_STORAGE" == "true" ]; then
            mount_gcs_fs
        elif [ "$ALIYUN_CLOUD_STORAGE" == "true" ]; then
            mount_aliyun_oss_fs
        fi
    fi
}

function unmount_cloud_fs() {
    echo "Unmounting cloud fs at ${CLOUD_FS_MOUNT_PATH}..."
    fusermount -u ${CLOUD_FS_MOUNT_PATH}
}
