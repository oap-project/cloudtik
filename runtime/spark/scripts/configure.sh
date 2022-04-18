#!/bin/bash


args=$(getopt -a -o h::p: -l head::,fuse_flag::,head_address::,provider:,aws_s3_bucket::,aws_s3_access_key_id::,aws_s3_secret_access_key::,project_id::,gcs_bucket::,gcs_service_account_client_email::,gcs_service_account_private_key_id::,gcs_service_account_private_key::,azure_storage_type::,azure_storage_account::,azure_container::,azure_account_key:: -- "$@")
eval set -- "${args}"

IS_HEAD_NODE=false
FUSE_FLAG=false
export USER_HOME=/home/$(whoami)
export MOUNT_PATH=$USER_HOME/share

while true
do
    case "$1" in
    --head)
        IS_HEAD_NODE=true
        ;;
    --fuse_flag)
        FUSE_FLAG=true
        ;;
    -h|--head_address)
        HEAD_ADDRESS=$2
        shift
        ;;
    -p|--provider)
        provider=$2
        shift
        ;;
    --aws_s3_bucket)
        AWS_S3_BUCKET=$2
        shift
        ;;
    --aws_s3_access_key_id)
        AWS_S3_ACCESS_KEY_ID=$2
        shift
        ;;
    --aws_s3_secret_access_key)
        AWS_S3_SECRET_ACCESS_KEY=$2
        shift
        ;;
    --project_id)
        PROJECT_ID=$2
        shift
        ;;
    --gcs_bucket)
        GCS_BUCKET=$2
        shift
        ;;
    --gcs_service_account_client_email)
        GCS_SERVICE_ACCOUNT_CLIENT_EMAIL=$2
        shift
        ;;
    --gcs_service_account_private_key_id)
        GCS_SERVICE_ACCOUNT_PRIVATE_KEY_ID=$2
        shift
        ;;
    --gcs_service_account_private_key)
        GCS_SERVICE_ACCOUNT_PRIVATE_KEY=$2
        shift
        ;;
    --azure_storage_type)
        AZURE_STORAGE_TYPE=$2
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

function prepare_base_conf() {
    source_dir=$(cd $(dirname ${BASH_SOURCE[0]})/..;pwd)/conf
    output_dir=/tmp/spark/conf
    rm -rf  $output_dir
    mkdir -p $output_dir
    cp -r $source_dir/* $output_dir
}

function check_spark_installed() {
    if [ ! -n "${HADOOP_HOME}" ]; then
        echo "HADOOP_HOME environment variable is not set."
        exit 1
    fi

    if [ ! -n "${SPARK_HOME}" ]; then
        echo "SPARK_HOME environment variable is not set."
        exit 1
    fi
}

function configure_system_folders() {
    # Create logs in tmp for any application logs to include
    mkdir -p /tmp/logs
}

function set_head_address() {
    if [ ! -n "${HEAD_ADDRESS}" ]; then
	    HEAD_ADDRESS=$(hostname -I | awk '{print $1}')
	  fi
}

function set_resources_for_spark() {
    # For nodemanager
    total_memory=$(awk '($1 == "MemTotal:"){print $2/1024*0.8}' /proc/meminfo)
    total_memory=${total_memory%.*}
    total_vcores=$(cat /proc/cpuinfo | grep processor | wc -l)

    # For Head Node
    if [ $IS_HEAD_NODE == "true" ];then
        spark_executor_cores=$(cat ~/cloudtik_bootstrap_config.yaml | jq '."runtime"."spark"."spark_executor_resource"."spark_executor_cores"')
        spark_executor_memory=$(cat ~/cloudtik_bootstrap_config.yaml | jq '."runtime"."spark"."spark_executor_resource"."spark_executor_memory"')M
        spark_driver_memory=$(cat ~/cloudtik_bootstrap_config.yaml | jq '."runtime"."spark"."spark_executor_resource"."spark_driver_memory"')M
        yarn_container_maximum_vcores=$(cat ~/cloudtik_bootstrap_config.yaml | jq '."runtime"."spark"."yarn_container_resource"."yarn_container_maximum_vcores"')
        yarn_container_maximum_memory=$(cat ~/cloudtik_bootstrap_config.yaml | jq '."runtime"."spark"."yarn_container_resource"."yarn_container_maximum_memory"')
    fi
}

function update_config_for_aws() {
    sed -i "s#{%aws.s3a.bucket%}#${AWS_S3_BUCKET}#g" `grep "{%aws.s3a.bucket%}" -rl ./`
    sed -i "s#{%fs.s3a.access.key%}#${AWS_S3_ACCESS_KEY_ID}#g" `grep "{%fs.s3a.access.key%}" -rl ./`
    sed -i "s#{%fs.s3a.secret.key%}#${AWS_S3_SECRET_ACCESS_KEY}#g" `grep "{%fs.s3a.secret.key%}" -rl ./`

    # event log dir
    if [ -z "${AWS_S3_BUCKET}" ]; then
        event_log_dir="file:///tmp/spark-events"
    else
        event_log_dir="s3a://${AWS_S3_BUCKET}/shared/spark-events"
    fi
    sed -i "s!{%spark.eventLog.dir%}!${event_log_dir}!g" `grep "{%spark.eventLog.dir%}" -rl ./`
}

function update_config_for_gcp() {
    sed -i "s#{%project_id%}#${PROJECT_ID}#g" `grep "{%project_id%}" -rl ./`
    sed -i "s#{%gcs.bucket%}#${GCS_BUCKET}#g" `grep "{%gcs.bucket%}" -rl ./`
    sed -i "s#{%fs.gs.auth.service.account.email%}#${GCS_SERVICE_ACCOUNT_CLIENT_EMAIL}#g" `grep "{%fs.gs.auth.service.account.email%}" -rl ./`
    sed -i "s#{%fs.gs.auth.service.account.private.key.id%}#${GCS_SERVICE_ACCOUNT_PRIVATE_KEY_ID}#g" `grep "{%fs.gs.auth.service.account.private.key.id%}" -rl ./`
    sed -i "s#{%fs.gs.auth.service.account.private.key%}#${GCS_SERVICE_ACCOUNT_PRIVATE_KEY}#g" `grep "{%fs.gs.auth.service.account.private.key%}" -rl ./`

    # event log dir
    if [ -z "${GCS_BUCKET}" ]; then
        event_log_dir="file:///tmp/spark-events"
    else
        event_log_dir="gs://${GCS_BUCKET}/shared/spark-events"
    fi
    sed -i "s!{%spark.eventLog.dir%}!${event_log_dir}!g" `grep "{%spark.eventLog.dir%}" -rl ./`
}

function update_config_for_azure() {
    sed -i "s#{%azure.storage.account%}#${AZURE_STORAGE_ACCOUNT}#g" "$(grep "{%azure.storage.account%}" -rl ./)"
    sed -i "s#{%azure.container%}#${AZURE_CONTAINER}#g" "$(grep "{%azure.container%}" -rl ./)"
    sed -i "s#{%azure.account.key%}#${AZURE_ACCOUNT_KEY}#g" "$(grep "{%azure.account.key%}" -rl ./)"
    if [ $AZURE_STORAGE_TYPE == "blob" ];then
        scheme="wasbs"
        endpoint="blob"
        sed -i "s#{%azure.storage.scheme%}#${scheme}#g" "$(grep "{%azure.storage.scheme%}" -rl ./)"
        sed -i "s#{%storage.endpoint%}#${endpoint}#g" "$(grep "{%storage.endpoint%}" -rl ./)"
    elif [ $AZURE_STORAGE_TYPE == "datalake" ];then
        scheme="abfs"
        endpoint="dfs"
        sed -i "s#{%azure.storage.scheme%}#${scheme}#g" "$(grep "{%azure.storage.scheme%}" -rl ./)"
        sed -i "s#{%storage.endpoint%}#${endpoint}#g" "$(grep "{%storage.endpoint%}" -rl ./)"
        sed -i "s#{%auth.type%}#SharedKey#g" "$(grep "{%auth.type%}" -rl ./)"
    else
        endpoint=""
        echo "Error: Azure storage kind must be blob (Azure Blob Storage) or datalake (Azure Data Lake Storage Gen 2)"
    fi

    # event log dir
    if [ -z "${AZURE_CONTAINER}" ] || [ -z "$endpoint" ]; then
        event_log_dir="file:///tmp/spark-events"
    else
        event_log_dir="${scheme}://${AZURE_CONTAINER}@${AZURE_STORAGE_ACCOUNT}.${endpoint}.core.windows.net/shared/spark-events"
    fi
    sed -i "s!{%spark.eventLog.dir%}!${event_log_dir}!g" `grep "{%spark.eventLog.dir%}" -rl ./`
}

function update_config_for_cloud() {
    if [ "$provider" == "aws" ]; then
      update_config_for_aws
    fi

    if [ "$provider" == "gcp" ]; then
      update_config_for_gcp
    fi

    if [ "$provider" == "azure" ]; then
      update_config_for_azure
    fi
}

function update_spark_runtime_config() {
    if [ $IS_HEAD_NODE == "true" ];then
        sed -i "s/{%yarn.scheduler.maximum-allocation-mb%}/${yarn_container_maximum_memory}/g" `grep "{%yarn.scheduler.maximum-allocation-mb%}" -rl ./`
        sed -i "s/{%yarn.nodemanager.resource.memory-mb%}/${yarn_container_maximum_memory}/g" `grep "{%yarn.nodemanager.resource.memory-mb%}" -rl ./`
        sed -i "s/{%yarn.nodemanager.resource.cpu-vcores%}/${yarn_container_maximum_vcores}/g" `grep "{%yarn.nodemanager.resource.cpu-vcores%}" -rl ./`
        sed -i "s/{%yarn.scheduler.maximum-allocation-vcores%}/${yarn_container_maximum_vcores}/g" `grep "{%yarn.scheduler.maximum-allocation-vcores%}" -rl ./`
        sed -i "s/{%spark.executor.cores%}/${spark_executor_cores}/g" `grep "{%spark.executor.cores%}" -rl ./`
        sed -i "s/{%spark.executor.memory%}/${spark_executor_memory}/g" `grep "{%spark.executor.memory%}" -rl ./`
        sed -i "s/{%spark.driver.memory%}/${spark_driver_memory}/g" `grep "{%spark.driver.memory%}" -rl ./`
    else
        sed -i "s/{%yarn.scheduler.maximum-allocation-mb%}/${total_memory}/g" `grep "{%yarn.scheduler.maximum-allocation-mb%}" -rl ./`
        sed -i "s/{%yarn.nodemanager.resource.memory-mb%}/${total_memory}/g" `grep "{%yarn.nodemanager.resource.memory-mb%}" -rl ./`
        sed -i "s/{%yarn.nodemanager.resource.cpu-vcores%}/${total_vcores}/g" `grep "{%yarn.nodemanager.resource.cpu-vcores%}" -rl ./`
        sed -i "s/{%yarn.scheduler.maximum-allocation-vcores%}/${total_vcores}/g" `grep "{%yarn.scheduler.maximum-allocation-vcores%}" -rl ./`
    fi
}

function update_hdfs_data_disks_config() {
    hdfs_nn_dirs="${HADOOP_HOME}/data/dfs/nn"
    hdfs_dn_dirs=""
    if [ -d "/mnt/cloudtik" ]; then
        for data_disk in /mnt/cloudtik/*; do
            [ -d "$data_disk" ] || continue
            if [ -z "$hdfs_dn_dirs" ]; then
                hdfs_dn_dirs=$data_disk/dfs/dn
            else
                hdfs_dn_dirs="$hdfs_dn_dirs,$data_disk/dfs/dn"
            fi
        done
    fi

    # if no disks mounted on /mnt/cloudtik
    if [ -z "$hdfs_dn_dirs" ]; then
        hdfs_dn_dirs="${HADOOP_HOME}/data/dfs/dn"
    fi
    sed -i "s!{%dfs.namenode.name.dir%}!${hdfs_nn_dirs}!g" `grep "{%dfs.namenode.name.dir%}" -rl ./`
    sed -i "s!{%dfs.datanode.data.dir%}!${hdfs_dn_dirs}!g" `grep "{%dfs.datanode.data.dir%}" -rl ./`

    # event log dir
    event_log_dir="hdfs://${HEAD_ADDRESS}:9000/shared/spark-events"
    sed -i "s!{%spark.eventLog.dir%}!${event_log_dir}!g" `grep "{%spark.eventLog.dir%}" -rl ./`
}

function update_data_disks_config() {
    local_dirs=""
    if [ -d "/mnt/cloudtik" ]; then
        for data_disk in /mnt/cloudtik/*; do
            [ -d "$data_disk" ] || continue
            if [ -z "$local_dirs" ]; then
                local_dirs=$data_disk
            else
                local_dirs="$local_dirs,$data_disk"
            fi
      done
    fi

    # set nodemanager.local-dirs
    nodemanager_local_dirs=$local_dirs
    if [ -z "$nodemanager_local_dirs" ]; then
        nodemanager_local_dirs="${HADOOP_HOME}/data/nodemanager/local-dir"
    fi
    sed -i "s!{%yarn.nodemanager.local-dirs%}!${nodemanager_local_dirs}!g" `grep "{%yarn.nodemanager.local-dirs%}" -rl ./`

    # set spark local dir
    spark_local_dir=$local_dirs
    if [ -z "$spark_local_dir" ]; then
        spark_local_dir="/tmp"
    fi
    sed -i "s!{%spark.local.dir%}!${spark_local_dir}!g" `grep "{%spark.local.dir%}" -rl ./`
}

function configure_hadoop_and_spark() {
    prepare_base_conf

    cd $output_dir
    sed -i "s/HEAD_ADDRESS/${HEAD_ADDRESS}/g" `grep "HEAD_ADDRESS" -rl ./`
    sed -i "s!{%HADOOP_HOME%}!${HADOOP_HOME}!g" `grep "{%HADOOP_HOME%}" -rl ./`

    update_spark_runtime_config
    update_data_disks_config

    if [ "$ENABLE_HDFS" == "true" ];then
        update_hdfs_data_disks_config
        cp -r ${output_dir}/hadoop/core-site.xml  ${HADOOP_HOME}/etc/hadoop/
        cp -r ${output_dir}/hadoop/hdfs-site.xml  ${HADOOP_HOME}/etc/hadoop/
    else
        update_config_for_cloud
        cp -r ${output_dir}/hadoop/${provider}/core-site.xml  ${HADOOP_HOME}/etc/hadoop/
    fi

    cp -r ${output_dir}/hadoop/yarn-site.xml  ${HADOOP_HOME}/etc/hadoop/

    if [ $IS_HEAD_NODE == "true" ];then
        cp -r ${output_dir}/spark/*  ${SPARK_HOME}/conf

        if [ "$ENABLE_HDFS" == "true" ]; then
            # Format hdfs once
            ${HADOOP_HOME}/bin/hdfs namenode -format
            # Create event log dir on hdfs
            ${HADOOP_HOME}/bin/hdfs --daemon start namenode
            ${HADOOP_HOME}/bin/hadoop fs -mkdir -p /shared/spark-events
            ${HADOOP_HOME}/bin/hdfs --daemon stop namenode
        else
            # Create event log dir on cloud storage if needed
            # This needs to be done after hadoop file system has been configured correctly
            ${HADOOP_HOME}/bin/hadoop fs -mkdir -p /shared/spark-events
        fi
    fi
}

function configure_jupyter_for_spark() {
  if [ $IS_HEAD_NODE == "true" ]; then
      echo Y | jupyter notebook --generate-config;
      # Set default password(cloudtik) for JupyterLab
      sed -i  "1 ic.NotebookApp.password = 'argon2:\$argon2id\$v=19\$m=10240,t=10,p=8\$Y+sBd6UhAyKNsI+/mHsy9g\$WzJsUujSzmotUkblSTpMwCFoOBVSwm7S5oOPzpC+tz8'" ~/.jupyter/jupyter_notebook_config.py

      # Set default notebook_dir for JupyterLab
      export JUPYTER_WORKSPACE=/home/$(whoami)/jupyter
      mkdir -p $JUPYTER_WORKSPACE
      sed -i  "1 ic.NotebookApp.notebook_dir = '${JUPYTER_WORKSPACE}'" ~/.jupyter/jupyter_notebook_config.py
  fi
  # Config for PySpark
  echo "export PYTHONPATH=\${SPARK_HOME}/python:\${SPARK_HOME}/python/lib/py4j-0.10.9-src.zip" >> ~/.bashrc
  echo "export PYSPARK_PYTHON=\${CONDA_PREFIX}/envs/cloudtik_py37/bin/python" >> ~/.bashrc
  echo "export PYSPARK_DRIVER_PYTHON=\${CONDA_PREFIX}/envs/cloudtik_py37/bin/python" >> ~/.bashrc
}

function configure_ganglia() {
    cluster_name_head="Spark-Head"
    cluster_name="Spark-Workers"
    if [ $IS_HEAD_NODE == "true" ]; then
        # configure ganglia gmetad
        sudo sed -i "s/# default: There is no default value/data_source \"${cluster_name_head}\" ${HEAD_ADDRESS}:8650/g" /etc/ganglia/gmetad.conf
        sudo sed -i "s/data_source \"my cluster\" localhost/data_source \"${cluster_name}\" ${HEAD_ADDRESS}/g" /etc/ganglia/gmetad.conf
        sudo sed -i "s/# gridname \"MyGrid\"/gridname \"CloudTik\"/g" /etc/ganglia/gmetad.conf

        # Configure ganglia monitor
        sudo sed -i "s/send_metadata_interval = 0/send_metadata_interval = 30/g" /etc/ganglia/gmond.conf
        # replace the first occurrence of "mcast_join = 239.2.11.71" with "host = HEAD_IP"
        sudo sed -i "0,/mcast_join = 239.2.11.71/s//host = ${HEAD_ADDRESS}/" /etc/ganglia/gmond.conf
        # comment out the second occurrence
        sudo sed -i "s/mcast_join = 239.2.11.71/\/*mcast_join = 239.2.11.71*\//g" /etc/ganglia/gmond.conf
        sudo sed -i "s/bind = 239.2.11.71/\/*bind = 239.2.11.71*\//g" /etc/ganglia/gmond.conf

        # Make a copy for head cluster after common modifications
        sudo cp /etc/ganglia/gmond.conf /etc/ganglia/gmond.head.conf

        sudo sed -i "s/name = \"unspecified\"/name = \"${cluster_name}\"/g" /etc/ganglia/gmond.conf
        # Disable udp_send_channel
        sudo sed -i "s/udp_send_channel/\/*udp_send_channel/g" /etc/ganglia/gmond.conf
        sudo sed -i "s/\/\* You can specify as many udp_recv_channels/\*\/\/\* You can specify as many udp_recv_channels/g" /etc/ganglia/gmond.conf

        # Modifications for head cluster
        sudo sed -i "s/name = \"unspecified\"/name = \"${cluster_name_head}\"/g" /etc/ganglia/gmond.head.conf
        sudo sed -i "s/port = 8649/port = 8650/g" /etc/ganglia/gmond.head.conf

        # Configure apache2 for ganglia
        sudo cp /etc/ganglia-webfrontend/apache.conf /etc/apache2/sites-enabled/ganglia.conf
        # Fix the ganglia bug: https://github.com/ganglia/ganglia-web/issues/324
        # mention here: https://bugs.launchpad.net/ubuntu/+source/ganglia-web/+bug/1822048
        sudo sed -i "s/\$context_metrics = \"\";/\$context_metrics = array();/g" /usr/share/ganglia-webfrontend/cluster_view.php

        # Add gmond start command for head in service
        sudo sed -i '/\.pid/ a start-stop-daemon --start --quiet --startas $DAEMON --name $NAME.head -- --conf /etc/ganglia/gmond.head.conf --pid-file /var/run/$NAME.head.pid' /etc/init.d/ganglia-monitor
    else
        # Configure ganglia monitor
        sudo sed -i "s/send_metadata_interval = 0/send_metadata_interval = 30/g" /etc/ganglia/gmond.conf
        sudo sed -i "s/name = \"unspecified\"/name = \"${cluster_name}\"/g" /etc/ganglia/gmond.conf
        # replace the first occurrence of "mcast_join = 239.2.11.71" with "host = HEAD_IP"
        sudo sed -i "0,/mcast_join = 239.2.11.71/s//host = ${HEAD_ADDRESS}/" /etc/ganglia/gmond.conf
    fi
}


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
    sudo apt-get install s3fs -y

    echo "${FS_S3A_ACCESS_KEY}:${FS_S3A_SECRET_KEY}" > ${USER_HOME}/.passwd-s3fs
    chmod 600 ${USER_HOME}${USER_HOME}/.passwd-s3fs

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


    echo "accountName ${AZURE_STORAGE_ACCOUNT}" > ${USER_HOME}/fuse_connection.cfg
    echo "accountKey ${AZURE_ACCOUNT_KEY}" >> ${USER_HOME}/fuse_connection.cfg
    echo "containerName ${AZURE_CONTAINER}" >> ${USER_HOME}/fuse_connection.
    chmod 600 ${USER_HOME}/fuse_connection.cfg
    mkdir -p ${MOUNT_PATH}
    blobfuse ${MOUNT_PATH} --tmp-path=/mnt/ramdisk/blobfusetmp  --config-file=${USER_HOME}/fuse_connection.cfg  -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120

}


function gcs_fuse() {
    if [ ! -n "${GCP_GCS_BUCKET}" ]; then
        echo "GCP_GCS_BUCKET environment variable is not set."
        exit 1
    fi
    sudo apt-get update
    sudo apt-get install -y curl
    echo "deb http://packages.cloud.google.com/apt gcsfuse-bionic main" |sudo tee /etc/apt/sources.list.d/gcsfuse.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    sudo apt-get update
    sudo apt-get install gcsfuse -y
    mkdir -p ${MOUNT_PATH}
    gcsfuse ${GCP_GCS_BUCKET} ${MOUNT_PATH}
}


function mount_cloud_storage_for_cloudtik() {
    if [ $FUSE_FLAG == "true" ];then
        if [ "$provider" == "aws" ]; then
          s3_fuse
        fi

        if [ "$provider" == "gcp" ]; then
          gcs_fuse
        fi

        if [ "$provider" == "azure" ]; then
          blob_fuse
        fi
    fi
}


check_spark_installed
set_head_address
set_resources_for_spark
configure_system_folders
configure_hadoop_and_spark
configure_jupyter_for_spark
configure_ganglia
mount_cloud_storage_for_cloudtik
