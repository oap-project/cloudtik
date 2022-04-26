#!/bin/bash

args=$(getopt -a -o h::p: -l head::,node_ip_address::,head_address::,provider:,aws_s3_bucket::,aws_s3_access_key_id::,aws_s3_secret_access_key::,project_id::,gcs_bucket::,gcs_service_account_client_email::,gcs_service_account_private_key_id::,gcs_service_account_private_key::,azure_storage_type::,azure_storage_account::,azure_container::,azure_account_key:: -- "$@")
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
    echo "export CLOUDTIK_HEAD_IP=$HEAD_ADDRESS">> ${USER_HOME}/.bashrc
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

function update_config_for_hdfs() {
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

function update_metastore_config() {
    # To be improved for external metastore cluster
    SPARK_DEFAULTS=${output_dir}/spark/spark-defaults.conf
    if [ "$METASTORE_ENABLED" == "true" ];then
        METASTORE_IP=${HEAD_ADDRESS}

        hive_metastore_uris="thrift://${METASTORE_IP}:9083"
        hive_metastore_version="3.1.2"

        if [ ! -n "${HIVE_HOME}" ]; then
            hive_metastore_jars=maven
        else
            hive_metastore_jars="${HIVE_HOME}/lib/*"

        sed -i "s!{%spark.hadoop.hive.metastore.uris%}!spark.hadoop.hive.metastore.uris ${hive_metastore_uris}!g" ${SPARK_DEFAULTS}
        sed -i "s!{%spark.sql.hive.metastore.version%}!spark.sql.hive.metastore.version ${hive_metastore_version}!g" ${SPARK_DEFAULTS}
        sed -i "s!{%spark.sql.hive.metastore.jars%}!spark.sql.hive.metastore.jars ${hive_metastore_jars}!g" ${SPARK_DEFAULTS}
    else
        # replace with empty
        sed -i "s/{%spark.hadoop.hive.metastore.uris%}//g" ${SPARK_DEFAULTS}
        sed -i "s/{%spark.sql.hive.metastore.version%}//g" ${SPARK_DEFAULTS}
        sed -i "s/{%spark.sql.hive.metastore.jars%}//g" ${SPARK_DEFAULTS}
    fi
}

function configure_hadoop_and_spark() {
    prepare_base_conf

    cd $output_dir
    sed -i "s/HEAD_ADDRESS/${HEAD_ADDRESS}/g" `grep "HEAD_ADDRESS" -rl ./`

    update_spark_runtime_config
    update_data_disks_config

    if [ "$HDFS_ENABLED" == "true" ];then
        update_config_for_hdfs
    else
        update_config_for_cloud
        cp -r ${output_dir}/hadoop/${provider}/core-site.xml  ${HADOOP_HOME}/etc/hadoop/
    fi

    cp -r ${output_dir}/hadoop/yarn-site.xml  ${HADOOP_HOME}/etc/hadoop/

    if [ $IS_HEAD_NODE == "true" ];then
        update_metastore_config

        cp -r ${output_dir}/spark/*  ${SPARK_HOME}/conf

        if [ "$HDFS_ENABLED" == "true" ]; then
            # Create event log dir on hdfs
            ${HADOOP_HOME}/bin/hdfs --loglevel WARN --daemon start namenode
            ${HADOOP_HOME}/bin/hadoop --loglevel WARN fs -mkdir -p /shared/spark-events
            ${HADOOP_HOME}/bin/hdfs --loglevel WARN --daemon stop namenode
        else
            # Create event log dir on cloud storage if needed
            # This needs to be done after hadoop file system has been configured correctly
            ${HADOOP_HOME}/bin/hadoop --loglevel WARN fs -mkdir -p /shared/spark-events
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
      sed -i  "1 ic.NotebookApp.ip = '${HEAD_ADDRESS}'" ~/.jupyter/jupyter_notebook_config.py
  fi
  # Config for PySpark
  echo "export PYTHONPATH=\${SPARK_HOME}/python:\${SPARK_HOME}/python/lib/py4j-0.10.9-src.zip" >> ~/.bashrc
  echo "export PYSPARK_PYTHON=\${CONDA_PREFIX}/envs/cloudtik_py37/bin/python" >> ~/.bashrc
  echo "export PYSPARK_DRIVER_PYTHON=\${CONDA_PREFIX}/envs/cloudtik_py37/bin/python" >> ~/.bashrc
}

check_spark_installed
set_head_address
set_resources_for_spark
configure_system_folders
configure_hadoop_and_spark
configure_jupyter_for_spark

exit 0
