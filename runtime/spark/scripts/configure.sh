#!/bin/bash

args=$(getopt -a -o h::p: -l head::,head_address::,provider:,aws_s3a_bucket::,s3a_access_key::,s3a_secret_key::,project_id::,gcp_gcs_bucket::,fs_gs_auth_service_account_email::,fs_gs_auth_service_account_private_key_id::,fs_gs_auth_service_account_private_key::,azure_storage_kind::,azure_storage_account::,azure_container::,azure_account_key:: -- "$@")
eval set -- "${args}"

IS_HEAD_NODE=false

while true
do
    case "$1" in
    --head)
        IS_HEAD_NODE=true
        ;;
    -h|--head_address)
        HEAD_ADDRESS=$2
        shift
        ;;
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

function prepare_base_conf() {
    source_dir=$(cd $(dirname ${BASH_SOURCE[0]})/..;pwd)/conf
    output_dir=$(dirname ${source_dir})/outconf
    rm -rf  $output_dir
    mkdir -p $output_dir
    cp -r $source_dir/* $output_dir
}

function check_env() {
    if [ ! -n "${HADOOP_HOME}" ]; then
        echo "HADOOP_HOME environment variable is not set."
        exit 1
    fi

    if [ ! -n "${SPARK_HOME}" ]; then
        echo "SPARK_HOME environment variable is not set."
        exit 1
    fi
}

function set_head_address() {
    if [ ! -n "${HEAD_ADDRESS}" ]; then
	    HEAD_ADDRESS=$(hostname -I | awk '{print $1}')
	  fi
}

function caculate_worker_resources() {
    #For nodemanager
    total_memory=$(awk '($1 == "MemTotal:"){print $2/1024*0.8}' /proc/meminfo)
    total_memory=${total_memory%.*}
    total_vcores=$(cat /proc/cpuinfo | grep processor | wc -l)
}

function set_resources_for_spark() {
    #For Head Node
    if [ $IS_HEAD_NODE == "true" ];then
        spark_executor_cores=$(cat ~/cloudtik_bootstrap_config.yaml | jq '."runtime"."spark"."spark_executor_resource"."spark_executor_cores"')
        spark_executor_memory=$(cat ~/cloudtik_bootstrap_config.yaml | jq '."runtime"."spark"."spark_executor_resource"."spark_executor_memory"')M
        spark_driver_memory=$(cat ~/cloudtik_bootstrap_config.yaml | jq '."runtime"."spark"."spark_executor_resource"."spark_driver_memory"')M
        yarn_container_maximum_vcores=$(cat ~/cloudtik_bootstrap_config.yaml | jq '."runtime"."spark"."yarn_container_resource"."yarn_container_maximum_vcores"')
        yarn_container_maximum_memory=$(cat ~/cloudtik_bootstrap_config.yaml | jq '."runtime"."spark"."yarn_container_resource"."yarn_container_maximum_memory"')
    fi
}

function update_aws_hadoop_config() {
    sed -i "s#{%aws.s3a.bucket%}#${AWS_S3A_BUCKET}#g" `grep "{%aws.s3a.bucket%}" -rl ./`
    sed -i "s#{%fs.s3a.access.key%}#${FS_S3A_ACCESS_KEY}#g" `grep "{%fs.s3a.access.key%}" -rl ./`
    sed -i "s#{%fs.s3a.secret.key%}#${FS_S3A_SECRET_KEY}#g" `grep "{%fs.s3a.secret.key%}" -rl ./`

    # event log dir
    if [ -z "${AWS_S3A_BUCKET}" ]; then
        event_log_dir="file:///tmp/spark-events"
    else
        event_log_dir="s3a://${AWS_S3A_BUCKET}/shared/spark-events"
    fi
    sed -i "s!{%spark.eventLog.dir%}!${event_log_dir}!g" `grep "{%spark.eventLog.dir%}" -rl ./`
}

function update_gcp_hadoop_config() {
    sed -i "s#{%project_id%}#${PROJECT_ID}#g" `grep "{%project_id%}" -rl ./`
    sed -i "s#{%gcp.gcs.bucket%}#${GCP_GCS_BUCKET}#g" `grep "{%gcp.gcs.bucket%}" -rl ./`
    sed -i "s#{%fs.gs.auth.service.account.email%}#${FS_GS_AUTH_SERVICE_ACCOUNT_EMAIL}#g" `grep "{%fs.gs.auth.service.account.email%}" -rl ./`
    sed -i "s#{%fs.gs.auth.service.account.private.key.id%}#${FS_GS_AUTH_SERVICE_ACCOUNT_PRIVATE_KEY_ID}#g" `grep "{%fs.gs.auth.service.account.private.key.id%}" -rl ./`
    private_key_has_open_quote=${FS_GS_AUTH_SERVICE_ACCOUNT_PRIVATE_KEY%\"}
    private_key=${private_key_has_open_quote#\"}
    sed -i "s#{%fs.gs.auth.service.account.private.key%}#${private_key}#g" `grep "{%fs.gs.auth.service.account.private.key%}" -rl ./`

    # event log dir
    if [ -z "${GCP_GCS_BUCKET}" ]; then
        event_log_dir="file:///tmp/spark-events"
    else
        event_log_dir="gs://${GCP_GCS_BUCKET}/shared/spark-events"
    fi
    sed -i "s!{%spark.eventLog.dir%}!${event_log_dir}!g" `grep "{%spark.eventLog.dir%}" -rl ./`
}

function update_azure_hadoop_config() {
    sed -i "s#{%azure.storage.account%}#${AZURE_STORAGE_ACCOUNT}#g" "$(grep "{%azure.storage.account%}" -rl ./)"
    sed -i "s#{%azure.container%}#${AZURE_CONTAINER}#g" "$(grep "{%azure.container%}" -rl ./)"
    sed -i "s#{%azure.account.key%}#${AZURE_ACCOUNT_KEY}#g" "$(grep "{%azure.account.key%}" -rl ./)"
    if [ $AZURE_STORAGE_KIND == "wasbs" ];then
        endpoint="blob"
        sed -i "s#{%azure.storage.kind%}#wasbs#g" "$(grep "{%azure.storage.kind%}" -rl ./)"
        sed -i "s#{%storage.endpoint%}#blob#g" "$(grep "{%storage.endpoint%}" -rl ./)"
    elif [ $AZURE_STORAGE_KIND == "abfs" ];then
        endpoint="dfs"
        sed -i "s#{%azure.storage.kind%}#abfs#g" "$(grep "{%azure.storage.kind%}" -rl ./)"
        sed -i "s#{%storage.endpoint%}#dfs#g" "$(grep "{%storage.endpoint%}" -rl ./)"
        sed -i "s#{%auth.type%}#SharedKey#g" "$(grep "{%auth.type%}" -rl ./)"
    else
        endpoint=""
       echo "Azure storage kind must be wasbs(Azure Blob storage) or abfs(Azure Data Lake Gen 2)"
    fi

    # event log dir
    if [ -z "${AZURE_CONTAINER}" ] || [ -z "$endpoint" ]; then
        event_log_dir="file:///tmp/spark-events"
    else
        event_log_dir="$AZURE_STORAGE_KIND://${AZURE_CONTAINER}@${AZURE_STORAGE_ACCOUNT}.{%storage.endpoint%}.core.windows.net/shared/spark-events"
    fi
    sed -i "s!{%spark.eventLog.dir%}!${event_log_dir}!g" `grep "{%spark.eventLog.dir%}" -rl ./`
}

function update_hadoop_config_for_cloud() {
    if [ "$provider" == "aws" ]; then
      update_aws_hadoop_config
    fi

    if [ "$provider" == "gcp" ]; then
      update_gcp_hadoop_config
    fi

    if [ "$provider" == "azure" ]; then
      update_azure_hadoop_config
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
        nodemanager_local_dirs="{%HADOOP_HOME%}/data/nodemanager/local-dir"
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
    update_hadoop_config_for_cloud
    update_data_disks_config

    cp -r ${output_dir}/hadoop/${provider}/core-site.xml  ${HADOOP_HOME}/etc/hadoop/
    cp -r ${output_dir}/hadoop/yarn-site.xml  ${HADOOP_HOME}/etc/hadoop/

    if [ $IS_HEAD_NODE == "true" ];then
	      cp -r ${output_dir}/spark/*  ${SPARK_HOME}/conf

	      # Create event log dir on cloud storage if needed
	      # This needs to be done after hadoop file system has been configured correctly
	      ${HADOOP_HOME}/bin/hadoop fs -mkdir -p /shared/spark-events
    fi
}

function configure_jupyter_for_spark() {
  # Set default password(cloudtik) for JupyterLab
  echo Y | jupyter notebook --generate-config;
  sed -i  "1 ic.NotebookApp.password = 'argon2:\$argon2id\$v=19\$m=10240,t=10,p=8\$Y+sBd6UhAyKNsI+/mHsy9g\$WzJsUujSzmotUkblSTpMwCFoOBVSwm7S5oOPzpC+tz8'" ~/.jupyter/jupyter_notebook_config.py
  # Config for PySpark
  echo "export PYTHONPATH=\${SPARK_HOME}/python:\${SPARK_HOME}/python/lib/py4j-0.10.9-src.zip \
        export PYSPARK_PYTHON=\${CONDA_PREFIX}/bin/python \
        export PYSPARK_DRIVER_PYTHON=\${CONDA_PREFIX}/bin/python" >> ~/.bashrc
}


check_env
set_head_address
caculate_worker_resources
set_resources_for_spark
configure_hadoop_and_spark
configure_jupyter_for_spark

