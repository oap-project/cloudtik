#!/bin/bash

args=$(getopt -a -o h::p: -l head::,node_ip_address::,head_address:: -- "$@")
eval set -- "${args}"

IS_HEAD_NODE=false
USER_HOME=/home/$(whoami)
HADOOP_CREDENTIAL_FILE_PATH="jceks://file@${HADOOP_HOME}/etc/hadoop/credential.jceks"
HADOOP_CREDENTIAL_PROPERTY="<property>\n      <name>hadoop.security.credential.provider.path</name>\n      <value>${HADOOP_CREDENTIAL_FILE_PATH}</value>\n    </property>"

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
}

function set_resources_for_spark() {
    # For nodemanager
    memory_ratio=0.8
    if [ ! -z "${YARN_RESOURCE_MEMORY_RATIO}" ]; then
        memory_ratio=${YARN_RESOURCE_MEMORY_RATIO}
    fi
    total_memory=$(awk -v ratio=${memory_ratio} -v total_physical_memory=$(cloudtik resources --memory --in-mb) 'BEGIN{print ratio * total_physical_memory}')
    total_memory=${total_memory%.*}
    total_vcores=$(cloudtik resources --cpu)

    # For Head Node
    if [ $IS_HEAD_NODE == "true" ];then
        spark_executor_cores=$(cat ~/cloudtik_bootstrap_config.yaml | jq '."runtime"."spark"."spark_executor_resource"."spark_executor_cores"')
        spark_executor_memory=$(cat ~/cloudtik_bootstrap_config.yaml | jq '."runtime"."spark"."spark_executor_resource"."spark_executor_memory"')M
        spark_driver_memory=$(cat ~/cloudtik_bootstrap_config.yaml | jq '."runtime"."spark"."spark_executor_resource"."spark_driver_memory"')M
        yarn_container_maximum_vcores=$(cat ~/cloudtik_bootstrap_config.yaml | jq '."runtime"."spark"."yarn_container_resource"."yarn_container_maximum_vcores"')
        yarn_container_maximum_memory=$(cat ~/cloudtik_bootstrap_config.yaml | jq '."runtime"."spark"."yarn_container_resource"."yarn_container_maximum_memory"')
    fi
}

function check_hdfs_storage() {
    if [ -n  "${HDFS_NAMENODE_URI}" ];then
        HDFS_STORAGE="true"
    else
        HDFS_STORAGE="false"
    fi
}

function set_cloud_storage_provider() {
    cloud_storage_provider="none"
    if [ "$AWS_CLOUD_STORAGE" == "true" ]; then
        cloud_storage_provider="aws"
    elif [ "$AZURE_CLOUD_STORAGE" == "true" ]; then
        cloud_storage_provider="azure"
    elif [ "$GCP_CLOUD_STORAGE" == "true" ]; then
        cloud_storage_provider="gcp"
    fi
}

function update_credential_config_for_aws() {
    if [ "$AWS_WEB_IDENTITY" == "true" ]; then
        # Replace with InstanceProfileCredentialsProvider with WebIdentityTokenCredentialsProvider for Kubernetes
        sed -i "s#InstanceProfileCredentialsProvider#WebIdentityTokenCredentialsProvider#g" `grep "InstanceProfileCredentialsProvider" -rl ./`

        if [ ! -z "${AWS_ROLE_ARN}" ] && [ ! -z "${AWS_WEB_IDENTITY_TOKEN_FILE}" ]; then
            WEB_IDENTITY_ENVS="spark.yarn.appMasterEnv.AWS_ROLE_ARN ${AWS_ROLE_ARN}\nspark.yarn.appMasterEnv.AWS_WEB_IDENTITY_TOKEN_FILE ${AWS_WEB_IDENTITY_TOKEN_FILE}\nspark.executorEnv.AWS_ROLE_ARN ${AWS_ROLE_ARN}\nspark.executorEnv.AWS_WEB_IDENTITY_TOKEN_FILE ${AWS_WEB_IDENTITY_TOKEN_FILE}\n"
            sed -i "$ a ${WEB_IDENTITY_ENVS}" ${SPARK_DEFAULTS}
        fi
    fi

    sed -i "s#{%fs.s3a.access.key%}#${AWS_S3_ACCESS_KEY_ID}#g" `grep "{%fs.s3a.access.key%}" -rl ./`

    if [ ! -z "${AWS_S3_SECRET_ACCESS_KEY}" ]; then
        ${HADOOP_HOME}/bin/hadoop credential create fs.s3a.secret.key -value ${AWS_S3_SECRET_ACCESS_KEY}  -provider ${HADOOP_CREDENTIAL_FILE_PATH}
        sed -i "s#{%hadoop.credential.property%}#${HADOOP_CREDENTIAL_PROPERTY}#g" `grep "{%hadoop.credential.property%}" -rl ./`
    else
        sed -i "s#{%hadoop.credential.property%}#""#g" `grep "{%hadoop.credential.property%}" -rl ./`
    fi
}

function update_credential_config_for_gcp() {
    sed -i "s#{%fs.gs.project.id%}#${GCP_PROJECT_ID}#g" `grep "{%fs.gs.project.id%}" -rl ./`

    sed -i "s#{%fs.gs.auth.service.account.email%}#${GCS_SERVICE_ACCOUNT_CLIENT_EMAIL}#g" `grep "{%fs.gs.auth.service.account.email%}" -rl ./`
    sed -i "s#{%fs.gs.auth.service.account.private.key.id%}#${GCS_SERVICE_ACCOUNT_PRIVATE_KEY_ID}#g" `grep "{%fs.gs.auth.service.account.private.key.id%}" -rl ./`

    if [ ! -z "${GCS_SERVICE_ACCOUNT_PRIVATE_KEY}" ]; then
        ${HADOOP_HOME}/bin/hadoop credential create fs.gs.auth.service.account.private.key -value ${GCS_SERVICE_ACCOUNT_PRIVATE_KEY}  -provider ${HADOOP_CREDENTIAL_FILE_PATH}
        sed -i "s#{%hadoop.credential.property%}#${HADOOP_CREDENTIAL_PROPERTY}#g" `grep "{%hadoop.credential.property%}" -rl ./`
    else
        sed -i "s#{%hadoop.credential.property%}#""#g" `grep "{%hadoop.credential.property%}" -rl ./`
    fi
}

function update_credential_config_for_azure() {
    sed -i "s#{%azure.storage.account%}#${AZURE_STORAGE_ACCOUNT}#g" "$(grep "{%azure.storage.account%}" -rl ./)"
    sed -i "s#{%fs.azure.account.oauth2.msi.tenant%}#${AZURE_MANAGED_IDENTITY_TENANT_ID}#g" "$(grep "{%fs.azure.account.oauth2.msi.tenant%}" -rl ./)"
    sed -i "s#{%fs.azure.account.oauth2.client.id%}#${AZURE_MANAGED_IDENTITY_CLIENT_ID}#g" "$(grep "{%fs.azure.account.oauth2.client.id%}" -rl ./)"

    if [ "$AZURE_STORAGE_TYPE" == "blob" ];then
        AZURE_ENDPOINT="blob"
    else
        # Default to datalake
        AZURE_ENDPOINT="dfs"
    fi
    sed -i "s#{%storage.endpoint%}#${AZURE_ENDPOINT}#g" "$(grep "{%storage.endpoint%}" -rl ./)"

    if [ "$AZURE_STORAGE_TYPE" != "blob" ];then
        # datalake
        if [ -n  "${AZURE_ACCOUNT_KEY}" ];then
            sed -i "s#{%auth.type%}#SharedKey#g" "$(grep "{%auth.type%}" -rl ./)"
        else
            sed -i "s#{%auth.type%}##g" "$(grep "{%auth.type%}" -rl ./)"
        fi
    fi

    if [ ! -z "${AZURE_ACCOUNT_KEY}" ]; then
        FS_KEY_NAME_FOR_AZURE="fs.azure.account.key.${%AZURE_STORAGE_ACCOUNT%}.${%AZURE_ENDPOINT%}.core.windows.net"
        ${HADOOP_HOME}/bin/hadoop credential create ${FS_KEY_NAME_FOR_AZURE} -value ${AZURE_ACCOUNT_KEY}  -provider ${HADOOP_CREDENTIAL_FILE_PATH}
        sed -i "s#{%hadoop.credential.property%}#${HADOOP_CREDENTIAL_PROPERTY}#g" `grep "{%hadoop.credential.property%}" -rl ./`
    else
        sed -i "s#{%hadoop.credential.property%}#""#g" `grep "{%hadoop.credential.property%}" -rl ./`
    fi
}

function update_credential_config_for_provider() {
    if [ "${cloud_storage_provider}" == "aws" ]; then
        update_credential_config_for_aws
    elif [ "${cloud_storage_provider}" == "azure" ]; then
        update_credential_config_for_azure
    elif [ "${cloud_storage_provider}" == "gcp" ]; then
        update_credential_config_for_gcp
    fi
}

function update_config_for_spark_dirs() {
    sed -i "s!{%spark.eventLog.dir%}!${event_log_dir}!g" `grep "{%spark.eventLog.dir%}" -rl ./`
    sed -i "s!{%spark.sql.warehouse.dir%}!${sql_warehouse_dir}!g" `grep "{%spark.sql.warehouse.dir%}" -rl ./`
}

function update_config_for_local_hdfs() {
    fs_default_dir="hdfs://${HEAD_ADDRESS}:9000"
    # event log dir
    event_log_dir="${fs_default_dir}/shared/spark-events"
    sql_warehouse_dir="${fs_default_dir}/shared/spark-warehouse"

    update_config_for_spark_dirs
}

function update_config_for_hdfs() {
    # configure namenode uri for core-site.xml
    fs_default_dir="${HDFS_NAMENODE_URI}"
    sed -i "s!{%fs.default.name%}!${fs_default_dir}!g" `grep "{%fs.default.name%}" -rl ./`

    # Still update credential config for cloud provider storage in the case of explict usage
    update_credential_config_for_provider

    # event log dir
    event_log_dir="${fs_default_dir}/shared/spark-events"
    sql_warehouse_dir="${fs_default_dir}/shared/spark-warehouse"

    update_config_for_spark_dirs
}

function update_config_for_aws() {
    fs_default_dir="s3a://${AWS_S3_BUCKET}"
    sed -i "s!{%fs.default.name%}!${fs_default_dir}!g" `grep "{%fs.default.name%}" -rl ./`

    update_credential_config_for_aws

    # event log dir
    if [ -z "${AWS_S3_BUCKET}" ]; then
        event_log_dir="file:///tmp/spark-events"
        sql_warehouse_dir="$USER_HOME/shared/spark-warehouse"
    else
        event_log_dir="${fs_default_dir}/shared/spark-events"
        sql_warehouse_dir="${fs_default_dir}/shared/spark-warehouse"
    fi

    update_config_for_spark_dirs
}

function update_config_for_gcp() {
    fs_default_dir="gs://${GCS_BUCKET}"
    sed -i "s!{%fs.default.name%}!${fs_default_dir}!g" `grep "{%fs.default.name%}" -rl ./`

    update_credential_config_for_gcp

    # event log dir
    if [ -z "${GCS_BUCKET}" ]; then
        event_log_dir="file:///tmp/spark-events"
        sql_warehouse_dir="$USER_HOME/shared/spark-warehouse"
    else
        event_log_dir="${fs_default_dir}/shared/spark-events"
        sql_warehouse_dir="${fs_default_dir}/shared/spark-warehouse"
    fi

    update_config_for_spark_dirs
}

function update_config_for_azure() {
    if [ "$AZURE_STORAGE_TYPE" == "blob" ];then
        AZURE_SCHEMA="wasbs"
        AZURE_ENDPOINT="blob"
    else
        # Default to datalake
        # Must be Azure storage kind must be blob (Azure Blob Storage) or datalake (Azure Data Lake Storage Gen 2)
        AZURE_SCHEMA="abfs"
        AZURE_ENDPOINT="dfs"
    fi

    fs_default_dir="${AZURE_SCHEMA}://${AZURE_CONTAINER}@${AZURE_STORAGE_ACCOUNT}.${AZURE_ENDPOINT}.core.windows.net"
    sed -i "s!{%fs.default.name%}!${fs_default_dir}!g" `grep "{%fs.default.name%}" -rl ./`

    update_credential_config_for_azure

    # event log dir
    if [ -z "${AZURE_CONTAINER}" ]; then
        event_log_dir="file:///tmp/spark-events"
        sql_warehouse_dir="$USER_HOME/shared/spark-warehouse"
    else
        event_log_dir="${fs_default_dir}/shared/spark-events"
        sql_warehouse_dir="${fs_default_dir}/shared/spark-warehouse"
    fi

    update_config_for_spark_dirs
}

function update_config_for_remote_storage() {
    if [ "$HDFS_STORAGE" == "true" ]; then
        update_config_for_hdfs
    elif [ "${cloud_storage_provider}" == "aws" ]; then
        update_config_for_aws
    elif [ "${cloud_storage_provider}" == "azure" ]; then
        update_config_for_azure
    elif [ "${cloud_storage_provider}" == "gcp" ]; then
        update_config_for_gcp
    fi
}

function update_config_for_storage() {
    if [ "$HDFS_ENABLED" == "true" ];then
        update_config_for_local_hdfs
    else
        check_hdfs_storage
        set_cloud_storage_provider
        update_config_for_remote_storage

        if [ "${cloud_storage_provider}" != "none" ];then
            cp -r ${output_dir}/hadoop/${cloud_storage_provider}/core-site.xml ${HADOOP_HOME}/etc/hadoop/
        else
            # Possible remote hdfs without cloud storage
            cp -r ${output_dir}/hadoop/core-site.xml ${HADOOP_HOME}/etc/hadoop/
        fi
    fi
}

function update_yarn_config() {
    if [ $IS_HEAD_NODE == "true" ];then
        sed -i "s/{%yarn.scheduler.maximum-allocation-mb%}/${yarn_container_maximum_memory}/g" `grep "{%yarn.scheduler.maximum-allocation-mb%}" -rl ./`
        sed -i "s/{%yarn.nodemanager.resource.memory-mb%}/${yarn_container_maximum_memory}/g" `grep "{%yarn.nodemanager.resource.memory-mb%}" -rl ./`
        sed -i "s/{%yarn.nodemanager.resource.cpu-vcores%}/${yarn_container_maximum_vcores}/g" `grep "{%yarn.nodemanager.resource.cpu-vcores%}" -rl ./`
        sed -i "s/{%yarn.scheduler.maximum-allocation-vcores%}/${yarn_container_maximum_vcores}/g" `grep "{%yarn.scheduler.maximum-allocation-vcores%}" -rl ./`
    else
        sed -i "s/{%yarn.scheduler.maximum-allocation-mb%}/${total_memory}/g" `grep "{%yarn.scheduler.maximum-allocation-mb%}" -rl ./`
        sed -i "s/{%yarn.nodemanager.resource.memory-mb%}/${total_memory}/g" `grep "{%yarn.nodemanager.resource.memory-mb%}" -rl ./`
        sed -i "s/{%yarn.nodemanager.resource.cpu-vcores%}/${total_vcores}/g" `grep "{%yarn.nodemanager.resource.cpu-vcores%}" -rl ./`
        sed -i "s/{%yarn.scheduler.maximum-allocation-vcores%}/${total_vcores}/g" `grep "{%yarn.scheduler.maximum-allocation-vcores%}" -rl ./`
    fi
}

function update_spark_runtime_config() {
    if [ $IS_HEAD_NODE == "true" ];then
        sed -i "s/{%spark.executor.cores%}/${spark_executor_cores}/g" `grep "{%spark.executor.cores%}" -rl ./`
        sed -i "s/{%spark.executor.memory%}/${spark_executor_memory}/g" `grep "{%spark.executor.memory%}" -rl ./`
        sed -i "s/{%spark.driver.memory%}/${spark_driver_memory}/g" `grep "{%spark.driver.memory%}" -rl ./`
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
    if [ "$METASTORE_ENABLED" == "true" ] || [ ! -z "$HIVE_METASTORE_URI" ]; then
        if [ "$METASTORE_ENABLED" == "true" ]; then
            METASTORE_IP=${HEAD_ADDRESS}
            hive_metastore_uris="thrift://${METASTORE_IP}:9083"
        else
            hive_metastore_uris="$HIVE_METASTORE_URI"
        fi

        hive_metastore_version="3.1.2"

        if [ ! -n "${HIVE_HOME}" ]; then
            hive_metastore_jars=maven
        else
            hive_metastore_jars="${HIVE_HOME}/lib/*"
        fi

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
    SPARK_DEFAULTS=${output_dir}/spark/spark-defaults.conf

    cd $output_dir
    sed -i "s/HEAD_ADDRESS/${HEAD_ADDRESS}/g" `grep "HEAD_ADDRESS" -rl ./`

    update_yarn_config
    update_spark_runtime_config
    update_data_disks_config
    update_config_for_storage

    cp -r ${output_dir}/hadoop/yarn-site.xml ${HADOOP_HOME}/etc/hadoop/

    if [ $IS_HEAD_NODE == "true" ];then
        update_metastore_config

        cp -r ${output_dir}/spark/* ${SPARK_HOME}/conf

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
      echo Y | jupyter lab --generate-config;
      # Set default password(cloudtik) for JupyterLab
      sed -i  "1 ic.NotebookApp.password = 'argon2:\$argon2id\$v=19\$m=10240,t=10,p=8\$Y+sBd6UhAyKNsI+/mHsy9g\$WzJsUujSzmotUkblSTpMwCFoOBVSwm7S5oOPzpC+tz8'" ~/.jupyter/jupyter_lab_config.py

      # Set default notebook_dir for JupyterLab
      export JUPYTER_WORKSPACE=/home/$(whoami)/jupyter
      mkdir -p $JUPYTER_WORKSPACE
      sed -i  "1 ic.NotebookApp.notebook_dir = '${JUPYTER_WORKSPACE}'" ~/.jupyter/jupyter_lab_config.py
      sed -i  "1 ic.NotebookApp.ip = '${HEAD_ADDRESS}'" ~/.jupyter/jupyter_lab_config.py
  fi
}
check_spark_installed
set_head_address
set_resources_for_spark
configure_system_folders
configure_hadoop_and_spark
configure_jupyter_for_spark

exit 0
