#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

USER_HOME=/home/$(whoami)
RUNTIME_PATH=$USER_HOME/runtime

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

# Hadoop cloud credential configuration functions
. "$ROOT_DIR"/common/scripts/hadoop-cloud-credential.sh

# Cloud storage fuse functions
. "$ROOT_DIR"/common/scripts/cloud-storage-fuse.sh

function prepare_base_conf() {
    source_dir=$(dirname "${BIN_DIR}")/conf
    output_dir=/tmp/spark/conf
    rm -rf  $output_dir
    mkdir -p $output_dir
    cp -r $source_dir/* $output_dir

    # Include hadoop config file for cloud providers
    cp -r "$ROOT_DIR"/common/conf/hadoop $output_dir
    # Make copy for local and remote HDFS
    cp $output_dir/hadoop/core-site.xml $output_dir/hadoop/core-site-local.xml
    sed -i "s!{%fs.default.name%}!{%local.fs.default.name%}!g" $output_dir/hadoop/core-site-local.xml
    cp $output_dir/hadoop/core-site.xml $output_dir/hadoop/core-site-remote.xml
    sed -i "s!{%fs.default.name%}!{%remote.fs.default.name%}!g" $output_dir/hadoop/core-site-remote.xml

    cd $output_dir
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
    # Create dirs for data
    mkdir -p ${RUNTIME_PATH}/jupyter/logs
    mkdir -p ${HADOOP_HOME}/logs
}

function set_resources_for_spark() {
    # For nodemanager
    memory_ratio=0.8
    if [ ! -z "${YARN_RESOURCE_MEMORY_RATIO}" ]; then
        memory_ratio=${YARN_RESOURCE_MEMORY_RATIO}
    fi
    total_memory=$(awk -v ratio=${memory_ratio} -v total_physical_memory=$(cloudtik node resources --memory --in-mb) 'BEGIN{print ratio * total_physical_memory}')
    total_memory=${total_memory%.*}
    total_vcores=$(cloudtik node resources --cpu)

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
    if [ ! -z  "${HDFS_NAMENODE_URI}" ];then
        HDFS_STORAGE="true"
    else
        HDFS_STORAGE="false"
    fi
}

function update_spark_credential_config_for_aws() {
    if [ "$AWS_WEB_IDENTITY" == "true" ]; then
        if [ ! -z "${AWS_ROLE_ARN}" ] && [ ! -z "${AWS_WEB_IDENTITY_TOKEN_FILE}" ]; then
            WEB_IDENTITY_ENVS="spark.yarn.appMasterEnv.AWS_ROLE_ARN ${AWS_ROLE_ARN}\nspark.yarn.appMasterEnv.AWS_WEB_IDENTITY_TOKEN_FILE ${AWS_WEB_IDENTITY_TOKEN_FILE}\nspark.executorEnv.AWS_ROLE_ARN ${AWS_ROLE_ARN}\nspark.executorEnv.AWS_WEB_IDENTITY_TOKEN_FILE ${AWS_WEB_IDENTITY_TOKEN_FILE}\n"
            sed -i "$ a ${WEB_IDENTITY_ENVS}" ${SPARK_DEFAULTS}
        fi
    fi
}

function update_cloud_storage_credential_config() {
    # update hadoop credential config
    update_credential_config_for_provider

    # We need do some specific config for AWS kubernetes web identity environment variables
    if [ "${cloud_storage_provider}" == "aws" ]; then
        update_spark_credential_config_for_aws
    fi
}

function update_config_for_spark_dirs() {
    sed -i "s!{%spark.eventLog.dir%}!${event_log_dir}!g" `grep "{%spark.eventLog.dir%}" -rl ./`
    sed -i "s!{%spark.sql.warehouse.dir%}!${sql_warehouse_dir}!g" `grep "{%spark.sql.warehouse.dir%}" -rl ./`
}

function update_config_for_local_hdfs() {
    fs_default_dir="hdfs://${HEAD_ADDRESS}:9000"
    sed -i "s!{%fs.default.name%}!${fs_default_dir}!g" `grep "{%fs.default.name%}" -rl ./`

    # Still update credential config for cloud provider storage in the case of explict usage
    update_cloud_storage_credential_config

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
    update_cloud_storage_credential_config

    # event log dir
    event_log_dir="${fs_default_dir}/shared/spark-events"
    sql_warehouse_dir="${fs_default_dir}/shared/spark-warehouse"

    update_config_for_spark_dirs
}

function update_config_for_aws() {
    fs_default_dir="s3a://${AWS_S3_BUCKET}"
    sed -i "s!{%fs.default.name%}!${fs_default_dir}!g" `grep "{%fs.default.name%}" -rl ./`

    update_cloud_storage_credential_config

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
    fs_default_dir="gs://${GCP_GCS_BUCKET}"
    sed -i "s!{%fs.default.name%}!${fs_default_dir}!g" `grep "{%fs.default.name%}" -rl ./`

    update_cloud_storage_credential_config

    # event log dir
    if [ -z "${GCP_GCS_BUCKET}" ]; then
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

    update_cloud_storage_credential_config

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

function update_config_for_aliyun() {
    fs_default_dir="oss://${ALIYUN_OSS_BUCKET}"
    sed -i "s!{%fs.default.name%}!${fs_default_dir}!g" `grep "{%fs.default.name%}" -rl ./`
    sed -i "s!{%fs.oss.endpoint%}!${ALIYUN_OSS_INTERNAL_ENDPOINT}!g" `grep "{%fs.oss.endpoint%}" -rl ./`

    update_cloud_storage_credential_config

    # event log dir
    if [ -z "${ALIYUN_OSS_BUCKET}" ]; then
        event_log_dir="file:///tmp/spark-events"
        sql_warehouse_dir="$USER_HOME/shared/spark-warehouse"
    else
        event_log_dir="${fs_default_dir}/shared/spark-events"
        sql_warehouse_dir="${fs_default_dir}/shared/spark-warehouse"
    fi

    update_config_for_spark_dirs
}

function update_config_for_huaweicloud() {
    fs_default_dir="obs://${HUAWEICLOUD_OBS_BUCKET}"
    sed -i "s!{%fs.default.name%}!${fs_default_dir}!g" `grep "{%fs.default.name%}" -rl ./`
    sed -i "s!{%fs.obs.endpoint.property%}!${HUAWEICLOUD_OBS_ENDPOINT}!g" `grep "{%fs.obs.endpoint.property%}" -rl ./`

    update_cloud_storage_credential_config

    # event log dir
    if [ -z "${HUAWEICLOUD_OBS_BUCKET}" ]; then
        event_log_dir="file:///tmp/spark-events"
        sql_warehouse_dir="$USER_HOME/shared/spark-warehouse"
    else
        event_log_dir="${fs_default_dir}/shared/spark-events"
        sql_warehouse_dir="${fs_default_dir}/shared/spark-warehouse"
    fi

    update_config_for_spark_dirs
}

function update_config_for_hadoop_storage() {
    if [ "${HADOOP_DEFAULT_CLUSTER}" == "true" ]; then
        if [ "$HDFS_STORAGE" == "true" ]; then
            update_config_for_hdfs
            return 0
        elif [ "$HDFS_ENABLED" == "true" ]; then
            update_config_for_local_hdfs
            return 0
        fi
    fi

    if [ "${cloud_storage_provider}" == "aws" ]; then
        update_config_for_aws
    elif [ "${cloud_storage_provider}" == "azure" ]; then
        update_config_for_azure
    elif [ "${cloud_storage_provider}" == "gcp" ]; then
        update_config_for_gcp
    elif [ "${cloud_storage_provider}" == "aliyun" ]; then
        update_config_for_aliyun
    elif [ "${cloud_storage_provider}" == "huaweicloud" ]; then
        update_config_for_huaweicloud
    elif [ "$HDFS_STORAGE" == "true" ]; then
        update_config_for_hdfs
    elif [ "$HDFS_ENABLED" == "true" ]; then
        update_config_for_local_hdfs
    fi
}

function update_nfs_dump_dir() {
    # set nfs gateway dump dir
    data_disk_dir=$(get_first_data_disk_dir)
    if [ -z "$data_disk_dir" ]; then
        nfs_dump_dir="/tmp/.hdfs-nfs"
    else
        nfs_dump_dir="$data_disk_dir/tmp/.hdfs-nfs"
    fi
    sed -i "s!{%dfs.nfs3.dump.dir%}!${nfs_dump_dir}!g" `grep "{%dfs.nfs3.dump.dir%}" -rl ./`
}

function update_local_storage_config_remote_hdfs() {
    REMOTE_HDFS_CONF_DIR=${HADOOP_HOME}/etc/remote
    # copy the existing hadoop conf
    mkdir -p ${REMOTE_HDFS_CONF_DIR}
    cp -r  ${HADOOP_HOME}/etc/hadoop/* ${REMOTE_HDFS_CONF_DIR}/

    fs_default_dir="${HDFS_NAMENODE_URI}"
    sed -i "s!{%remote.fs.default.name%}!${fs_default_dir}!g" ${output_dir}/hadoop/core-site-remote.xml

    # override with remote hdfs conf
    cp ${output_dir}/hadoop/core-site-remote.xml ${REMOTE_HDFS_CONF_DIR}/core-site.xml
    cp -r ${output_dir}/hadoop/hdfs-site.xml  ${REMOTE_HDFS_CONF_DIR}/
}

function update_local_storage_config_local_hdfs() {
    LOCAL_HDFS_CONF_DIR=${HADOOP_HOME}/etc/local
    # copy the existing hadoop conf
    mkdir -p ${LOCAL_HDFS_CONF_DIR}
    cp -r  ${HADOOP_HOME}/etc/hadoop/* ${LOCAL_HDFS_CONF_DIR}/

    fs_default_dir="hdfs://${HEAD_ADDRESS}:9000"
    sed -i "s!{%local.fs.default.name%}!${fs_default_dir}!g" ${output_dir}/hadoop/core-site-local.xml

    # override with local hdfs conf
    cp ${output_dir}/hadoop/core-site-local.xml ${LOCAL_HDFS_CONF_DIR}/core-site.xml
    cp -r ${output_dir}/hadoop/hdfs-site.xml  ${LOCAL_HDFS_CONF_DIR}/
}

function update_local_storage_config() {
    update_nfs_dump_dir

    if [ "${HDFS_STORAGE}" == "true" ]; then
        update_local_storage_config_remote_hdfs
    fi
    if [ "${HDFS_ENABLED}" == "true" ]; then
        update_local_storage_config_local_hdfs
    fi
}

function update_config_for_storage() {
    check_hdfs_storage
    set_cloud_storage_provider
    update_config_for_hadoop_storage
    update_local_storage_config

    if [ "${cloud_storage_provider}" != "none" ];then
        cp -r ${output_dir}/hadoop/${cloud_storage_provider}/core-site.xml ${HADOOP_HOME}/etc/hadoop/
    else
        # hdfs without cloud storage
        cp -r ${output_dir}/hadoop/core-site.xml ${HADOOP_HOME}/etc/hadoop/
    fi
}

function update_yarn_config() {
    yarn_scheduler_class="org.apache.hadoop.yarn.server.resourcemanager.scheduler.capacity.CapacityScheduler"
    if [ "${YARN_SCHEDULER}" == "fair" ];then
        yarn_scheduler_class="org.apache.hadoop.yarn.server.resourcemanager.scheduler.fair.FairScheduler"
    fi
    sed -i "s/{%yarn.resourcemanager.scheduler.class%}/${yarn_scheduler_class}/g" `grep "{%yarn.resourcemanager.scheduler.class%}" -rl ./`
    if [ $IS_HEAD_NODE == "true" ];then
        sed -i "s/{%yarn.scheduler.maximum-allocation-mb%}/${yarn_container_maximum_memory}/g" `grep "{%yarn.scheduler.maximum-allocation-mb%}" -rl ./`
        sed -i "s/{%yarn.scheduler.maximum-allocation-vcores%}/${yarn_container_maximum_vcores}/g" `grep "{%yarn.scheduler.maximum-allocation-vcores%}" -rl ./`
        sed -i "s/{%yarn.nodemanager.resource.memory-mb%}/${yarn_container_maximum_memory}/g" `grep "{%yarn.nodemanager.resource.memory-mb%}" -rl ./`
        sed -i "s/{%yarn.nodemanager.resource.cpu-vcores%}/${yarn_container_maximum_vcores}/g" `grep "{%yarn.nodemanager.resource.cpu-vcores%}" -rl ./`
    else
        sed -i "s/{%yarn.scheduler.maximum-allocation-mb%}/${total_memory}/g" `grep "{%yarn.scheduler.maximum-allocation-mb%}" -rl ./`
        sed -i "s/{%yarn.scheduler.maximum-allocation-vcores%}/${total_vcores}/g" `grep "{%yarn.scheduler.maximum-allocation-vcores%}" -rl ./`
        sed -i "s/{%yarn.nodemanager.resource.memory-mb%}/${total_memory}/g" `grep "{%yarn.nodemanager.resource.memory-mb%}" -rl ./`
        sed -i "s/{%yarn.nodemanager.resource.cpu-vcores%}/${total_vcores}/g" `grep "{%yarn.nodemanager.resource.cpu-vcores%}" -rl ./`
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
    if [ ! -z "$HIVE_METASTORE_URI" ] || [ "$METASTORE_ENABLED" == "true" ]; then
        if [ ! -z "$HIVE_METASTORE_URI" ]; then
            hive_metastore_uris="$HIVE_METASTORE_URI"
        else
            METASTORE_IP=${HEAD_ADDRESS}
            hive_metastore_uris="thrift://${METASTORE_IP}:9083"
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

    sed -i "s/HEAD_ADDRESS/${HEAD_ADDRESS}/g" `grep "HEAD_ADDRESS" -rl ./`

    update_yarn_config
    update_spark_runtime_config
    update_data_disks_config
    update_config_for_storage

    cp -r ${output_dir}/hadoop/yarn-site.xml ${HADOOP_HOME}/etc/hadoop/

    if [ $IS_HEAD_NODE == "true" ];then
        update_metastore_config

        cp -r ${output_dir}/spark/* ${SPARK_HOME}/conf
    fi
}

function configure_jupyter_for_spark() {
  if [ $IS_HEAD_NODE == "true" ]; then
      echo Y | jupyter lab --generate-config;
      # Set default password(cloudtik) for JupyterLab
      sed -i  "1 ic.NotebookApp.password = 'argon2:\$argon2id\$v=19\$m=10240,t=10,p=8\$Y+sBd6UhAyKNsI+/mHsy9g\$WzJsUujSzmotUkblSTpMwCFoOBVSwm7S5oOPzpC+tz8'" ~/.jupyter/jupyter_lab_config.py

      # Set default notebook_dir for JupyterLab
      export JUPYTER_WORKSPACE=${RUNTIME_PATH}/jupyter/notebooks
      mkdir -p $JUPYTER_WORKSPACE
      sed -i  "1 ic.NotebookApp.notebook_dir = '${JUPYTER_WORKSPACE}'" ~/.jupyter/jupyter_lab_config.py
      sed -i  "1 ic.NotebookApp.ip = '${HEAD_ADDRESS}'" ~/.jupyter/jupyter_lab_config.py
  fi
}

set_head_option "$@"
check_spark_installed
set_head_address
set_resources_for_spark
configure_system_folders
configure_hadoop_and_spark
configure_jupyter_for_spark
configure_cloud_fs

exit 0
