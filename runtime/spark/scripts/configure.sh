#!/bin/bash

echo original parameters=[$@]
args=$(getopt -a -o h::p: -l head_address::,provider:,aws_s3a_bucket::,s3a_access_key::,s3a_secret_key::,project_id::,gcp_gcs_bucket::,fs_gs_auth_service_account_email::,fs_gs_auth_service_account_private_key_id::,fs_gs_auth_service_account_private_key::,azure_storage_kind::,azure_storage_account::,azure_container::,azure_account_key:: -- "$@")
echo ARGS=[$args]
eval set -- "${args}"
echo formatted parameters=[$@]

while true
do
    case "$1" in
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


function check_head_or_worker() {
    if [ ! -n "${HEAD_ADDRESS}" ]; then
	    HEAD_ADDRESS=$(hostname -I | awk '{print $1}')
	    Is_head_node=true
    else
	    Is_head_node=false
    fi
}


function caculate_worker_resources() {
    #For nodemanager
    total_memory=$(awk '($1 == "MemTotal:"){print $2/1024}' /proc/meminfo)
    total_memory=${total_memory%.*}
    total_vcores=$(cat /proc/cpuinfo | grep processor | wc -l)
}


function set_resources_for_spark() {
    #For Head Node
    if [ $Is_head_node == "true" ];then
        spark_executor_cores=$(cat ~/cloudtik_bootstrap_config.yaml | jq '."spark_executor_resource"."spark_executor_cores"')
        spark_executor_memory=$(cat ~/cloudtik_bootstrap_config.yaml | jq '."spark_executor_resource"."spark_executor_memory"')M
        spark_driver_memory=$(cat ~/cloudtik_bootstrap_config.yaml | jq '."spark_executor_resource"."spark_driver_memory"')M
    fi
}

function update_aws_hadoop_config() {
    sed -i "s#{%aws.s3a.bucket%}#${AWS_S3A_BUCKET}#g" `grep "{%aws.s3a.bucket%}" -rl ./`
    sed -i "s#{%fs.s3a.access.key%}#${FS_S3A_ACCESS_KEY}#g" `grep "{%fs.s3a.access.key%}" -rl ./`
    sed -i "s#{%fs.s3a.secret.key%}#${FS_S3A_SECRET_KEY}#g" `grep "{%fs.s3a.secret.key%}" -rl ./`
}

function update_gcp_hadoop_config() {
    sed -i "s#{%project_id%}#${PROJECT_ID}#g" `grep "{%project_id%}" -rl ./`
    sed -i "s#{%gcp.gcs.bucket%}#${GCP_GCS_BUCKET}#g" `grep "{%gcp.gcs.bucket%}" -rl ./`
    sed -i "s#{%fs.gs.auth.service.account.email%}#${FS_GS_AUTH_SERVICE_ACCOUNT_EMAIL}#g" `grep "{%fs.gs.auth.service.account.email%}" -rl ./`
    sed -i "s#{%fs.gs.auth.service.account.private.key.id%}#${FS_GS_AUTH_SERVICE_ACCOUNT_PRIVATE_KEY_ID}#g" `grep "{%fs.gs.auth.service.account.private.key.id%}" -rl ./`
    private_key_has_open_quote=${FS_GS_AUTH_SERVICE_ACCOUNT_PRIVATE_KEY%\"}
    private_key=${private_key_has_open_quote#\"}
    sed -i "s#{%fs.gs.auth.service.account.private.key%}#${private_key}#g" `grep "{%fs.gs.auth.service.account.private.key%}" -rl ./`
}

function update_azure_hadoop_config() {
    sed -i "s#{%azure.storage.account%}#${AZURE_STORAGE_ACCOUNT}#g" "$(grep "{%azure.storage.account%}" -rl ./)"
    sed -i "s#{%azure.container%}#${AZURE_CONTAINER}#g" "$(grep "{%azure.container%}" -rl ./)"
    sed -i "s#{%azure.account.key%}#${AZURE_ACCOUNT_KEY}#g" "$(grep "{%azure.account.key%}" -rl ./)"
    if [ $AZURE_STORAGE_KIND == "wasbs" ];then
      sed -i "s#{%azure.storage.kind%}#wasbs#g" "$(grep "{%azure.storage.kind%}" -rl ./)"
      sed -i "s#{%storage.endpoint%}#blob#g" "$(grep "{%storage.endpoint%}" -rl ./)"
    elif [ $AZURE_STORAGE_KIND == "abfs" ];then
      sed -i "s#{%azure.storage.kind%}#abfs#g" "$(grep "{%azure.storage.kind%}" -rl ./)"
      sed -i "s#{%storage.endpoint%}#dfs#g" "$(grep "{%storage.endpoint%}" -rl ./)"
      sed -i "s#{%auth.type%}#SharedKey#g" "$(grep "{%auth.type%}" -rl ./)"
    else
       echo "Azure storage kind must be wasbs(Azure Blob storage) or abfs(Azure Data Lake Gen 2)"
    fi

}

function update_spark_runtime_config() {
    cd $output_dir
    sed -i "s/HEAD_ADDRESS/${HEAD_ADDRESS}/g" `grep "HEAD_ADDRESS" -rl ./`
    sed -i "s!{%HADOOP_HOME%}!${HADOOP_HOME}!g" `grep "{%HADOOP_HOME%}" -rl ./`
    sed -i "s/{%yarn.scheduler.maximum-allocation-mb%}/${total_memory}/g" `grep "{%yarn.scheduler.maximum-allocation-mb%}" -rl ./`
    sed -i "s/{%yarn.nodemanager.resource.memory-mb%}/${total_memory}/g" `grep "{%yarn.nodemanager.resource.memory-mb%}" -rl ./`
    sed -i "s/{%yarn.nodemanager.resource.cpu-vcores%}/${total_vcores}/g" `grep "{%yarn.nodemanager.resource.cpu-vcores%}" -rl ./`
    sed -i "s/{%yarn.scheduler.maximum-allocation-vcores%}/${total_vcores}/g" `grep "{%yarn.scheduler.maximum-allocation-vcores%}" -rl ./`

    if [ $Is_head_node == "true" ];then
	    sed -i "s/{%spark.executor.cores%}/${spark_executor_cores}/g" `grep "{%spark.executor.cores%}" -rl ./`
	    sed -i "s/{%spark.executor.memory%}/${spark_executor_memory}/g" `grep "{%spark.executor.memory%}" -rl ./`
	    sed -i "s/{%spark.driver.memory%}/${spark_driver_memory}/g" `grep "{%spark.driver.memory%}" -rl ./`
    fi

    if [ "$provider" == "aws" ]; then
      update_aws_hadoop_config
    fi

    if [ "$provider" == "gcp" ]; then
      update_gcp_hadoop_config
    fi

    if [ "$provider" == "azure" ]; then
      update_azure_hadoop_config
    fi

    cp -r ${output_dir}/hadoop/${provider}/core-site.xml  ${HADOOP_HOME}/etc/hadoop/
    cp -r ${output_dir}/hadoop/yarn-site.xml  ${HADOOP_HOME}/etc/hadoop/

    if [ $Is_head_node == "true" ];then
	    cp -r ${output_dir}/spark/*  ${SPARK_HOME}/conf
    fi
}


function prepare_spark_jars() {
    #cp spark jars to hadoop path
    jars=('spark-[0-9]*[0-9]-yarn-shuffle.jar' 'jackson-databind-[0-9]*[0-9].jar' 'jackson-core-[0-9]*[0-9].jar' 'jackson-annotations-[0-9]*[0-9].jar' 'metrics-core-[0-9]*[0-9].jar' 'netty-all-[0-9]*[0-9].Final.jar' 'commons-lang3-[0-9]*[0-9].jar')
    find ${HADOOP_HOME}/share/hadoop/yarn/lib -name netty-all-[0-9]*[0-9].Final.jar| xargs -i mv -f {} {}.old
    # Download gcs-connector to ${HADOOP_HOME}/share/hadoop/tools/lib/* for gcp cloud storage support
    wget -nc -P "${HADOOP_HOME}"/share/hadoop/tools/lib/  https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop3-2.2.0.jar
    # Download jetty-util to ${HADOOP_HOME}/share/hadoop/tools/lib/* for Azure cloud storage support
    wget -nc -P "${HADOOP_HOME}"/share/hadoop/tools/lib/  https://repo1.maven.org/maven2/org/eclipse/jetty/jetty-util-ajax/9.3.24.v20180605/jetty-util-ajax-9.3.24.v20180605.jar
    wget -nc -P "${HADOOP_HOME}"/share/hadoop/tools/lib/  https://repo1.maven.org/maven2/org/eclipse/jetty/jetty-util/9.3.24.v20180605/jetty-util-9.3.24.v20180605.jar
    for jar in ${jars[@]};
    do
	    find ${SPARK_HOME}/jars/ -name $jar | xargs -i cp {} ${HADOOP_HOME}/share/hadoop/yarn/lib;
	    find ${SPARK_HOME}/yarn/ -name $jar | xargs -i cp {} ${HADOOP_HOME}/share/hadoop/yarn/lib;
    done
}


function set_hadoop_classpath() {
    #Add share/hadoop/tools/lib/* into classpath
    echo "export HADOOP_CLASSPATH=$HADOOP_CLASSPATH:\$HADOOP_HOME/share/hadoop/tools/lib/*" >> ${HADOOP_HOME}/etc/hadoop/hadoop-env.sh
}


check_env
prepare_base_conf
check_head_or_worker
caculate_worker_resources
set_resources_for_spark
update_spark_runtime_config
prepare_spark_jars
set_hadoop_classpath
