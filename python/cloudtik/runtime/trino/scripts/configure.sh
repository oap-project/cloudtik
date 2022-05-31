#!/bin/bash

args=$(getopt -a -o h::p: -l head::,node_ip_address::,head_address:: -- "$@")
eval set -- "${args}"

IS_HEAD_NODE=false
USER_HOME=/home/$(whoami)
RUNTIME_PATH=$USER_HOME/runtime

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
    output_dir=/tmp/trino/conf
    rm -rf  $output_dir
    mkdir -p $output_dir
    cp -r $source_dir/* $output_dir
}

function check_trino_installed() {
    if [ ! -n "${TRINO_HOME}" ]; then
        echo "Trino is not installed for TRINO_HOME environment variable is not set."
        exit 1
    fi
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


function retrieve_resources() {
    jvm_max_memory=$(awk '($1 == "MemTotal:"){print $2/1024*0.8}' /proc/meminfo)
    jvm_max_memory=${jvm_max_memory%.*}
    query_max_memory_per_node=$(echo $jvm_max_memory | awk '{print $1*0.5}')
    query_max_memory_per_node=${query_max_memory_per_node%.*}
    query_max_total_memory_per_node=$(echo $jvm_max_memory | awk '{print $1*0.7}')
    query_max_total_memory_per_node=${query_max_total_memory_per_node%.*}
    memory_heap_headroom_per_node=$(echo $jvm_max_memory | awk '{print $1*0.25}')
    memory_heap_headroom_per_node=${memory_heap_headroom_per_node%.*}
}

function update_trino_data_disks_config() {
    trino_data_dir=""
    if [ -d "/mnt/cloudtik" ]; then
        for data_disk in /mnt/cloudtik/*; do
            [ -d "$data_disk" ] || continue
            if [ -z "$trino_data_dir" ]; then
                trino_data_dir=$data_disk/trino/data
            fi
        done
    fi

    # if no disks mounted on /mnt/cloudtik
    if [ -z "$trino_data_dir" ]; then
        trino_data_dir="${RUNTIME_PATH}/shared/trino/data"
    fi

    mkdir -p $trino_data_dir
    sed -i "s!{%node.data-dir%}!${trino_data_dir}!g" $output_dir/trino/node.properties
}

function update_storage_config_for_aws() {
    # AWS_S3_ACCESS_KEY_ID
    # AWS_S3_SECRET_ACCESS_KEY
    # Since hive.s3.use-instance-credentials is default true
    if [ ! -z "$AWS_S3_ACCESS_KEY_ID" ]; then
        sed -i "s#{%s3.aws-access-key%}#${AWS_S3_ACCESS_KEY_ID}#g" $catalog_dir/hive.s3.properties
        sed -i "s#{%s3.aws-secret-key%}#${AWS_S3_SECRET_ACCESS_KEY}#g" $catalog_dir/hive.s3.properties
        cat $catalog_dir/hive.s3.properties >> $catalog_dir/hive.properties
    fi
}

function update_credential_config_for_azure() {
    if [ "$AZURE_STORAGE_TYPE" == "blob" ];then
        AZURE_ENDPOINT="blob"
    else
        # Default to datalake
        AZURE_ENDPOINT="dfs"
    fi

    sed -i "s#{%azure.storage.account%}#${AZURE_STORAGE_ACCOUNT}#g" $catalog_dir/hive-azure-core-site.xml
    sed -i "s#{%storage.endpoint%}#${AZURE_ENDPOINT}#g" $catalog_dir/hive-azure-core-site.xml

    sed -i "s#{%azure.account.key%}#${AZURE_ACCOUNT_KEY}#g" $catalog_dir/hive-azure-core-site.xml
    sed -i "s#{%fs.azure.account.oauth2.msi.tenant%}#${AZURE_MANAGED_IDENTITY_TENANT_ID}#g" $catalog_dir/hive-azure-core-site.xml
    sed -i "s#{%fs.azure.account.oauth2.client.id%}#${AZURE_MANAGED_IDENTITY_CLIENT_ID}#g" $catalog_dir/hive-azure-core-site.xml

    if [ "$AZURE_STORAGE_TYPE" != "blob" ];then
        # datalake
        if [ -n  "${AZURE_ACCOUNT_KEY}" ];then
            sed -i "s#{%auth.type%}#SharedKey#g" $catalog_dir/hive-azure-core-site.xml
        else
            sed -i "s#{%auth.type%}##g" $catalog_dir/hive-azure-core-site.xml
        fi
    fi
}

function update_storage_config_for_azure() {
    # Use Hadoop core-site.xml configurations directly
    update_credential_config_for_azure

    HIVE_AZURE_CORE_SITE="${TRINO_HOME}/etc/catalog/hive-azure-core-site.xml"
    cp $catalog_dir/hive-azure-core-site.xml ${HIVE_AZURE_CORE_SITE}
    sed -i "s!{%hive.config.resources%}!${HIVE_AZURE_CORE_SITE}!g" $catalog_dir/hive.config.properties
    cat $catalog_dir/hive.config.properties >> $catalog_dir/hive.properties
}

function update_storage_config_for_gcp() {
    # PROJECT_ID
    # GCS_SERVICE_ACCOUNT_CLIENT_EMAIL
    # GCS_SERVICE_ACCOUNT_PRIVATE_KEY_ID
    # GCS_SERVICE_ACCOUNT_PRIVATE_KEY
    if [ ! -z "$GCS_SERVICE_ACCOUNT_PRIVATE_KEY_ID" ]; then
        sed -i "s#{%project_id%}#${PROJECT_ID}#g" $catalog_dir/gcs.key-file.json
        sed -i "s#{%private_key_id%}#${GCS_SERVICE_ACCOUNT_CLIENT_EMAIL}#g" $catalog_dir/gcs.key-file.json
        sed -i "s#{%private_key%}#${GCS_SERVICE_ACCOUNT_PRIVATE_KEY_ID}#g" $catalog_dir/gcs.key-file.json
        sed -i "s#{%client_email%}#${GCS_SERVICE_ACCOUNT_PRIVATE_KEY}#g" $catalog_dir/gcs.key-file.json

        cp $catalog_dir/gcs.key-file.json ${TRINO_HOME}/etc/catalog/gcs.key-file.json

        sed -i "s#{%gcs.use-access-token%}#false#g" $catalog_dir/hive.gcs.properties
        sed -i "s!{%gcs.json-key-file-path%}!${TRINO_HOME}/etc/catalog/gcs.key-file.json!g" $catalog_dir/hive.gcs.properties
    else
        sed -i "s#{%gcs.use-access-token%}#true#g" $catalog_dir/hive.gcs.properties
        sed -i "s#{%gcs.json-key-file-path%}##g" $catalog_dir/hive.gcs.properties
    fi

    cat $catalog_dir/hive.gcs.properties >> $catalog_dir/hive.properties
}

function update_storage_config() {
    if [ "$CLOUDTIK_PROVIDER_TYPE" == "aws" ]; then
        update_storage_config_for_aws
    elif [ "$CLOUDTIK_PROVIDER_TYPE" == "gcp" ]; then
        update_storage_config_for_gcp
    elif [ "$CLOUDTIK_PROVIDER_TYPE" == "azure" ]; then
        update_storage_config_for_azure
    fi
}

function update_hive_metastore_config() {
    # To be improved for external metastore cluster
    catalog_dir=$output_dir/trino/catalog
    hive_properties=${catalog_dir}/hive.properties
    if [ "$METASTORE_ENABLED" == "true" ] || [ ! -z "$HIVE_METASTORE_URI" ]; then
        if [ "$METASTORE_ENABLED" == "true" ]; then
            METASTORE_IP=${HEAD_ADDRESS}
            hive_metastore_uris="thrift://${METASTORE_IP}:9083"
        else
            hive_metastore_uris="$HIVE_METASTORE_URI"
        fi

        sed -i "s!{%HIVE_METASTORE_URI%}!${hive_metastore_uris}!g" ${hive_properties}

        mkdir -p ${TRINO_HOME}/etc/catalog

        update_storage_config

        cp ${hive_properties}  ${TRINO_HOME}/etc/catalog/hive.properties
    fi
}

function update_metastore_config() {
    update_hive_metastore_config
}

function update_trino_memory_config() {
    if [ ! -z "$TRINO_JVM_MAX_MEMORY" ]; then
        jvm_max_memory=$TRINO_JVM_MAX_MEMORY
    fi
    if [ ! -z "$TRINO_MAX_MEMORY_PER_NODE" ]; then
        query_max_memory_per_node=$TRINO_MAX_MEMORY_PER_NODE
    fi
    if [ ! -z "$TRINO_MAX_TOTAL_MEMORY_PER_NODE" ]; then
        query_max_total_memory_per_node=$TRINO_MAX_TOTAL_MEMORY_PER_NODE
    fi

    if [ ! -z "$TRINO_HEAP_HEADROOM_PER_NODE" ]; then
        memory_heap_headroom_per_node=$TRINO_HEAP_HEADROOM_PER_NODE
    fi

    query_max_memory="50GB"
    if [ ! -z "$TRINO_QUERY_MAX_MEMORY" ]; then
        query_max_memory=$TRINO_QUERY_MAX_MEMORY
    fi

    sed -i "s/{%jvm.max-memory%}/${jvm_max_memory}m/g" `grep "{%jvm.max-memory%}" -rl ./`
    sed -i "s/{%query.max-memory-per-node%}/${query_max_memory_per_node}MB/g" `grep "{%query.max-memory-per-node%}" -rl ./`
    sed -i "s/{%query.max-total-memory-per-node%}/${query_max_total_memory_per_node}MB/g" `grep "{%query.max-total-memory-per-node%}" -rl ./`
    sed -i "s/{%memory.heap-headroom-per-node%}/${memory_heap_headroom_per_node}MB/g" `grep "{%memory.heap-headroom-per-node%}" -rl ./`

    sed -i "s/{%query.max-memory%}/${query_max_memory}/g" `grep "{%query.max-memory%}" -rl ./`
}

function configure_trino() {
    prepare_base_conf
    update_metastore_config

    cd $output_dir
    node_id=$(uuid)
    trino_log_dir=${TRINO_HOME}/logs
    mkdir -p ${trino_log_dir}
    sed -i "s/{%HEAD_ADDRESS%}/${HEAD_ADDRESS}/g" `grep "{%HEAD_ADDRESS%}" -rl ./`
    sed -i "s/{%node.environment%}/trino/g" $output_dir/trino/node.properties
    sed -i "s/{%node.id%}/${node_id}/g" $output_dir/trino/node.properties
    sed -i "s!{%node.log-dir%}!${trino_log_dir}!g" $output_dir/trino/node.properties

    update_trino_memory_config
    update_trino_data_disks_config

    mkdir -p ${TRINO_HOME}/etc
    if [ $IS_HEAD_NODE == "true" ]; then
        cp ${output_dir}/trino/config.properties  ${TRINO_HOME}/etc/config.properties
    else
        cp ${output_dir}/trino/config.worker.properties  ${TRINO_HOME}/etc/config.properties
    fi

    cp ${output_dir}/trino/jvm.config  ${TRINO_HOME}/etc/jvm.config
    cp ${output_dir}/trino/node.properties  ${TRINO_HOME}/etc/node.properties
}

check_trino_installed
set_head_address
retrieve_resources
configure_trino

exit 0
