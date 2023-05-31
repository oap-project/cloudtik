#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

IS_HEAD_NODE=false
USER_HOME=/home/$(whoami)
RUNTIME_PATH=$USER_HOME/runtime

while true
do
    case "$1" in
    -h|--head)
        IS_HEAD_NODE=true
        ;;
    --)
        shift
        break
        ;;
    esac
    shift
done

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function prepare_base_conf() {
    source_dir=$(cd $(dirname ${BASH_SOURCE[0]})/..;pwd)/conf
    output_dir=/tmp/presto/conf
    rm -rf  $output_dir
    mkdir -p $output_dir
    cp -r $source_dir/* $output_dir
}

function check_presto_installed() {
    if [ ! -n "${PRESTO_HOME}" ]; then
        echo "Presto is not installed for PRESTO_HOME environment variable is not set."
        exit 1
    fi
}

function retrieve_resources() {
    jvm_max_memory=$(awk -v total_physical_memory=$(cloudtik node resources --memory --in-mb) 'BEGIN{print 0.8 * total_physical_memory}')
    jvm_max_memory=${jvm_max_memory%.*}
    query_max_memory_per_node=$(echo $jvm_max_memory | awk '{print $1*0.5}')
    query_max_memory_per_node=${query_max_memory_per_node%.*}
    query_max_total_memory_per_node=$(echo $jvm_max_memory | awk '{print $1*0.7}')
    query_max_total_memory_per_node=${query_max_total_memory_per_node%.*}
    memory_heap_headroom_per_node=$(echo $jvm_max_memory | awk '{print $1*0.25}')
    memory_heap_headroom_per_node=${memory_heap_headroom_per_node%.*}
}

function update_presto_data_disks_config() {
    presto_data_dir=""
    if [ -d "/mnt/cloudtik" ]; then
        for data_disk in /mnt/cloudtik/*; do
            [ -d "$data_disk" ] || continue
            if [ -z "$presto_data_dir" ]; then
                presto_data_dir=$data_disk/presto/data
            fi
        done
    fi

    # if no disks mounted on /mnt/cloudtik
    if [ -z "$presto_data_dir" ]; then
        presto_data_dir="${RUNTIME_PATH}/shared/presto/data"
    fi

    mkdir -p $presto_data_dir
    sed -i "s!{%node.data-dir%}!${presto_data_dir}!g" $output_dir/presto/node.properties
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
    AZURE_ENDPOINT="blob"
    sed -i "s#{%azure.storage.account%}#${AZURE_STORAGE_ACCOUNT}#g" $catalog_dir/hive-azure-core-site.xml
    sed -i "s#{%storage.endpoint%}#${AZURE_ENDPOINT}#g" $catalog_dir/hive-azure-core-site.xml
    sed -i "s#{%azure.account.key%}#${AZURE_ACCOUNT_KEY}#g" $catalog_dir/hive-azure-core-site.xml
}

function update_storage_config_for_azure() {
    if [ "$AZURE_STORAGE_TYPE" == "blob" ];then
        update_credential_config_for_azure

        HIVE_AZURE_CORE_SITE="${PRESTO_HOME}/etc/catalog/hive-azure-core-site.xml"
        cp $catalog_dir/hive-azure-core-site.xml ${HIVE_AZURE_CORE_SITE}
        sed -i "s!{%hive.config.resources%}!${HIVE_AZURE_CORE_SITE}!g" $catalog_dir/hive.config.properties
        cat $catalog_dir/hive.config.properties >> $catalog_dir/hive.properties
    else
        # datalake is not supported
        echo "WARNING: Azure Data Lake Storage Gen 2 is not supported for this version."
    fi
}

function update_storage_config_for_gcp() {
    # GCP_PROJECT_ID
    # GCP_GCS_SERVICE_ACCOUNT_CLIENT_EMAIL
    # GCP_GCS_SERVICE_ACCOUNT_PRIVATE_KEY_ID
    # GCP_GCS_SERVICE_ACCOUNT_PRIVATE_KEY
    if [ ! -z "$GCP_GCS_SERVICE_ACCOUNT_PRIVATE_KEY_ID" ]; then
        sed -i "s#{%project_id%}#${GCP_PROJECT_ID}#g" $catalog_dir/gcs.key-file.json
        sed -i "s#{%private_key_id%}#${GCP_GCS_SERVICE_ACCOUNT_CLIENT_EMAIL}#g" $catalog_dir/gcs.key-file.json
        sed -i "s#{%private_key%}#${GCP_GCS_SERVICE_ACCOUNT_PRIVATE_KEY_ID}#g" $catalog_dir/gcs.key-file.json
        sed -i "s#{%client_email%}#${GCP_GCS_SERVICE_ACCOUNT_PRIVATE_KEY}#g" $catalog_dir/gcs.key-file.json

        cp $catalog_dir/gcs.key-file.json ${PRESTO_HOME}/etc/catalog/gcs.key-file.json

        sed -i "s#{%gcs.use-access-token%}#false#g" $catalog_dir/hive.gcs.properties
        sed -i "s!{%gcs.json-key-file-path%}!${PRESTO_HOME}/etc/catalog/gcs.key-file.json!g" $catalog_dir/hive.gcs.properties
    else
        sed -i "s#{%gcs.use-access-token%}#true#g" $catalog_dir/hive.gcs.properties
        sed -i "s#{%gcs.json-key-file-path%}##g" $catalog_dir/hive.gcs.properties
    fi

    cat $catalog_dir/hive.gcs.properties >> $catalog_dir/hive.properties
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

function update_storage_config() {
    set_cloud_storage_provider
    if [ "${cloud_storage_provider}" == "aws" ]; then
        update_storage_config_for_aws
    elif [ "${cloud_storage_provider}" == "azure" ]; then
        update_storage_config_for_azure
    elif [ "${cloud_storage_provider}" == "gcp" ]; then
        update_storage_config_for_gcp
    fi
}

function update_hive_metastore_config() {
    # To be improved for external metastore cluster
    catalog_dir=$output_dir/presto/catalog
    hive_properties=${catalog_dir}/hive.properties
    if [ "$METASTORE_ENABLED" == "true" ] || [ ! -z "$HIVE_METASTORE_URI" ]; then
        if [ "$METASTORE_ENABLED" == "true" ]; then
            METASTORE_IP=${HEAD_ADDRESS}
            hive_metastore_uris="thrift://${METASTORE_IP}:9083"
        else
            hive_metastore_uris="$HIVE_METASTORE_URI"
        fi

        sed -i "s!{%HIVE_METASTORE_URI%}!${hive_metastore_uris}!g" ${hive_properties}

        mkdir -p ${PRESTO_HOME}/etc/catalog

        update_storage_config

        cp ${hive_properties}  ${PRESTO_HOME}/etc/catalog/hive.properties
    fi
}

function update_metastore_config() {
    update_hive_metastore_config
}

function update_presto_memory_config() {
    if [ ! -z "$PRESTO_JVM_MAX_MEMORY" ]; then
        jvm_max_memory=$PRESTO_JVM_MAX_MEMORY
    fi
    if [ ! -z "$PRESTO_MAX_MEMORY_PER_NODE" ]; then
        query_max_memory_per_node=$PRESTO_MAX_MEMORY_PER_NODE
    fi
    if [ ! -z "$PRESTO_MAX_TOTAL_MEMORY_PER_NODE" ]; then
        query_max_total_memory_per_node=$PRESTO_MAX_TOTAL_MEMORY_PER_NODE
    fi

    if [ ! -z "$PRESTO_HEAP_HEADROOM_PER_NODE" ]; then
        memory_heap_headroom_per_node=$PRESTO_HEAP_HEADROOM_PER_NODE
    fi

    query_max_memory="50GB"
    if [ ! -z "$PRESTO_QUERY_MAX_MEMORY" ]; then
        query_max_memory=$PRESTO_QUERY_MAX_MEMORY
    fi

    sed -i "s/{%jvm.max-memory%}/${jvm_max_memory}m/g" `grep "{%jvm.max-memory%}" -rl ./`
    sed -i "s/{%query.max-memory-per-node%}/${query_max_memory_per_node}MB/g" `grep "{%query.max-memory-per-node%}" -rl ./`
    sed -i "s/{%query.max-total-memory-per-node%}/${query_max_total_memory_per_node}MB/g" `grep "{%query.max-total-memory-per-node%}" -rl ./`
    sed -i "s/{%memory.heap-headroom-per-node%}/${memory_heap_headroom_per_node}MB/g" `grep "{%memory.heap-headroom-per-node%}" -rl ./`

    sed -i "s/{%query.max-memory%}/${query_max_memory}/g" `grep "{%query.max-memory%}" -rl ./`
}

function configure_presto() {
    prepare_base_conf
    update_metastore_config

    cd $output_dir
    node_id=$(uuid)

    sed -i "s/{%HEAD_ADDRESS%}/${HEAD_ADDRESS}/g" `grep "{%HEAD_ADDRESS%}" -rl ./`
    sed -i "s/{%node.environment%}/presto/g" $output_dir/presto/node.properties
    sed -i "s/{%node.id%}/${node_id}/g" $output_dir/presto/node.properties

    update_presto_memory_config
    update_presto_data_disks_config

    mkdir -p ${PRESTO_HOME}/etc
    if [ $IS_HEAD_NODE == "true" ]; then
        cp ${output_dir}/presto/config.properties  ${PRESTO_HOME}/etc/config.properties
    else
        cp ${output_dir}/presto/config.worker.properties  ${PRESTO_HOME}/etc/config.properties
    fi

    cp ${output_dir}/presto/jvm.config  ${PRESTO_HOME}/etc/jvm.config
    cp ${output_dir}/presto/node.properties  ${PRESTO_HOME}/etc/node.properties
}

check_presto_installed
set_head_address
retrieve_resources
configure_presto

exit 0
