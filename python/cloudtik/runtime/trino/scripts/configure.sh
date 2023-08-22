#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

USER_HOME=/home/$(whoami)
RUNTIME_PATH=$USER_HOME/runtime

HADOOP_CREDENTIAL_FILE="${TRINO_HOME}/etc/catalog/hadoop_credential.jceks"
HADOOP_CREDENTIAL_PROPERTY="<property>\n      <name>hadoop.security.credential.provider.path</name>\n      <value>jceks://file@${HADOOP_CREDENTIAL_FILE}</value>\n    </property>"

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function prepare_base_conf() {
    source_dir=$(dirname "${BIN_DIR}")/conf
    output_dir=/tmp/trino/conf
    rm -rf  $output_dir
    mkdir -p $output_dir
    cp -r $source_dir/* $output_dir
}

function check_trino_installed() {
    if [ ! -n "${HADOOP_HOME}" ]; then
        echo "HADOOP_HOME environment variable is not set."
        exit 1
    fi

    if [ ! -n "${TRINO_HOME}" ]; then
        echo "Trino is not installed for TRINO_HOME environment variable is not set."
        exit 1
    fi
}

function retrieve_resources() {
    jvm_max_memory=$(awk -v total_physical_memory=$(cloudtik node resources --memory --in-mb) 'BEGIN{print 0.8 * total_physical_memory}')
    jvm_max_memory=${jvm_max_memory%.*}
    query_max_memory_per_node=$(echo $jvm_max_memory | awk '{print $1*0.5}')
    query_max_memory_per_node=${query_max_memory_per_node%.*}
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

    if [ "$AZURE_STORAGE_TYPE" != "blob" ];then
        # datalake
        if [ -n  "${AZURE_ACCOUNT_KEY}" ];then
            sed -i "s#{%auth.type%}#SharedKey#g" $catalog_dir/hive-azure-core-site.xml
        else
            sed -i "s#{%auth.type%}##g" $catalog_dir/hive-azure-core-site.xml
        fi
    fi

    # Use hadoop credential files so that the client id is properly set to worker client id
    HAS_HADOOP_CREDENTIAL=false
    HADOOP_CREDENTIAL_TMP_FILE="${output_dir}/credential.jceks"
    HADOOP_CREDENTIAL_TMP_PROVIDER_PATH="jceks://file@${HADOOP_CREDENTIAL_TMP_FILE}"
    if [ ! -z "${AZURE_ACCOUNT_KEY}" ]; then
        FS_KEY_NAME_ACCOUNT_KEY="fs.azure.account.key.${AZURE_STORAGE_ACCOUNT}.${AZURE_ENDPOINT}.core.windows.net"
        ${HADOOP_HOME}/bin/hadoop credential create ${FS_KEY_NAME_ACCOUNT_KEY} -value ${AZURE_ACCOUNT_KEY} -provider ${HADOOP_CREDENTIAL_TMP_PROVIDER_PATH} > /dev/null
        HAS_HADOOP_CREDENTIAL=true
    fi

    if [ ! -z "${AZURE_MANAGED_IDENTITY_TENANT_ID}" ]; then
        FS_KEY_NAME_TENANT_ID="fs.azure.account.oauth2.msi.tenant"
        ${HADOOP_HOME}/bin/hadoop credential create ${FS_KEY_NAME_TENANT_ID} -value ${AZURE_MANAGED_IDENTITY_TENANT_ID} -provider ${HADOOP_CREDENTIAL_TMP_PROVIDER_PATH} > /dev/null
        HAS_HADOOP_CREDENTIAL=true
    fi

    if [ ! -z "${AZURE_MANAGED_IDENTITY_CLIENT_ID}" ]; then
        FS_KEY_NAME_CLIENT_ID="fs.azure.account.oauth2.client.id"
        ${HADOOP_HOME}/bin/hadoop credential create ${FS_KEY_NAME_CLIENT_ID} -value ${AZURE_MANAGED_IDENTITY_CLIENT_ID} -provider ${HADOOP_CREDENTIAL_TMP_PROVIDER_PATH} > /dev/null
        HAS_HADOOP_CREDENTIAL=true
    fi
    if [  -f "$HADOOP_CREDENTIAL_TMP_FILE"  ]; then
        cp  ${HADOOP_CREDENTIAL_TMP_FILE} ${HADOOP_CREDENTIAL_FILE}
    fi
    if [ "${HAS_HADOOP_CREDENTIAL}" == "true" ]; then
        sed -i "s#{%hadoop.credential.property%}#${HADOOP_CREDENTIAL_PROPERTY}#g" $catalog_dir/hive-azure-core-site.xml
    else
        sed -i "s#{%hadoop.credential.property%}#""#g" $catalog_dir/hive-azure-core-site.xml
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
    # GCP_PROJECT_ID
    # GCP_GCS_SERVICE_ACCOUNT_CLIENT_EMAIL
    # GCP_GCS_SERVICE_ACCOUNT_PRIVATE_KEY_ID
    # GCP_GCS_SERVICE_ACCOUNT_PRIVATE_KEY
    if [ ! -z "$GCP_GCS_SERVICE_ACCOUNT_PRIVATE_KEY_ID" ]; then
        sed -i "s#{%project_id%}#${GCP_PROJECT_ID}#g" $catalog_dir/gcs.key-file.json
        sed -i "s#{%private_key_id%}#${GCP_GCS_SERVICE_ACCOUNT_CLIENT_EMAIL}#g" $catalog_dir/gcs.key-file.json
        sed -i "s#{%private_key%}#${GCP_GCS_SERVICE_ACCOUNT_PRIVATE_KEY_ID}#g" $catalog_dir/gcs.key-file.json
        sed -i "s#{%client_email%}#${GCP_GCS_SERVICE_ACCOUNT_PRIVATE_KEY}#g" $catalog_dir/gcs.key-file.json

        cp $catalog_dir/gcs.key-file.json ${TRINO_HOME}/etc/catalog/gcs.key-file.json

        sed -i "s#{%gcs.use-access-token%}#false#g" $catalog_dir/hive.gcs.properties
        sed -i "s!{%gcs.json-key-file-path%}!${TRINO_HOME}/etc/catalog/gcs.key-file.json!g" $catalog_dir/hive.gcs.properties
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
    catalog_dir=$output_dir/trino/catalog
    hive_properties=${catalog_dir}/hive.properties
    if [ ! -z "$HIVE_METASTORE_URI" ] || [ "$METASTORE_ENABLED" == "true" ]; then
        if [ ! -z "$HIVE_METASTORE_URI" ]; then
            hive_metastore_uri="$HIVE_METASTORE_URI"
        else
            METASTORE_IP=${HEAD_ADDRESS}
            hive_metastore_uri="thrift://${METASTORE_IP}:9083"
        fi

        sed -i "s!{%HIVE_METASTORE_URI%}!${hive_metastore_uri}!g" ${hive_properties}
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

    if [ ! -z "$TRINO_HEAP_HEADROOM_PER_NODE" ]; then
        memory_heap_headroom_per_node=$TRINO_HEAP_HEADROOM_PER_NODE
    fi

    query_max_memory="50GB"
    if [ ! -z "$TRINO_QUERY_MAX_MEMORY" ]; then
        query_max_memory=$TRINO_QUERY_MAX_MEMORY
    fi

    sed -i "s/{%jvm.max-memory%}/${jvm_max_memory}m/g" `grep "{%jvm.max-memory%}" -rl ${output_dir}`
    sed -i "s/{%query.max-memory-per-node%}/${query_max_memory_per_node}MB/g" `grep "{%query.max-memory-per-node%}" -rl ${output_dir}`
    sed -i "s/{%memory.heap-headroom-per-node%}/${memory_heap_headroom_per_node}MB/g" `grep "{%memory.heap-headroom-per-node%}" -rl ${output_dir}`

    sed -i "s/{%query.max-memory%}/${query_max_memory}/g" `grep "{%query.max-memory%}" -rl ${output_dir}`
}

function configure_trino() {
    prepare_base_conf
    update_metastore_config

    node_id=$(uuid)

    sed -i "s/{%HEAD_ADDRESS%}/${HEAD_ADDRESS}/g" `grep "{%HEAD_ADDRESS%}" -rl ${output_dir}`
    sed -i "s/{%node.environment%}/trino/g" $output_dir/trino/node.properties
    sed -i "s/{%node.id%}/${node_id}/g" $output_dir/trino/node.properties

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

set_head_option "$@"
check_trino_installed
set_head_address
retrieve_resources
configure_trino

exit 0
