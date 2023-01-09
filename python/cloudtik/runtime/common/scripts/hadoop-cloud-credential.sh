#!/bin/bash

# assumptions for using the functions of this script:
# 1. cloud_storage_provider variable is set to the name of supported public providers. For example, "aws", "gcp" or "azure"
# 2. Credential values are exported through the environment variables through provider.with_environment_variables.
# 3. Current dir is set to the root of the working conf.
# 4. HADOOP_HOME is set to the hadoop installation home.

HADOOP_CREDENTIAL_FILE="${HADOOP_HOME}/etc/hadoop/credential.jceks"
HADOOP_CREDENTIAL_PROPERTY="<property>\n      <name>hadoop.security.credential.provider.path</name>\n      <value>jceks://file@${HADOOP_CREDENTIAL_FILE}</value>\n    </property>"

function update_credential_config_for_aws() {
    if [ "$AWS_WEB_IDENTITY" == "true" ]; then
        # Replace with InstanceProfileCredentialsProvider with WebIdentityTokenCredentialsProvider for Kubernetes
        sed -i "s#InstanceProfileCredentialsProvider#WebIdentityTokenCredentialsProvider#g" `grep "InstanceProfileCredentialsProvider" -rl ./`
    fi

    sed -i "s#{%fs.s3a.access.key%}#${AWS_S3_ACCESS_KEY_ID}#g" `grep "{%fs.s3a.access.key%}" -rl ./`

    if [ ! -z "${AWS_S3_SECRET_ACCESS_KEY}" ]; then
        ${HADOOP_HOME}/bin/hadoop credential create fs.s3a.secret.key -value ${AWS_S3_SECRET_ACCESS_KEY} -provider ${HADOOP_CREDENTIAL_TEP_PROVIDER_PATH} > /dev/null
        sed -i "s#{%hadoop.credential.property%}#${HADOOP_CREDENTIAL_PROPERTY}#g" `grep "{%hadoop.credential.property%}" -rl ./`
    else
        sed -i "s#{%hadoop.credential.property%}#""#g" `grep "{%hadoop.credential.property%}" -rl ./`
    fi
}

function update_credential_config_for_gcp() {
    sed -i "s#{%fs.gs.project.id%}#${GCP_PROJECT_ID}#g" `grep "{%fs.gs.project.id%}" -rl ./`

    sed -i "s#{%fs.gs.auth.service.account.email%}#${GCP_GCS_SERVICE_ACCOUNT_CLIENT_EMAIL}#g" `grep "{%fs.gs.auth.service.account.email%}" -rl ./`
    sed -i "s#{%fs.gs.auth.service.account.private.key.id%}#${GCP_GCS_SERVICE_ACCOUNT_PRIVATE_KEY_ID}#g" `grep "{%fs.gs.auth.service.account.private.key.id%}" -rl ./`

    if [ ! -z "${GCP_GCS_SERVICE_ACCOUNT_PRIVATE_KEY}" ]; then
        ${HADOOP_HOME}/bin/hadoop credential create fs.gs.auth.service.account.private.key -value ${GCP_GCS_SERVICE_ACCOUNT_PRIVATE_KEY} -provider ${HADOOP_CREDENTIAL_TEP_PROVIDER_PATH} > /dev/null
        sed -i "s#{%hadoop.credential.property%}#${HADOOP_CREDENTIAL_PROPERTY}#g" `grep "{%hadoop.credential.property%}" -rl ./`
    else
        sed -i "s#{%hadoop.credential.property%}#""#g" `grep "{%hadoop.credential.property%}" -rl ./`
    fi
}

function update_credential_config_for_azure() {
    sed -i "s#{%azure.storage.account%}#${AZURE_STORAGE_ACCOUNT}#g" "$(grep "{%azure.storage.account%}" -rl ./)"

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

    HAS_HADOOP_CREDENTIAL=false
    if [ ! -z "${AZURE_ACCOUNT_KEY}" ]; then
        FS_KEY_NAME_ACCOUNT_KEY="fs.azure.account.key.${AZURE_STORAGE_ACCOUNT}.${AZURE_ENDPOINT}.core.windows.net"
        ${HADOOP_HOME}/bin/hadoop credential create ${FS_KEY_NAME_ACCOUNT_KEY} -value ${AZURE_ACCOUNT_KEY} -provider ${HADOOP_CREDENTIAL_TEP_PROVIDER_PATH} > /dev/null
        HAS_HADOOP_CREDENTIAL=true
    fi

    if [ ! -z "${AZURE_MANAGED_IDENTITY_TENANT_ID}" ]; then
        FS_KEY_NAME_TENANT_ID="fs.azure.account.oauth2.msi.tenant"
        ${HADOOP_HOME}/bin/hadoop credential create ${FS_KEY_NAME_TENANT_ID} -value ${AZURE_MANAGED_IDENTITY_TENANT_ID} -provider ${HADOOP_CREDENTIAL_TEP_PROVIDER_PATH} > /dev/null
        HAS_HADOOP_CREDENTIAL=true
    fi

    if [ ! -z "${AZURE_MANAGED_IDENTITY_CLIENT_ID}" ]; then
        FS_KEY_NAME_CLIENT_ID="fs.azure.account.oauth2.client.id"
        ${HADOOP_HOME}/bin/hadoop credential create ${FS_KEY_NAME_CLIENT_ID} -value ${AZURE_MANAGED_IDENTITY_CLIENT_ID} -provider ${HADOOP_CREDENTIAL_TEP_PROVIDER_PATH} > /dev/null
        echo "${AZURE_MANAGED_IDENTITY_CLIENT_ID}" > ~/azure_managed_identity.config
        HAS_HADOOP_CREDENTIAL=true
    fi

    if [ "${HAS_HADOOP_CREDENTIAL}" == "true" ]; then
        sed -i "s#{%hadoop.credential.property%}#${HADOOP_CREDENTIAL_PROPERTY}#g" `grep "{%hadoop.credential.property%}" -rl ./`
    else
        sed -i "s#{%hadoop.credential.property%}#""#g" `grep "{%hadoop.credential.property%}" -rl ./`
    fi
}

function update_credential_config_for_provider() {
    HADOOP_CREDENTIAL_TMP_FILE="${output_dir}/credential.jceks"
    HADOOP_CREDENTIAL_TEP_PROVIDER_PATH="jceks://file@${HADOOP_CREDENTIAL_TMP_FILE}"
    if [ "${cloud_storage_provider}" == "aws" ]; then
        update_credential_config_for_aws
    elif [ "${cloud_storage_provider}" == "azure" ]; then
        update_credential_config_for_azure
    elif [ "${cloud_storage_provider}" == "gcp" ]; then
        update_credential_config_for_gcp
    fi
    if [  -f "$HADOOP_CREDENTIAL_TMP_FILE"  ]; then
        cp  ${HADOOP_CREDENTIAL_TMP_FILE} ${HADOOP_CREDENTIAL_FILE}
    fi
}
