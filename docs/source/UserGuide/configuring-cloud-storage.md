# Configuring Cloud Storage

- [Managed Cloud Storage](#managed-cloud-storage)
- [AWS S3](#aws-s3)
- [Azure Storage](#azure-storage)
- [Google GCS](#google-gcs)

## Managed Cloud Storage
To make storage management and configuration simple for user, CloudTik does two good things for you
When you are creating workspace for a specific cloud:
- CloudTik creates a managed cloud storage for you (S3 for AWS, Data Lake Storage Gen 2 for Azure, GCS for GCP) to use without any configurations.
- CloudTik creates roles to access your cloud storage in the account and the cluster instances are assigned with the roles for gaining access without any credential configurations.

These give great convenience for most of the use cases.
For users who need perform advanced configurations, CloudTik provide the flexibility to do so.

## AWS S3
By default, CloudTik will create a workspace managed S3 bucket for use out of box without any user configurations.
The following applies only when you want to create or use your own storage and configurations.

### Creating a S3 bucket
Every object in Amazon S3 is stored in a bucket. Before you can store data in Amazon S3, you must create a bucket.

Please refer to the S3 [Creating buckets](https://docs.aws.amazon.com/AmazonS3/latest/userguide/creating-bucket.html) for instructions.

### Configuring S3 in CloudTik
The name of S3 bucket will be used in CloudTik S3 storage configurations.

```
# Cloud-provider specific configuration.
provider:
    type: aws
    region: us-west-2
    # S3 configurations for storage
    aws_s3_storage:
        s3.bucket: your_s3_bucket
        s3.access.key.id: your_s3_access_key_id
        s3.secret.access.key: your_s3_secret_access_key

```

`s3.access.key.id`:  your AWS Access Key ID.

`s3.secret.access.key`:  your AWS Secret Access Key.

 *AWS Access Key ID* and *AWS Secret Access Key* can be found from the AWS guide of
[Managing access keys](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html).

## Azure Storage
By default, CloudTik will create a workspace managed storage account and a Data Lake Storage Gen 2 container
for use out of box without any user configurations. The following applies only when you want to create or use
your own storage and configurations.

### Creating Azure Storage
Azure **Blob storage** or **Data Lake Storage Gen 2** are both supported by CloudTik. For performance,
we suggest you to use Azure Data Lake Storage Gen 2.

If you want to create your own Azure storage account and a storage container within the storage account,
please refer to [Creating an Azure storage account](https://docs.microsoft.com/en-us/azure/storage/common/storage-account-create?tabs=azure-portal)
for instructions.

### Configuring Azure Storage in CloudTik
Storage account name and storage container name will be used when configuring Azure cluster yaml.

You will also need [Azure account access keys](https://docs.microsoft.com/en-us/azure/storage/common/storage-account-keys-manage?tabs=azure-portal#view-account-access-keys)
when configuring an Azure configuration yaml file, which grants the access to the created Azure storage.

You will be able to fill out the `azure_cloud_storage` for your cluster configuration yaml file.

```
# Cloud-provider specific configuration.
provider:
    type: azure
    location: westus
    subscription_id: your_subscription_id
    azure_cloud_storage:
        # Choose cloud storage type: blob (Azure Blob Storage) or datalake (Azure Data Lake Storage Gen 2).
        azure.storage.type: datalake
        azure.storage.account: your_storage_account
        azure.container: your_container
        azure.account.key: your_account_key

```

`subscription_id`: [Subscription ID](https://docs.microsoft.com/en-us/azure/azure-portal/get-subscription-tenant-id#find-your-azure-subscription)
of your Azure account.

`azure.storage.account`: Azure Storage Account name that you want CloudTik help to create.

`azure.container`: Azure Storage Container name that you have created.

`azure.account.key`: your [Azure account access keys](https://docs.microsoft.com/en-us/azure/storage/common/storage-account-keys-manage?tabs=azure-portal#view-account-access-keys).


## Google GCS
By default, CloudTik will create a workspace managed GCS bucket for use out of box without any user configurations.
The following applies only when you want to create or use your own storage and configurations.

### Creating GCS Bucket
If you want to use your own GCS bucket, you can create one by following the
[Creating buckets](https://cloud.google.com/storage/docs/creating-buckets#create_a_new_bucket).

### Configuring GCS in CloudTik
To control access to the bucket, please refer to [Google cloud buckets](../GettingStarted/gcs-bucket.md) for instructions. 
The name of bucket will be used when configuring GCP cluster yaml.

You will also need the previously downloaded Json file's `project_id`, `client_email`, `private_key_id` and 
`gcs.service.account.private.key` when configuring a GCP cluster yaml, which grants the access to the created GCP bucket.

You will be able to fill out the `gcp_cloud_storage` for your cluster configuration yaml file.

```
# Cloud-provider specific configuration.
provider:
    type: gcp
    region: us-central1
    availability_zone: us-central1-a
    project_id: your_project_id
    # GCS configurations for storage
    gcp_cloud_storage:
        gcs.bucket: your_gcs_bucket
        gcs.service.account.client.email: your_service_account_client_email
        gcs.service.account.private.key.id: your_service_account_private_key_id
        gcs.service.account.private.key: your_service_account_private_key

```
A JSON file should be safely downloaded and kept after a service account is created.

`project_id`: "project_id" in the JSON file.

`gcs.service.account.client.email `: "client_email" in the JSON file.

`gcs.service.account.private.key.id`: "private_key_id" in the JSON file.

`gcs.service.account.private.key`: "private_key" in the JSON file, 
in the format of `-----BEGIN PRIVATE KEY-----\n......\n-----END PRIVATE KEY-----\n`
