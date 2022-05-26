# Login to Cloud

- [AWS](#aws)
- [Azure](#azure)
- [GCP](#gcp)

## AWS

### AWS Account

Create an AWS account if you don't have one, then login to [AWS](https://console.aws.amazon.com/).

Please refer to [Creating an AWS account](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/)
for instructions.

### Authentication to AWS CLI

First, install AWS CLI (command line interface) on your working machine. Please refer to
[Installing AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
for detailed instructions.

After AWS CLI is installed, you need to configure AWS CLI about credentials. The quickest way to configure it 
is to run `aws configure` command, and you can refer to
[Managing access keys](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html#Using_CreateAccessKey)
to get *AWS Access Key ID* and *AWS Secret Access Key*.

More details for AWS CLI can be found in [AWS CLI getting started](https://github.com/aws/aws-cli/tree/v2#getting-started).

### Creating a S3 bucket

Every object in Amazon S3 is stored in a bucket. Before you can store data in Amazon S3, you must create a bucket.

Please refer to the S3 [Creating buckets](https://docs.aws.amazon.com/AmazonS3/latest/userguide/creating-bucket.html) for instructions.
The name of S3 bucket will be used in the next step.

You will be able to fill out the `aws_s3_storage` for your AWS cluster configuration yaml file, which is introduced
at [Quick start](../GettingStarted/quick-start.md) **step 6. Starting a cluster**.

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

## Azure

### Azure Account

Create an Azure account if you don't have one, then login to [Microsoft Azure portal](https://portal.azure.com/) to get
[Subscription ID](https://docs.microsoft.com/en-us/azure/azure-portal/get-subscription-tenant-id#find-your-azure-subscription)
of your account.

Please refer to [Creating an Azure account](https://docs.microsoft.com/en-us/learn/modules/create-an-azure-account/)
for instructions.

### Authentication to Azure CLI

After CloudTik is installed on your working machine, login to Azure using `az login`.
Refer to [Signing in with Azure CLI](https://docs.microsoft.com/en-us/cli/azure/authenticate-azure-cli) for more details.

### Configuring Cloud Storage

Create an Azure storage account and a storage container within this storage account.
Please refer to [Creating an Azure storage account](https://docs.microsoft.com/en-us/azure/storage/common/storage-account-create?tabs=azure-portal)
for instructions.

Azure **Blob storage** or **Data Lake Storage Gen2** are both supported by CloudTik. Storage account name
and storage container name will be used when configuring Azure cluster yaml.

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


## GCP

### Google Cloud Account

Created a Google Cloud account if you don't have one, then login to [GCP](https://console.cloud.google.com/).

Please refer to [Creating a GCP account](https://cloud.google.com/apigee/docs/hybrid/v1.3/precog-gcpaccount)
for instructions.

### Creating a Google Cloud Project

Google Cloud projects form the basis for creating, enabling, and using all Google Cloud services.
Create a project within your Google Cloud account. 

Please refer to [Creating projects](https://cloud.google.com/resource-manager/docs/creating-managing-projects) for instructions.

### Authentication calls to Google Cloud APIs.

First, follow [Creating a service account](https://cloud.google.com/docs/authentication/getting-started#creating_a_service_account)
to create a service account on Google Cloud. 

A JSON file should be safely downloaded to your local computer, and then set the `GOOGLE_APPLICATION_CREDENTIALS` environment
variable as described in the [Setting the environment variable](https://cloud.google.com/docs/authentication/getting-started#setting_the_environment_variable)
on your working machine.

### Configuring Cloud Storage

If you do not already have a GCS bucket, create one by following the 
[Creating buckets](https://cloud.google.com/storage/docs/creating-buckets#create_a_new_bucket).

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
