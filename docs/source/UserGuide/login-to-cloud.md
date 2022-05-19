# Login to Cloud

- [AWS](#aws)
- [Azure](#azure)
- [GCP](#gcp)

## AWS

### AWS Account

Create an AWS account if you don't have one, then login to [AWS](https://console.aws.amazon.com/).

Please refer to related [AWS documentation](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/)
for instructions.

### Authentication to AWS CLI

First, install AWS CLI(command line interface). Please refer to
[AWS CLI Installation Guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) 
for detailed instructions.

After AWS CLI is installed, then you need configure AWS CLI about credentials. Please refer to 
[Configuring the AWS CLI](https://github.com/aws/aws-cli/tree/v2#getting-started) for detailed instructions.

The quickest way to configure is to run `aws configure` command as below, fill these items out then AWS CLI will 
be configured and authenticated. *AWS Access Key ID* and *AWS Secret Access Key* can be found from the AWS guide of
[managing access keys](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html).

```
$ aws configure
AWS Access Key ID [None]: ...
AWS Secret Access Key [None]: ...
Default region name [None]: ...
Default output format [None]:
```

### Creating a bucket

Every object in Amazon S3 is stored in a bucket. Before you can store data in Amazon S3, you must create a bucket.
Please refer to the S3 [guide](https://docs.aws.amazon.com/AmazonS3/latest/userguide/creating-bucket.html) for instructions.

Then you will be able to fill out the `aws_s3_storage` field for your cluster configuration yaml file.

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
[managing access keys](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html).

## Azure

### Azure Account

Create an Azure account if you don't have one, then login to [Microsoft Azure portal](https://portal.azure.com/) to get
[Subscription ID](https://docs.microsoft.com/en-us/azure/azure-portal/get-subscription-tenant-id#find-your-azure-subscription)
of your account.

Please refer to related [Azure documentation](https://docs.microsoft.com/en-us/learn/modules/create-an-azure-account/)
for instructions.

### Authentication to Azure CLI

First, install the Azure CLI (`pip install azure-cli azure-identity`) then login using (`az login`).

Then set the subscription to use from the command line (`az account set -s <subscription_id>`) on your working machine.

Then the Azure CLI is configured to manage resources on your Azure account.


### Configuring Cloud Storage

Create an Azure Storage Account if you don't have one.

Azure **Blob storage** or **Data Lake Storage Gen2** are both supported by CloudTik. Please refer to Azure related 
[guides](https://docs.microsoft.com/en-us/azure/storage/common/storage-account-create?tabs=azure-portal) for details.

Then you will be able to fill out the `azure_cloud_storage` for your cluster configuration yaml file.

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

`azure.storage.account`: Azure Storage Account name that you want CloudTik help to create.

`azure.container`: Azure Storage Container name that you have created.

`azure.account.key`: your [Azure account access keys](https://docs.microsoft.com/en-us/azure/storage/common/storage-account-keys-manage?tabs=azure-portal#view-account-access-keys).


## GCP

### Google Cloud Account

Created a Google Cloud account if you don't have one, then login to [GCP](https://console.cloud.google.com/).

Please refer to related [GCP documentation](https://cloud.google.com/apigee/docs/hybrid/v1.3/precog-gcpaccount)
for instructions.

### Creating a Google Cloud Project

Google Cloud projects form the basis for creating, enabling, and using all Google Cloud services.
Create a project within your Google Cloud account. 

Please refer to 
[Google Cloud Guide](https://cloud.google.com/resource-manager/docs/creating-managing-projects) for instructions.

### Authentication calls to Google Cloud APIs.

First, follow the [Google Cloud docs](https://cloud.google.com/docs/authentication/getting-started#creating_a_service_account) 
to create a service account on Google Cloud, A JSON file should be safely downloaded and kept by you after the
service account created.

Then set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable as described in
[GCP docs](https://cloud.google.com/docs/authentication/getting-started#setting_the_environment_variable) on your working machine.

### Configuring Cloud Storage

If you do not already have a GCS bucket, create one and configure its permission for your service account.
More details, please refer to [gcs bucket guide](../GettingStarted/gcs-bucket.md).

Then you will be able to fill out the `gcp_cloud_storage` for your cluster configuration yaml file.

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
