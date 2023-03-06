# Login to Cloud

- [AWS](#aws)
- [Azure](#azure)
- [GCP](#gcp)
- [Alibaba Cloud](#alibaba-cloud)

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

## GCP

### Google Cloud Account

Created a Google Cloud account if you don't have one, then login to [GCP](https://console.cloud.google.com/).

Please refer to [Creating a GCP account](https://cloud.google.com/apigee/docs/hybrid/v1.3/precog-gcpaccount)
for instructions.

### Creating a Google Cloud Project

Google Cloud projects form the basis for creating, enabling, and using all Google Cloud services.
Create a project within your Google Cloud account. 

Please refer to [Creating projects](https://cloud.google.com/resource-manager/docs/creating-managing-projects) for instructions.

### Authentication calls to Google Cloud APIs
User have two options to authenticate to Google Cloud.
- Authenticate with service account
- Authenticate with user account

Please refer to [Authentication Principal](https://cloud.google.com/docs/authentication#principal)
for detailed information as to these two methods.

#### Authenticate with Service Account
First, follow [Creating a service account](https://cloud.google.com/docs/authentication/getting-started#creating_a_service_account)
to create a service account on Google Cloud. 

To use the service account through API, you need a service account key. Refer to [Create and manage service account keys](https://cloud.google.com/iam/docs/creating-managing-service-account-keys) for details.

A JSON key file should be safely downloaded to your local computer, and then set the `GOOGLE_APPLICATION_CREDENTIALS` environment
variable as described in the [Setting the environment variable](https://cloud.google.com/docs/authentication/getting-started#setting_the_environment_variable)
on your working machine.

#### Authenticate with user account
If you have a user account with the right permissions,
You can authenticate using gcloud command. After you have authenticated,
You need to configure credentials in the workspace configuration file
or cluster configuration file with OAuth token information.

After you authenticated using gcloud, you can find a file named "adc.json"
in ~/.config/gcloud/legacy_credentials/your_account_name folder.
Using the information from this file, you can configure workspace configuration file
or cluster configuration file as following:

```
# Cloud-provider specific configuration.
provider:
    type: gcp
    region: us-central1
    availability_zone: us-central1-a
    project_id: your_project_id
    # Use allowed_ssh_sources to allow SSH access from your client machine
    allowed_ssh_sources:
      - 0.0.0.0/0
    gcp_credentials:
        type: oauth_token
        credentials:
            token: nil
            client_id: "your_client_id"
            client_secret: "your_client_secret"
            token_uri: https://oauth2.googleapis.com/token
            refresh_token: "your_refresh_token"
```

## Alibaba Cloud

CloudTik will try a few credentials options and use them automatically
if one of them is found.
- Environment variables for access key id and secret
- CloudTik configuration file
- Alibaba Cloud API Configuration file

Please note that Aliyun CLI login credentials will not be used by CloudTik.

### Environment variables
The simple way to set up Alibaba Cloud credentials for CloudTik use is
to export the access key ID and access key secret of your cloud account:

export ALIBABA_CLOUD_ACCESS_KEY_ID=xxxxxxxxxxxxxxxxxxxxxxxx
export ALIBABA_CLOUD_ACCESS_KEY_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

### CloudTik configuration file
You can specify the access key id and secret in CloudTik workspace or cluster configuration file.

For example,
```
# Cloud-provider specific configuration.
provider:
    type: aliyun
    region: cn-shanghai
    aliyun_credentials:
        aliyun_access_key_id: your-access-key-id
        aliyun_access_key_secret: your-access-key-secret
```

or with the optional security token for STS authentication,
```
# Cloud-provider specific configuration.
provider:
    type: aliyun
    region: cn-shanghai
    aliyun_credentials:
        aliyun_access_key_id: your-access-key-id
        aliyun_access_key_secret: your-access-key-secret
        aliyun_security_token: your-security-token
```

### Alibaba Cloud API Configuration file
You can also use a configuration file to cover many other cases.

Please refer to [Configure credentials: Configuration file](https://www.alibabacloud.com/help/en/alibaba-cloud-sdk-262060/latest/configure-credentials-378659)
section for more details.

#### Kubernetes
If you are running CloudTik on a generic Kubernetes cluster, the authentication setup is simple.
You just need to authenticate your kubectl at your working machine to be able to access the Kubernetes cluster.

##### AWS EKS
If you are running CloudTik on AWS EKS, CloudTik has more integration with AWS EKS
so that your CloudTik cluster running on EKS can access the S3 storage with IAM CloudTik workspace IAM roles.

You need not only to authenticate your kubectl at your working machine to be able to access the Kubernetes cluster,
but also setup your AWS credentials following the steps in [AWS](#aws) section above.

Instead of setting the cloud provider configurations at 'provider" section,
for EKS, you set the cloud provider configurations at 'cloud_provider' section under 'provider' configuration key,
For example,

```
# Kubernetes provider specific configurations
provider:
    type: kubernetes

    # Cloud-provider specific configuration.
    cloud_provider:
        type: aws
        region: us-west-2
        eks_cluster_name: your-eks-cluster
        managed_cloud_storage: True
```

##### GCP GKE
If you are running CloudTik on GCP GKE, CloudTik has more integration with GCP GKE
so that your CloudTik cluster running on GKE can access the GCS storage with IAM CloudTik workspace roles.

You need not only to authenticate your kubectl at your working machine to be able to access the Kubernetes cluster,
but also setup your GCP credentials following the steps in [GCP](#gcp) section above.

Instead of setting the cloud provider configurations at 'provider" section,
for GKE, you set the cloud provider configurations at 'cloud_provider' section under 'provider' configuration key,
For example,

```
# Kubernetes provider specific configurations
provider:
    type: kubernetes

    # Cloud-provider specific configuration.
    cloud_provider:
        type: gcp
        region: us-central1
        project_id: your_gcp_project_id
        managed_cloud_storage: True
```
