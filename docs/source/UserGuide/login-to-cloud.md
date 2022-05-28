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

### Authentication calls to Google Cloud APIs.

First, follow [Creating a service account](https://cloud.google.com/docs/authentication/getting-started#creating_a_service_account)
to create a service account on Google Cloud. 

A JSON file should be safely downloaded to your local computer, and then set the `GOOGLE_APPLICATION_CREDENTIALS` environment
variable as described in the [Setting the environment variable](https://cloud.google.com/docs/authentication/getting-started#setting_the_environment_variable)
on your working machine.

