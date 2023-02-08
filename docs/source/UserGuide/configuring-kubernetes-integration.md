# Configuring Kubernetes Integration with Cloud Providers

- [Introduction](#introduction)
- [Integration with AWS EKS](#integration-with-aws-eks)
- [Integration with GCP GKE](#integration-with-gcp-gke)
- [Integration with Azure AKS](#integration-with-azure-aks)

## Introduction
CloudTik supports to run on a generic K8S cluster. For most of the public providers,
cloud provider usually offers managed Kubernetes cluster/engine
which gives a better integration with its cloud resources. This integration provides convenience
for accessing other cloud resources using the integrated credentials.
For example, to access the cloud storage from the managed Kubernetes pod without additional
configuration of access keys.

CloudTik supports such integration with popular managed Kubernetes services
such as AWS EKS, GCP GKE and Azure AKS. With the integration, the managed cloud storage
can be created for a Kubernetes workspace and the Kubernetes pod of the clusters in the workspace
gain access to the cloud storage without the need for any further credential configurations.

Most of these integrations designed based on mechanism of OIDC-based federation with the
cloud authentication and authorization infrastructure. This is usually called Web Identity
or Workload Identity. Although the implementations may vary in some aspects, the fundamental
mechanism is similar. It put some requirements to the managed Kubernetes cluster such as
enable the OIDC and the version of the Kubernetes engine.

This document gives some most important notes for the integration to work. 

## Integration with AWS EKS

### Requirements of EKS cluster
The integration needs OpenID Connect (OIDC) Identity Provider (IDP) to work.
OIDC is enabled by default by AWS when you create a EKS cluster.

For example, you can use the following command to create a EKS cluster,

```
eksctl create cluster --name example-eks --region us-west-1 --version 1.22 --vpc-private-subnets "subnet-xxx, subnet-xxx" --without-nodegroup
```

Refer to the [Amazon EKS documentation](https://docs.aws.amazon.com/eks/latest/userguide/enable-iam-roles-for-service-accounts.html)
for more information on the OIDC issuer URL for the EKS cluster.

### Configuring CloudTik Kubernetes for EKS
The configuration for the integration is added in the "cloud_provider" section
of Kubernetes workspace configuration file and cluster configuration file.

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

When configured with "cloud_provider" section, the workspace creation
will perform all the deployment and configuration steps including OIDC provider
federation with IAM.

Check [EKS Examples](https://github.com/oap-project/cloudtik/tree/main/example/cluster/kubernetes/eks)
folder for more completed examples.

### Limitations
#### Limitation 1: Fuse mount 
Fuse mount from S3 to local path doesn't work using the default credential
due to the limitation of s3fs implementation.

## Integration with GCP GKE

### Requirements of GKE cluster
GKE cluster has an OIDC issuer URL associated with it by default.

For Workload Identity integration to work,
you need to specify "--workload-pool=YOUR-PROJECT-ID.svc.id.goog" parameter when creating the GKE cluster,

```
gcloud container clusters create examle-gke --project=YOUR-PROJECT-ID --region=us-central1 --workload-pool=YOUR-PROJECT-ID.svc.id.goog --network=YOUR-GKE-VPC --subnetwork=YOUR-GKE-SUBNET
```

Refer to [Enable Workload Identity section](https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity#enable)
for more details.

When you create the nodepool, you need '--workload-metadata=GKE_METADATA' parameter for Workload Identity feature to work.
```
gcloud container node-pools create examle-gke-nodepool --cluster=examle-gke --workload-metadata=GKE_METADATA --machine-type=n2-standard-8 --project=YOUR-PROJECT-ID --region=us-central1 --zone=us-central1-a
```
Refer to [Migrate existing workloads to Workload Identity section](https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity#migrate_applications_to)
for more details.

### Configuring CloudTik Kubernetes for GKE
The configuration for the integration is added in the "cloud_provider" section
of Kubernetes workspace configuration file and cluster configuration file.

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

Check [GKE Examples](https://github.com/oap-project/cloudtik/tree/main/example/cluster/kubernetes/gke)
folder for more completed examples.

## Integration with Azure AKS

### Requirements of AKS cluster
Since workload identity on an Azure Kubernetes Service (AKS) is still in public preview,
you need to follow the "Install the aks-preview Azure CLI extension" step and
"Register the 'EnableWorkloadIdentityPreview' feature flag" step in
[Deploy and configure workload identity on AKS](https://learn.microsoft.com/en-us/azure/aks/workload-identity-deploy-cluster)
for using preview features.

And then create an AKS cluster with "--enable-oidc-issuer" and "--enable-workload-identity" parameters, for example

```
az group create --name myResourceGroup --location eastus

az aks create -g myResourceGroup -n myAKSCluster --node-count 1 --enable-oidc-issuer --enable-workload-identity --generate-ssh-keys
```

For information of creating a new AKS cluster with OIDC Issuer URL enabled or update an existing cluster,
follow the instructions in the [Azure Kubernetes Service (AKS) documentation](https://learn.microsoft.com/en-us/azure/aks/cluster-configuration#oidc-issuer).

### Configuring CloudTik Kubernetes for AKS
The configuration for the integration is added in the "cloud_provider" section
of Kubernetes workspace configuration file and cluster configuration file.

For example,
```
# Kubernetes provider specific configurations
provider:
    type: kubernetes

    # Cloud-provider specific configuration.
    cloud_provider:
        type: azure
        location: eastus
        subscription_id: your-subscription-id
        aks_resource_group: your-aks-resource-group
        aks_cluster_name: your-aks-cluster
        managed_cloud_storage: True
```

Check [AKS Examples](https://github.com/oap-project/cloudtik/tree/main/example/cluster/kubernetes/aks)
folder for more completed examples.

### Limitations
#### Limitation 1: Fuse mount 
Fuse mount from Azure Blob/DataLake storage to local path doesn't work using the user assigned identity
due to the limitation of blobfuse2 implementation.
