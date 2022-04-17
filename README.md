# CloudTik

CloudTik is a cloud scaling platform for scaling your distributed analytics and AI cluster such as Spark easily
on public Cloud environment including AWS, Azure, GCP and so on. The CloudTik target is enable any users can
easily create and manage analytics and AI clusters, provide out of box optimized Spark runtime for
your Analytics and AI needs, and go quickly to focus on your workload and business need instead
of taking a lot of time constructing the cluster and platform. We target:
- Support major public Cloud providers (AWS, Azure and GCP, ...)
- Out of box and optimized Spark runtime for Analytics and AI
- Easy and unified operation experiences across Cloud
- Open architecture and user full control
- Runtime directly on VM or in Container
- A full open-sourced solution

## Getting Started with CloudTik
### 1. Prepare Python environment
CloudTik requires a Python environment to run. We suggest you use Conda to manage Python environments and packages. If you don't have Conda , you can refer ```dev/install-conda.sh``` to install conda on Ubuntu systems. 
```
bash dev/install-conda.sh;  ## Optional
```
Once Conda is installed, create an environment specify a python version as below.
```
conda create -n cloudtik -y python=3.7;
conda activate cloudtik;
```
### 2. Install CloudTik
Installation of CloudTik is simple. Execute the below pip commands to install CloudTik to the working machine
for specific cloud providers. For AWS example,

```
# if running CloudTik on aws
pip install -U "cloudtik[aws] @ http://23.95.96.95:8000/latest/cloudtik-0.9.0-cp37-cp37m-manylinux2014_x86_64.whl"
```
Replace "cloudtik[aws]" with "clouditk[azure]" or "cloudtik[gcp]" if you want to use Azure or GCP.
Use "cloudtik[all]" if you want to manage clusters with all supported Cloud providers.

### 3. Configure Credentials for Cloud Providers
You need to configure or log into your Cloud account to gain access to your cloud provider API.
#### If you use AWS
Please follow the instructions described in [the AWS docs](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html) for configuring AWS credentials needed to acccess AWS.
#### If you use Azure
Use "az login" to log into your Azure cloud account at the machine.
#### If you use GCP
Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable as described in [the GCP docs](https://cloud.google.com/docs/authentication/getting-started).

### 4 Create a workspace for your clusters.
CloudTik uses Workspace concept to manage your Cloud network and other resources. In a Workspace, you can start one or more clusters.
Use the following command to create and provision a Workspace:

```
cloudtik workspace create your-workspace-config.yaml
```
A typical workspace configuration file is usually very simple. Specific the unique workspace name, cloud provider type
and a few cloud provider specific properties. Take AWS for example,
```
# A unique identifier for the workspace.
workspace_name: example-workspace

# Cloud-provider specific configuration.
provider:
    type: aws
    region: us-west-2
    security_group:
        # Use IpPermissions to allow SSH access from your working node
        IpPermissions:
        - FromPort: 22
          ToPort: 22
          IpProtocol: TCP
          IpRanges:
          - CidrIp: 0.0.0.0/0
```
Check example/cluster folder for more Workspace configuration file examples.

### 5. Configure Cloud Storage
Running Spark on Cloud needs a Cloud storage to store staging and events data.
#### If you use AWS
#### If you use Azure
#### If you use GCP
If you do not already have a GCS bucket, create one and configure its permission for your service account.
More details, please refer to configure [gcs bucket guide](./doc/Configure-GCS-Bucket.md).

### 6. Start a cluster
Now you can start a cluster:
```
cloudtik start your-cluster-config.yaml
```
A typical cluster configuration file is usually very simple thanks to CloudTik hierarchy templates design. Take AWS
for example,
```
# An example of standard 1 + 3 nodes cluster with standard instance type
from: aws/standard

# Workspace into which to launch the cluster
workspace_name: exmaple-workspace

# A unique identifier for the cluster.
cluster_name: example-docker

# Enable container
docker:
    enabled: True

# Cloud-provider specific configuration.
provider:
    type: aws
    region: us-west-2
    # S3 configurations for storage
    aws_s3_storage:
        s3.bucket: your_s3_bucket
        s3.access.key.id: your_s3_access_key_id
        s3.secret.access.key: your_s3_secret_access_key

auth:
    ssh_user: ubuntu
    # Set proxy if you are in corporation network. For example,
    # ssh_proxy_command: "ncat --proxy-type socks5 --proxy your_proxy_host:your_proxy_port %h %p"

available_node_types:
    worker.default:
        # The minimum number of worker nodes to launch.
        min_workers: 3
```
You need the cloud storage access information in Step 5 and only a few additional key settings in the configuration file to launch a cluster.
Refer to example/cluster folder for more cluster configurations examples.

### 7. Manage the cluster
CloudTik provides very powerful capability to monitor and manage the cluster.

#### Show cluster status and information
Use the following commands to show various cluster information.
```
cloudtik status your-cluster-config.yaml
cloudtik info your-cluster-config.yaml
cloudtik head-ip your-cluster-config.yaml
cloudtik worker-ips your-cluster-config.yaml
cloudtik process-status your-cluster-config.yaml
cloudtik monitor your-cluster-config.yaml
```
#### Attach to the cluster head (or specific node)
```
cloudtik attach your-cluster-config.yaml
```
#### Execute commands on cluster head (or specified node or on all nodes)
```
cloudtik exec your-cluster-config.yaml
```
#### Submit a job to the cluster to run
```
cloudtik submit your-cluster-config.yaml your-job-file.(py|sh|scala)
```
#### Copy local files to cluster head (or to all nodes)
```
cloudtik rsync-up your-cluster-config.yaml [source] [target]
```
#### Copy file from cluster to local
```
cloudtik rsync-down your-cluster-config.yaml [source] [target]
```
#### Stop a cluster
```
cloudtik stop your-cluster-config.yaml
```
For more information as to the commands, you can use `cloudtik --help` or `cloudtik [command] --help` to get detailed instructions.
