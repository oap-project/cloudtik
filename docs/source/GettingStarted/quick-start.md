# Quick Start

### 1. Preparing Python environment

CloudTik requires a Python environment to run. We suggest you use Conda to manage Python environments and packages. If you don't have Conda , you can refer ```dev/install-conda.sh``` to install conda on Ubuntu systems. 
```
git clone https://github.com/oap-project/cloudtik.git && cd cloudtik
bash dev/install-conda.sh
```
Once Conda is installed, create an environment specify a python version as below.
```
conda create -n cloudtik -y python=3.7;
conda activate cloudtik;
```
### 2. Installing CloudTik

Installation of CloudTik is simple. Execute the below pip commands to install CloudTik to the working machine
for specific cloud providers. For AWS example,

```
# if running CloudTik on aws
pip install -U "cloudtik[aws] @ https://d30257nes7d4fq.cloudfront.net/downloads/cloudtik/cloudtik-0.9.0-cp37-cp37m-manylinux2014_x86_64.whl"
```
Replace "cloudtik[aws]" with "clouditk[azure]" or "cloudtik[gcp]" if you want to use Azure or GCP.
Use "cloudtik[all]" if you want to manage clusters with all supported Cloud providers.

### 3. Credentials for Cloud Providers

You need to configure or log into your Cloud account to gain access to your cloud provider API.

#### AWS

Please follow the instructions described in [the AWS docs](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html) for configuring AWS credentials needed to acccess AWS.

#### Azure

Use "az login" to log into your Azure cloud account at the machine.

#### GCP

Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable as described in [the GCP docs](https://cloud.google.com/docs/authentication/getting-started).

### 4. Creating a Workspace for Clusters.

CloudTik uses Workspace concept to manage your Cloud network and other resources. In a Workspace, you can start one or more clusters.
Use the following command to create and provision a Workspace:

```
cloudtik workspace create your-workspace-config.yaml
```
A typical workspace configuration file is usually very simple. Specific the unique workspace name, cloud provider type
and a few cloud provider specific properties. Take AWS as example,
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

### 5. Configuring Cloud Storage

Running Spark on Cloud needs a Cloud storage to store staging and events data.

#### AWS

#### Azure

#### GCP

If you do not already have a GCS bucket, create one and configure its permission for your service account.
More details, please refer to [gcs bucket guide](gcs-bucket.md).

### 6. Starting a cluster

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
workspace_name: example-workspace

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

### 7. Managing clusters

CloudTik provides very powerful capability to monitor and manage the cluster.

#### Cluster status and information

Use the following commands to show various cluster information.

```
cloudtik status /path/to/your-cluster-config.yaml
cloudtik info /path/to/your-cluster-config.yaml
cloudtik head-ip /path/to/your-cluster-config.yaml
cloudtik worker-ips /path/to/your-cluster-config.yaml
cloudtik process-status /path/to/your-cluster-config.yaml
cloudtik monitor /path/to/your-cluster-config.yaml
```
#### Attach to the cluster head (or specific node)

```
cloudtik attach your-cluster-config.yaml
```
#### Execute and Submit Jobs

```
cloudtik exec /path/to/your-cluster-config.yaml
```

#### Manage Files

###### Copy local files to cluster head (or to all nodes)

```
cloudtik rsync-up /path/to/your-cluster-config.yaml [source] [target]
```

###### Copyfile from cluster to local
```
cloudtik rsync-down /path/to/your-cluster-config.yaml [source] [target]
```

#### Start or Stop Cluster

###### Start a cluster

```
cloudtik start /path/to/your-cluster-config.yaml -y
```

###### Stop a cluster

```
cloudtik stop your-cluster-config.yaml -y
```

For more information as to the commands, you can use `cloudtik --help` or `cloudtik [command] --help` to get detailed instructions.
