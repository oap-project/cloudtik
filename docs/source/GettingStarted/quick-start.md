# Quick Start

### 1. Preparing Python environment

CloudTik requires a Python environment on Linux. We recommend using Conda to manage Python environments and packages.

If you don't have Conda installed, please refer to `dev/install-conda.sh` to install Conda on Linux.

```
git clone https://github.com/oap-project/cloudtik.git && cd cloudtik
bash dev/install-conda.sh
```

Once Conda is installed, create an environment with a specific Python version as below.
CloudTik currently supports Python 3.8 or above. Take Python 3.9 as an example,

```
conda create -n cloudtik -y python=3.9
conda activate cloudtik
```

### 2. Installing CloudTik

Execute the following `pip` commands to install CloudTik on your working machine for specific cloud providers. 

Take AWS for example,

```
pip install cloudtik[aws]
```

Replace `cloudtik[aws]` with `clouditk[azure]`, `cloudtik[gcp]`, `cloudtik[aliyun]`
if you want to create clusters on Azure, GCP, Alibaba Cloud respectively.
If you want to run on Kubernetes, install `cloudtik[kubernetes]`.
Or  `clouditk[eks]` or `cloudtik[gke]` if you are running on AWS EKS or GCP GKE cluster.
Use `cloudtik[all]` if you want to manage clusters with all supported Cloud providers.

Please refer to [User Guide: Installation](../UserGuide/installation.md) for the package links for other Python versions.

### 3. Authentication to Cloud Providers API

After CloudTik is installed on your working machine, you need to configure or log into your Cloud account to 
authenticate the cloud provider CLI on this machine.

#### AWS

First, install AWS CLI (command line interface) on your working machine. Please refer to
[Installing AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
for detailed instructions.

After AWS CLI is installed, you need to configure AWS CLI about credentials. The quickest way to configure it 
is to run `aws configure` command, and you can refer to
[Managing access keys](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html#Using_CreateAccessKey)
to get *AWS Access Key ID* and *AWS Secret Access Key*.

More details for AWS CLI can be found in [AWS CLI Getting Started](https://github.com/aws/aws-cli/tree/v2#getting-started).

#### Azure

After CloudTik is installed on your working machine, login to Azure using `az login`.
Refer to [Sign in with Azure CLI](https://docs.microsoft.com/en-us/cli/azure/authenticate-azure-cli) for more details.

#### GCP

If you use service account authentication, follow [Creating a service account](https://cloud.google.com/docs/authentication/getting-started#creating_a_service_account)
to create a service account on Google Cloud. 

A JSON file should be safely downloaded to your local computer, and then set the `GOOGLE_APPLICATION_CREDENTIALS` environment
variable as described in the [Setting the environment variable](https://cloud.google.com/docs/authentication/getting-started#setting_the_environment_variable)
on your working machine.

If you use user account authentication, refer to [User Guide: Login to Cloud](../UserGuide/login-to-cloud.md#gcp) for details.

#### Alibaba Cloud
The simple way to set up Alibaba Cloud credentials for CloudTik use is
to export the access key ID and access key secret of your cloud account:

export ALIBABA_CLOUD_ACCESS_KEY_ID=xxxxxxxxxxxxxxxxxxxxxxxx
export ALIBABA_CLOUD_ACCESS_KEY_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

For more options of Alibaba Cloud credentials configuration in CloudTik,
refer to [User Guide: Login to Cloud](../UserGuide/login-to-cloud.html#alibaba-cloud).

Note: please activate OSS through Alibaba Cloud Console before going to the next step.

#### Kubernetes
If you are running CloudTik on a generic Kubernetes cluster, the authentication setup is simple.
You just need to authenticate your kubectl at your working machine to be able to access the Kubernetes cluster.

If you are running cloud Kubernetes engine (such as AWS EKS, GCP GKE or Azure AKE)
with cloud integrations with access to cloud resources such as cloud storage,
you need both kubectl authentication to cloud Kubernetes cluster and cloud API credentials configuration above.

### 4. Creating a Workspace for Clusters.
Once you authenticated with your cloud provider, you can start to create a Workspace.

CloudTik uses **Workspace** concept to easily manage shared Cloud resources such as VPC network resources,
identity and role resources, firewall or security groups, and cloud storage resources.
By default, CloudTik will create a workspace managed cloud storage
(S3 for AWS, Data Lake Storage Gen 2 for Azure, GCS for GCP) for use without any user configurations.
Within a workspace, you can start one or more clusters with different combination of runtime services.

Create a configuration workspace yaml file to specify the unique workspace name, cloud provider type and a few cloud 
provider properties. 

Take AWS as an example,

```
# A unique identifier for the workspace.
workspace_name: example-workspace

# Cloud-provider specific configuration.
provider:
    type: aws
    region: us-west-2
    # Use allowed_ssh_sources to allow SSH access from your client machine
    allowed_ssh_sources:
      - 0.0.0.0/0
```
*NOTE:* `0.0.0.0/0` in `allowed_ssh_sources` will allow any IP addresses to connect to your cluster as long as it has the cluster private key.
For more security, you need to change from `0.0.0.0/0` to restricted CIDR ranges for your case.

Use the following command to create and provision a Workspace:

```
cloudtik workspace create /path/to/your-workspace-config.yaml
```

Check [Configuration Examples](https://github.com/oap-project/cloudtik/tree/main/example/cluster) folder for more Workspace configuration file examples
for AWS, Azure, GCP, Kubernetes (AWS EKS or GCP GKE).

If you encounter problems on creating a Workspace, a common cause is that your current login account
for the cloud doesn't have enough privileges to create some resources such as VPC, storages, public ip and so on.
Make sure your current account have enough privileges. An admin or owner role will give the latest chance to have
all these privileges.

### 5. Starting a cluster with default runtimes

Now you can start a cluster running Spark by default:

```
cloudtik start /path/to/your-cluster-config.yaml
```

A typical cluster configuration file is usually very simple thanks to design of CloudTik's templates with inheritance.

Take AWS for example,

```
# An example of standard 1 + 3 nodes cluster with standard instance type
from: aws/standard

# Workspace into which to launch the cluster
workspace_name: example-workspace

# A unique identifier for the cluster.
cluster_name: example

# Cloud-provider specific configuration.
provider:
    type: aws
    region: us-west-2

auth:
    ssh_user: ubuntu
    # Set proxy if you are in corporation network. For example,
    # ssh_proxy_command: "ncat --proxy-type socks5 --proxy your_proxy_host:your_proxy_port %h %p"

available_node_types:
    worker.default:
        # The minimum number of worker nodes to launch.
        min_workers: 3
```

This example can be found in CloudTik source code folder `example/cluster/aws/example-standard.yaml`.

It will start a cluster named "example" in workspace "example-workspace" with minimal of 3 worker nodes running
Ganglia and Spark runtime services by default.

You need only a few key settings in the configuration file to launch a Spark cluster.

As for `auth` above, please set proxy if your working node is using corporation network.

```
auth:
    ssh_user: ubuntu
    ssh_proxy_command: "ncat --proxy-type socks5 --proxy <your_proxy_host>:<your_proxy_port> %h %p"
```

The cluster key will be created automatically for AWS and GCP if not specified.
For Azure, you need to generate an RSA key pair manually (use `ssh-keygen -t rsa -b 4096` to generate a new ssh key pair).
and configure the public and private key as following,

```
auth:
    ssh_private_key: ~/.ssh/my_cluster_rsa_key
    ssh_public_key: ~/.ssh/my_cluster_rsa_key.pub
```
If you need different runtime components in the cluster,
in the cluster configuration file, you can set the runtime types. For example,
```
runtime:
    types: [ganglia, hdfs, spark, ml]
```
It will run a cluster with ganglia, hdfs, spark and ml runtimes.

Refer to `example/cluster` directory for more cluster configurations examples.

### 6. Running analytics and AI workloads

Once the cluster is started, you can run Spark analytics and AI workloads
which are designed to be distributed and large scale in nature.

Below provides the information of some basic examples to start with.
As to running optimized Spark and AI, you can refer to [Running Optimized Analytics with Spark](https://cloudtik.readthedocs.io/en/latest/UserGuide/running-optimized-ai.html)
and [Running Optimized AI](https://cloudtik.readthedocs.io/en/latest/UserGuide/running-optimized-ai.html) for more information.

#### Running spark PI example

Running a Spark job is very straight forward. Spark PI job for example,

```
cloudtik exec ./your-cluster-config.yaml "spark-submit --master yarn --deploy-mode cluster --name spark-pi --class org.apache.spark.examples.SparkPi --conf spark.yarn.submit.waitAppCompletion=false \$SPARK_HOME/examples/jars/spark-examples.jar 12345" --job-waiter=spark
```

Refer to [Run Spark PI Example](https://github.com/oap-project/cloudtik/tree/main/example/spark) for more details.

#### Running analytics benchmarks

CloudTik provides ready to use tools for running TPC-DS benchmark
on a CloudTik spark runtime cluster.

Refer to [Run TPC-DS performance benchmark for Spark](https://github.com/oap-project/cloudtik/tree/main/tools/benchmarks/spark)
for a detailed step-by-step guide.

#### Running machine learning and deep learning examples

CloudTik provides ready to run examples for demonstrating
how distributed machine learning and deep learning jobs can be implemented
in CloudTik Spark and ML runtime cluster.

Refer to [Distributed Machine Learning and Deep Learning Examples](https://github.com/oap-project/cloudtik/tree/main/example/ml)
for a detailed step-by-step guide.

#### Workflow examples
User can integrate CloudTik with external workflows using bash scripts or python
for running on-demand cluster and jobs.

Refer to [Workflow Integration Examples](https://github.com/oap-project/cloudtik/tree/main/example/workflows) for example scripts.

### 7. Managing clusters

CloudTik provides very powerful capability to monitor and manage the cluster.

#### Cluster status and information

Use the following commands to show various cluster information.

```
# Check cluster status with:
cloudtik status /path/to/your-cluster-config.yaml
# Show cluster summary information and useful links to connect to cluster web UI.
cloudtik info /path/to/your-cluster-config.yaml
cloudtik head-ip /path/to/your-cluster-config.yaml
cloudtik worker-ips /path/to/your-cluster-config.yaml
```
#### Attach to the cluster head (or specific node)

Connect to a terminal of cluster head node.

```
cloudtik attach /path/to/your-cluster-config.yaml
```

#### Execute and Submit Jobs

Execute a command via SSH on cluster head node or a specified node.

```
cloudtik exec /path/to/your-cluster-config.yaml [command]
```

#### Manage Files

Upload files or directories to cluster.

``` 
cloudtik rsync-up /path/to/your-cluster-config.yaml [source] [target]
```
  
Download files or directories from cluster.

```
cloudtik rsync-down /path/to/your-cluster-config.yaml [source] [target]
```

### 8. Tearing Down

#### Terminate a Cluster

Stop and delete the cluster.

```
cloudtik stop /path/to/your-cluster-config.yaml
```

#### Delete a Workspace

Delete the workspace and all the network resources within it.

```
cloudtik workspace delete /path/to/your-workspace-config.yaml
```
Be default, the managed cloud storage will not be deleted.
Add --delete-managed-storage option to force deletion of manged cloud storage.

For more information as to the commands, you can use `cloudtik --help` or `cloudtik [command] --help` to get detailed instructions.
