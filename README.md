# CloudTik

CloudTik is a cloud scaling infrastructure for scaling your distributed analytics and AI cluster such as Spark easily
on public Cloud environment including AWS, Azure, GCP and so on. The CloudTik target is enable any users can
easily create and manage analytics and AI clusters and go quickly to focus on workload and business need insteading
of taking a lot of time constructing the cluster and platform.


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
for specific cloud providers.

```
# if running CloudTik on aws
pip install -U "cloudtik[aws] @ http://23.95.96.95:8000/latest/cloudtik-0.9.0-cp37-cp37m-manylinux2014_x86_64.whl"

# if running CloudTik on azure
pip install -U "cloudtik[azure] @ http://23.95.96.95:8000/latest/cloudtik-0.9.0-cp37-cp37m-manylinux2014_x86_64.whl"

# if running CloudTik on gcp
pip install -U "cloudtik[gcp] @ http://23.95.96.95:8000/latest/cloudtik-0.9.0-cp37-cp37m-manylinux2014_x86_64.whl"

# if running CloudTik on k8s
pip install -U "cloudtik[k8s] @ http://23.95.96.95:8000/latest/cloudtik-0.9.0-cp37-cp37m-manylinux2014_x86_64.whl"

# if need support for all above
pip install -U "cloudtik[all] @ http://23.95.96.95:8000/latest/cloudtik-0.9.0-cp37-cp37m-manylinux2014_x86_64.whl"

```
for 
### 3. Configure Credentials for Cloud Providers

#### Configure for AWS
Please follow the instructions described in [the AWS docs](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html) for configuring AWS credentials needed to acccess AWS.

#### Configure for GCP

Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable as described in [the GCP docs](https://cloud.google.com/docs/authentication/getting-started).

### 4. Configure Cloud Storage Bucket on Cloud Providers

#### Configure GCS Bucket

If you do not already have a GCS bucket, create one and configure its permission for your service account.
More details, please refer to configure [gcs bucket guide](./doc/Configure-GCS-Bucket.md).

### 5. CloudTik Commands
```
# commands running on working node for handling a cluster
cloudtik up ./example/cluster/aws/example-minimal.yaml -y   # Create or up a  cluster.
cloudtik down ./example/cluster/aws/example-minimal.yaml -y # Tear down the cluster.

cloudtik attach ./example/cluster/aws/example-minimal.yaml    # Create or attach to a SSH session to the cluster.
cloudtik exec ./example/cluster/aws/example-minimal.yaml [command]   # Exec a command via SSH on a cloudtik cluster.
cloudtik submit ./example/cluster/aws/example-minimal.yaml  experiment.py  #  Uploads and runs a script on the specified cluster.

cloudtik rsync-up ./example/cluster/aws/example-minimal.yaml [source] [target]   # Upload file to head node.
cloudtik rsync-down ./example/cluster/aws/example-minimal.yaml [source] [target]   # Download file from head node.

cloudtik enable-proxy ./example/cluster/aws/example-minimal.yaml # Enable local SOCKS5 proxy to the cluster through SSH
cloudtik disable-proxy ./example/cluster/aws/example-minimal.yaml # Disable local SOCKS5 proxy to the cluster through SSH


# commands running on working node for information and status
cloudtik head-ip ./example/cluster/aws/example-minimal.yaml    # Get the ip of head node.
cloudtik worker-ips ./example/cluster/aws/example-minimal.yaml    # Get the ips of worker nodes.
cloudtik info ./example/cluster/aws/example-minimal.yaml # Show cluster summary information and useful links to use the cluster.
cloudtik status ./example/cluster/aws/example-minimal.yaml # Show cluster summary status.
cloudtik process-status ./example/cluster/aws/example-minimal.yaml # Show process status of cluster nodes.
cloudtik monitor ./example/cluster/aws/example-minimal.yaml # Tails the monitor logs of a cluster.


# commands running on working node for debug
cloudtik health-check ./example/cluster/aws/example-minimal.yaml   # Do cluster health check.
cloudtik debug-status ./example/cluster/aws/example-minimal.yaml   # Show debug status of cluster scaling.
cloudtik cluster-dump ./example/cluster/aws/example-minimal.yaml   # Get log data from one or more nodes.
cloudtik kill-node ./example/cluster/aws/example-minimal.yaml   # Kills a random node. For testing purposes only.


# workspace commands
cloudtik workspace   $workspace_config_file      # Create a Workspace on Cloud based on the workspace configuration file.
```
You can use the command `cloudtik --help` or `cloudtik up --help` to get detailed instructions.

Usually you can install CloudTik package directly through pip as above and don't need to build it from source code. If you are contributing to CloudTik, you can follow the instrucitons in [Building CloudTik](./doc/Building.md) for building. 

