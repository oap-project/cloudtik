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

### 5. Use CloudTik to manage Spark clusters
```
cloudtik up ./example/cluster/aws/example-minimal.yaml -y   # Create or up a  cluster.
cloudtik get-head-ip ./example/cluster/aws/example-minimal.yaml    # Get the ip of head node.
cloudtik get-worker-ips ./example/cluster/aws/example-minimal.yaml    # Get the ips of worker nodes.
cloudtik exec ./example/cluster/aws/example-minimal.yaml [command]   # Exec a command via SSH on a cloudtik cluster.
cloudtik rsync-down ./example/cluster/aws/example-minimal.yaml [source] [target]   # Download file from head node.
cloudtik rsync-up ./example/cluster/aws/example-minimal.yaml [source] [target]   # Upload file to head node.
cloudtik attach ./example/cluster/aws/example-minimal.yaml    # Create or attach to a SSH session to the cluster.
cloudtik down ./example/cluster/aws/example-minimal.yaml -y # Tear down the cluster.
```

## Building CloudTik

Usually you can install CloudTik package directly through pip as above and don't need to build it from source code. If you are contributing to CloudTik, you can follow the instrucitons in [Building CloudTik](./doc/Building.md) for building. 

