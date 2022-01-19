# CloudTik

CloudTik is a cloud scaling infrastructure for scaling your distributed analytics and AI cluster such as Spark easily
on public Cloud environment including AWS, Azure, GCP and so on. The CloudTik target is enable any users can
easily create and manage analytics and AI clusters and go quickly to focus on workload and business need insteading
of taking a lot of time constructing the cluster and platform.


## Getting Started with CloudTik
### 1. Pepare Python environment
CloudTik requires a Python envornment to run. We suggest you use Conda to manage Python environments and pakages. If you don't have Conda , you can refer ```dev/install-conda.sh``` to install conda on Ubuntu systems. 
```
bash dev/install-conda.sh;  ## Optional
```
Once Conda is installed, create an environment specify a python version as below.
```
conda create -n cloudtik -y python=3.7;
conda activate cloudtik;
```
### 2. Install CloudTik
Installation of CloudTik is simple. Execute the below pip commands to install CloudTik to the working machine.
```
pip install -U http://23.95.96.95:8000/latest/cloudtik-0.9.0-cp37-cp37m-manylinux2014_x86_64.whl
```
### 3. Configure Credentials for Cloud Providers

#### Configure for AWS
Please follow the instructions described in [the AWS docs](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html) for configuring AWS credentials needed to acccess AWS.

### 4. Use CloudTik to manage Spark clusters
```
cloudtik up ./example/aws/example-minimal.yaml -y   # Create or up a  cluster.
cloudtik get-head-ip ./example/aws/example-minimal.yaml    # Get the ip of head node.
cloudtik get-worker-ips ./example/aws/example-minimal.yaml    # Get the ips of worker nodes.
cloudtik exec ./example/aws/example-minimal.yaml [command]   # Exec a command via SSH on a cloudtik cluster.
cloudtik rsync-down ./example/aws/example-minimal.yaml [source] [target]   # Download file from head node.
cloudtik rsync-up ./example/aws/example-minimal.yaml [source] [target]   # Upload file to head node.
cloudtik attach ./example/aws/example-minimal.yaml    # Create or attach to a SSH session to the cluster.
cloudtik down ./example/aws/example-minimal.yaml -y # Tear down the cluster.
```

## Building CloudTik

Usually you can install CloudTik package directly through pip as above and don't need to build it from source code. If you are contributing to CloudTik, you can follow the instrucitons in [Building CloudTik](./doc/Building.md) for building. 

