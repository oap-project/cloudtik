# CloudTik

CloudTik is a cloud scaling infrastructure for scaling your distributed analytics and AI cluster such as Spark easily
on public Cloud environment including AWS, Azure, GCP and so on. The CloudTik target is enable any users can
easily create and manage analytics and AI clusters and go quickly to focus on workload and business need insteading
of taking a lot of time constructing the cluster and platform.

## Building CloudTik

Before you start to build wheels for CloudTik, we recommend you create a python environment (>= Python 3.7). We provide ```./dev/install-dev.sh``` to easily setup building environment for you.

### 1. Install prerequisit package for building enviroment:
```
bash ./dev/install-dev.sh
```

### 2. Create a python environment (>= Python 3.7):
```
conda create -n cloudtik-py37 -y python=3.7
conda activate cloudtik-py37
```

### 3. Build CloudTik wheels with our provided script.
```
bash build.sh
```
Then under `cloudtik/python/dist` directory, you will find the `*.whl` which is your current specific python version's CloudTik wheel for Linux.


## Using CloudTik on AWS
### 1. Create running env for working node(if the working node not contains conda, please use ```dev/install-conda.sh``` to install conda):
```
bash dev/install-conda.sh;  ## Opitional
conda create -n cloudtik-py37 -y python=3.7;
conda activate cloudtik-py37;
```
### 2. Install Cloudtik libraries:
```
pip install -U http://23.95.96.95:8000/latest/cloudtik-0.9.0-cp37-cp37m-manylinux2014_x86_64.whl
```
### 3. Configure your credentials in ~/.aws/credentials as described in [the AWS docs](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html).

### 4. Use Cloudtik to manager the cluster.
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


## Contributing to CloudTik