# Installation

## Preparing Python Environment

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

## Installing CloudTik
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

## Building CloudTik from Source and Installing

### Building CloudTik on Linux

After created a Python environment as above, you can build wheel for CloudTik on Linux.

Run the following command to start the building CloudTik with provided scripts.

```
git clone https://github.com/oap-project/cloudtik.git && cd cloudtik
bash build.sh
```

You will find the `*.whl` under `./python/dist` directory, which is your current Python version's CloudTik wheel for Linux.

### Installing CloudTik

Install your built wheel above.

```
pip install ./python/dist/<your-built-wheel>.whl 
```

If you want to install the built CloudTik above into the clusters to be created when running
`cloudtik start /path/to/<your-cluster-configuration>.yaml`, you need to upload the wheel to cloud or servers where can be visited via Internet. 

Add `cloudtik_wheel_url` to your cluster config yaml file as below.

```
workspace_name: ...

cluster_name: ...

cloudtik_wheel_url: "</link/to/cloudtik-*>.whl"

```
