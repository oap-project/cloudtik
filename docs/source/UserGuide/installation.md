# Installation

## Preparing Python Environment

CloudTik requires a Python environment on Linux. We recommend using Conda to manage Python environments and packages.

If you don't have Conda installed, please refer to `dev/install-conda.sh` to install Conda on Linux.

```
git clone https://github.com/oap-project/cloudtik.git && cd cloudtik
bash dev/install-conda.sh
```

Once Conda is installed, create an environment with a specific Python version as below.
CloudTik currently supports Python 3.7, 3.8, 3.9. Here we take Python 3.7 as an example.

```
conda create -n cloudtik -y python=3.7
conda activate cloudtik
```

## Installing CloudTik

The following table shows the installation links for latest CloudTik wheels of supported Python versions.
To install these wheels, use the following `pip` command and wheels on different cloud providers:

| Linux      | Installation                                                                                                                                       |
|:-----------|:---------------------------------------------------------------------------------------------------------------------------------------------------|
| Python 3.9 | `pip install -U "cloudtik[aws] @ https://d30257nes7d4fq.cloudfront.net/downloads/cloudtik/cloudtik-0.9.1-cp39-cp39-manylinux2014_x86_64.whl" `     |
| Python 3.8 | `pip install -U "cloudtik[aws] @ https://d30257nes7d4fq.cloudfront.net/downloads/cloudtik/cloudtik-0.9.1-cp38-cp38-manylinux2014_x86_64.whl" `     |
| Python 3.7 | `pip install -U "cloudtik[aws] @ https://d30257nes7d4fq.cloudfront.net/downloads/cloudtik/cloudtik-0.9.1-cp37-cp37m-manylinux2014_x86_64.whl" `    |

Replace `cloudtik[aws]` with `clouditk[azure]` or `cloudtik[gcp]` if you want to create clusters on Azure or GCP.
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
