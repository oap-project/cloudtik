# Installation

## Preparing Python Environment

CloudTik requires a Python environment to run. We suggest you use Conda to manage Python environments and packages. If you don't have Conda , you can refer ```dev/install-conda.sh``` to install conda on Ubuntu systems.

```
git clone https://github.com/oap-project/cloudtik.git && cd cloudtik
bash dev/install-conda.sh
```

Once Conda is installed, create an environment specify a Python version as below. 
CloudTik currently supports Python 3.6, 3.7, 3.8, 3.9. Here we take Python 3.7 as example.

```
conda create -n cloudtik -y python=3.7;
conda activate cloudtik;
```

## Installing CloudTik from Daily Releases

### Daily Releases for different Python versions

You can install the latest CloudTik wheels via the following links. These daily releases do not go through the full release process. 
To install these wheels, use the following `pip` command and wheels on different Cloud providers:



| Linux      | Installation                                                                                                                                       |
|:-----------|:---------------------------------------------------------------------------------------------------------------------------------------------------|
| Python 3.9 | `pip install -U "cloudtik[aws] @ https://d30257nes7d4fq.cloudfront.net/downloads/cloudtik/cloudtik-0.9.0-cp39-cp39-manylinux2014_x86_64.whl" `     |
| Python 3.8 | `pip install -U "cloudtik[aws] @ https://d30257nes7d4fq.cloudfront.net/downloads/cloudtik/cloudtik-0.9.0-cp38-cp38-manylinux2014_x86_64.whl" `     |
| Python 3.7 | `pip install -U "cloudtik[aws] @ https://d30257nes7d4fq.cloudfront.net/downloads/cloudtik/cloudtik-0.9.0-cp37-cp37m-manylinux2014_x86_64.whl" `    |
| Python 3.6 | `pip install -U "cloudtik[aws] @ https://d30257nes7d4fq.cloudfront.net/downloads/cloudtik/cloudtik-0.9.0-cp36-cp36m-manylinux2014_x86_64.whl" `    |

Replace `cloudtik[aws]` with `clouditk[azure]` or `cloudtik[gcp]` if you want to create clusters on Azure or GCP.

Use `cloudtik[all]` if you want to manage clusters with all supported Cloud providers.

## Building CloudTik from Source and Installing

### Building for Linux

After create a Python environment as above, then build cloudtik wheel for Linux.

#### Building CloudTik wheels with our provided script

Run the following command to start the build.

```
git clone https://github.com/oap-project/cloudtik.git && cd cloudtik
bash build.sh
```
Then under `./python/dist` directory, you will find the `*.whl` which is your current Python version's CloudTik wheel for Linux.

#### Installing CloudTik 

Then install your built wheel above.

```
pip install ./python/dist/<your-built-wheel>.whl 
```