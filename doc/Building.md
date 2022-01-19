# Building CloudTik

## Building for Linux

Before you start to build wheels for CloudTik, we recommend you create a python environment (>= Python 3.7).

### 1. Install prerequisit package for building enviroment
We provide ```./dev/install-dev.sh``` to for installing build dependencies on Ubtuntu systems.
```
bash ./dev/install-dev.sh
```

### 2. Create a Python environment (>= Python 3.7)
We suggest you use Conda to manage Python environment. You can refer ```./dev/install-conda.sh``` if Conda installation is needed. Execute the following command to create a Python environment for building, replacing the environment name if you want.

```
conda create -n cloudtik -y python=3.7
conda activate cloudtik
```

### 3. Build CloudTik wheels with our provided script
Execute below command to start the build.
```
bash build.sh
```
Then under `cloudtik/python/dist` directory, you will find the `*.whl` which is your current specific python version's CloudTik wheel for Linux.
