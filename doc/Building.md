# Building 

## Building for Linux

Before you start to build wheels for CloudTik, we recommend you create a python environment (>= Python 3.7)

If with conda, run
```
conda create -n cloudtik-py37 -y python=3.7
conda activate cloudtik-py37
```

Then build CloudTik wheels with our provided script.
```
git clone https://github.com/Intel-bigdata/cloudtik.git
cd cloudtik/python
bash build-wheel-manylinux2014.sh
```
Then under `cloudtik/python/dist` directory, you will find the `*.whl` which is your current specific python version's CloudTik wheel for Linux.
