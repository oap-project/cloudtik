#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CLOUDTIK_HOME=$( cd -- "$( dirname -- "${SCRIPT_DIR}" )" &> /dev/null && pwd )
CONDA_HOME=$( cd -- "$( dirname -- "$( dirname -- "$(which conda)" )" )" &> /dev/null && pwd )

CLOUDTIK_BRANCH="main"

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    --branch)
        # Override for the branch.
        shift
        CLOUDTIK_BRANCH=$1
        ;;
    *)
        echo "Usage: release.sh [ --branch main ]"
        exit 1
    esac
    shift
done

source $CONDA_HOME/bin/activate cloudtik_py37 || conda create -n cloudtik_py37 -y python=3.7
source $CONDA_HOME/bin/activate cloudtik_py38 || conda create -n cloudtik_py38 -y python=3.8
source $CONDA_HOME/bin/activate cloudtik_py39 || conda create -n cloudtik_py39 -y python=3.9

cd $CLOUDTIK_HOME
git reset --hard
git pull
git checkout ${CLOUDTIK_BRANCH}

CLOUDTIK_VERSION=$(sed -n 's/__version__ = \"\(..*\)\"/\1/p' ./python/cloudtik/__init__.py)

# Remove all existing wheels
rm -rf ./python/dist/*.whl

arch=$(uname -m)

cp ./build/thirdparty/ganglia/modpython.so ./build/thirdparty/ganglia/modpython-${arch}.so
aws s3 cp ./build/thirdparty/ganglia/modpython-${arch}.so s3://cloudtik/downloads/ganglia/

source $CONDA_HOME/bin/activate cloudtik_py37
bash ./build.sh --no-build-redis --no-build-ganglia
CLOUDTIK_PY37_WHEEL=cloudtik-${CLOUDTIK_VERSION}-cp37-cp37m-manylinux2014_${arch}.whl
aws s3 cp ./python/dist/$CLOUDTIK_PY37_WHEEL s3://cloudtik/downloads/cloudtik/

source $CONDA_HOME/bin/activate cloudtik_py38
bash ./build.sh --no-build-redis --no-build-ganglia
CLOUDTIK_PY38_WHEEL=cloudtik-${CLOUDTIK_VERSION}-cp38-cp38-manylinux2014_${arch}.whl
aws s3 cp ./python/dist/$CLOUDTIK_PY38_WHEEL s3://cloudtik/downloads/cloudtik/

source $CONDA_HOME/bin/activate cloudtik_py39
bash ./build.sh --no-build-redis --no-build-ganglia
CLOUDTIK_PY39_WHEEL=cloudtik-${CLOUDTIK_VERSION}-cp39-cp39-manylinux2014_${arch}.whl
aws s3 cp ./python/dist/$CLOUDTIK_PY39_WHEEL s3://cloudtik/downloads/cloudtik/

aws cloudfront create-invalidation --distribution-id E3703BSG9BICN1 --paths "/downloads/cloudtik/cloudtik-${CLOUDTIK_VERSION}-cp37-cp37m-manylinux2014_${arch}.whl" "/downloads/cloudtik/cloudtik-${CLOUDTIK_VERSION}-cp38-cp38-manylinux2014_${arch}.whl" "/downloads/cloudtik/cloudtik-${CLOUDTIK_VERSION}-cp39-cp39-manylinux2014_${arch}.whl"


# upload pip
twine check ./python/dist/*
twine upload ./python/dist/*
