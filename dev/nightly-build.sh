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
        echo "Usage: build.sh [ --branch main ]"
        exit 1
    esac
    shift
done


cd $CLOUDTIK_HOME
git reset --hard
git checkout ${CLOUDTIK_BRANCH}
git pull

CLOUDTIK_VERSION=$(sed -n 's/__version__ = \"\(..*\)\"/\1/p' ./python/cloudtik/__init__.py)

source $CONDA_HOME/bin/activate cloudtik_py37
bash ./build.sh --no-build-redis --no-build-ganglia

arch=$(uname -m)

CLOUDTIK_PY37_WHEEL=cloudtik-${CLOUDTIK_VERSION}-cp37-cp37m-manylinux2014_${arch}.whl
CLOUDTIK_PY37_WHEEL_NIGHTLY=cloudtik-${CLOUDTIK_VERSION}-cp37-cp37m-manylinux2014_${arch}.nightly.whl

cp ./python/dist/$CLOUDTIK_PY37_WHEEL ./python/dist/$CLOUDTIK_PY37_WHEEL_NIGHTLY
