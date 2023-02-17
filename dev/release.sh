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

INVALIDATION_PATHS=""
for PYTHON_VERSION in "3.8" "3.9" "3.10" "3.11"
do
    echo "Building CloudTik for Python ${PYTHON_VERSION}...."
    PYTHON_TAG=${PYTHON_VERSION//./}
    source $CONDA_HOME/bin/activate cloudtik_py${PYTHON_TAG} || conda create -n cloudtik_py${PYTHON_TAG} -y python=${PYTHON_VERSION}
    source $CONDA_HOME/bin/activate cloudtik_py${PYTHON_TAG}
    bash ./build.sh --no-build-redis --no-build-ganglia
    CLOUDTIK_WHEEL=cloudtik-${CLOUDTIK_VERSION}-cp${PYTHON_TAG}-cp${PYTHON_TAG}-manylinux2014_${arch}.whl
    aws s3 cp ./python/dist/$CLOUDTIK_WHEEL s3://cloudtik/downloads/cloudtik/
    INVALIDATION_PATHS="${INVALIDATION_PATHS} /downloads/cloudtik/${CLOUDTIK_WHEEL}"
done

aws cloudfront create-invalidation --distribution-id E3703BSG9BICN1 --paths ${INVALIDATION_PATHS}
