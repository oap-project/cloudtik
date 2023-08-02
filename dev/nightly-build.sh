#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CLOUDTIK_HOME=$( cd -- "$( dirname -- "${SCRIPT_DIR}" )" &> /dev/null && pwd )
CONDA_HOME=$( cd -- "$( dirname -- "$( dirname -- "$(which conda)" )" )" &> /dev/null && pwd )

# Import the default vars
. "$SCRIPT_DIR"/set-default-vars.sh

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
    --no-pull)
        NO_PULL=YES
        ;;
    --copy)
        # Copy the nightly to a directory
        shift
        COPY_DIR=$1
        ;;
    *)
        echo "Usage: build.sh [ --branch main ] [ --copy copy_to_dir]"
        exit 1
    esac
    shift
done

PYTHON_TAG=${PYTHON_VERSION//./}

cd $CLOUDTIK_HOME

if [ ! $NO_PULL ]; then
    git reset --hard
    git checkout ${CLOUDTIK_BRANCH}
    git pull
fi

CLOUDTIK_VERSION=$(sed -n 's/__version__ = \"\(..*\)\"/\1/p' ./python/cloudtik/__init__.py)

source $CONDA_HOME/bin/activate cloudtik_py${PYTHON_TAG} || conda create -n cloudtik_py${PYTHON_TAG} -y python=${PYTHON_VERSION}
source $CONDA_HOME/bin/activate cloudtik_py${PYTHON_TAG}
bash ./build.sh --no-build-redis

arch=$(uname -m)

CLOUDTIK_WHEEL=cloudtik-${CLOUDTIK_VERSION}-cp${PYTHON_TAG}-cp${PYTHON_TAG}-manylinux2014_${arch}.whl
CLOUDTIK_WHEEL_NIGHTLY=cloudtik-${CLOUDTIK_VERSION}-cp${PYTHON_TAG}-cp${PYTHON_TAG}-manylinux2014_${arch}.nightly.whl

cp ./python/dist/$CLOUDTIK_WHEEL ./python/dist/$CLOUDTIK_WHEEL_NIGHTLY

if [ "${COPY_DIR}" != "" ];then
    cp ./python/dist/$CLOUDTIK_WHEEL_NIGHTLY ${COPY_DIR}/$CLOUDTIK_WHEEL_NIGHTLY
    echo "Copied to ${COPY_DIR}/$CLOUDTIK_WHEEL_NIGHTLY"
fi
