#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h::p: -l head:: -- "$@")
eval set -- "${args}"

IS_HEAD_NODE=false

while true
do
    case "$1" in
    --head)
        IS_HEAD_NODE=true
        ;;
    --)
        shift
        break
        ;;
    esac
    shift
done

export USER_HOME=/home/$(whoami)
export RUNTIME_PATH=$USER_HOME/runtime
mkdir -p $RUNTIME_PATH


function prepare_s3_fuse() {
    sudo apt-get update
    sudo apt install s3fs -y
}


function prepare_blob_fuse() {
    wget -N https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb
    sudo dpkg -i packages-microsoft-prod.deb
    sudo apt-get update
    sudo apt-get install blobfuse -y
}


function prepare_gcs_fuse() {
    sudo apt-get update
    sudo apt-get install -y curl
    echo "deb http://packages.cloud.google.com/apt gcsfuse-bionic main" |sudo tee /etc/apt/sources.list.d/gcsfuse.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    sudo apt-get install gcsfuse -y
}

function prepare_mount_tools() {
    cloud_storage_provider="none"
    if [ "$AWS_CLOUD_STORAGE" == "true" ]; then
        prepare_s3_fuse
    elif [ "$AZURE_CLOUD_STORAGE" == "true" ]; then
        prepare_blob_fuse
    elif [ "$GCP_CLOUD_STORAGE" == "true" ]; then
        prepare_gcs_fuse
    fi
}


function install_tools() {
    # Install necessary tools
    which cmake > /dev/null || sudo apt-get -qq update -y > /dev/null; sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install cmake -y > /dev/null
    which g++-9 > /dev/null || sudo apt-get -qq update -y > /dev/null; sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install g++-9 -y > /dev/null
}

function install_ml() {
    # Install Machine Learning libraries and components
    echo "Installing deep learning frameworks: tensorflow, pytorch, mxnet..."
    pip -qq install mxnet==1.9.1 tensorflow==2.9.3
    pip -qq install torch==1.12.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
    echo "Installing machine learning tools: mlflow, hyperopt..."
    pip -qq install mlflow==1.27.0 pyarrow==8.0.0 hyperopt==0.2.7 scikit-learn==1.0.2
    mkdir -p $RUNTIME_PATH/mlflow
    echo "Installing deep learning libraries for music and audio analysis..."
    pip -qq install librosa==0.9.2
    echo "Installing deep learning libraries for facial recognition..."
    pip -qq install opencv-python-headless==4.6.0.66 tensorflow-addons==0.17.1
    CLOUDTIK_CONDA_ENV=$(dirname $(dirname $(which cloudtik)))
    conda install dlib=19.24.0 libffi=3.3 -p $CLOUDTIK_CONDA_ENV -c conda-forge -y

    echo "Installing Open MPI..."
    which mpirun > /dev/null \
    || mkdir /tmp/openmpi \
    && PREV_CUR_DIR=$(pwd) \
    && cd /tmp/openmpi \
    && wget -q --show-progress https://www.open-mpi.org/software/ompi/v4.1/downloads/openmpi-4.1.4.tar.gz -O openmpi.tar.gz  \
    && tar --extract --file openmpi.tar.gz --directory /tmp/openmpi --strip-components 1 --no-same-owner \
    && sudo ./configure --enable-orterun-prefix-by-default CC=gcc-9 CXX=g++-9 > /dev/null \
    && sudo make -j $(nproc) all > /dev/null \
    && sudo make install > /dev/null \
    && sudo ldconfig \
    && cd ${PREV_CUR_DIR} \
    && sudo rm -rf /tmp/openmpi

    echo "Installing horovod..."
    export CXX=/usr/bin/g++-9 && HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_MXNET=1 HOROVOD_WITH_GLOO=1 HOROVOD_WITH_MPI=1 pip -qq install horovod[all-frameworks]==0.25.0
}

install_tools
install_ml
prepare_mount_tools