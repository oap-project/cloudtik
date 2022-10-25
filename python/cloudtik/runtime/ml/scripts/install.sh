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

function install_tools() {
    # Install necessary tools
    which cmake > /dev/null || sudo apt-get -qq update -y > /dev/null; sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install cmake -y > /dev/null
    which g++-9 > /dev/null || sudo apt-get -qq update -y > /dev/null; sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install g++-9 -y > /dev/null
}

function install_ml() {
    # Install Machine Learning libraries and components
    echo "Installing deep learning frameworks: tensorflow, torch, mxnet..."
    pip -qq install mxnet==1.9.1 tensorflow==2.9.1
    pip -qq install torch==1.12.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
    echo "Installing machine learning tools: mlflow, hyperopt..."
    pip -qq install mlflow==1.27.0 pyarrow==8.0.0 hyperopt==0.2.7 scikit-learn==1.0.2
    mkdir -p $RUNTIME_PATH/mlflow
    echo "Installing horovod..."
    export CXX=/usr/bin/g++-9&&HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_MXNET=1 HOROVOD_WITH_GLOO=1 pip -qq install horovod[all-frameworks]==0.25.0
}

install_tools
install_ml
